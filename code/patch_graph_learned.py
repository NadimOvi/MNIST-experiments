import argparse
import csv
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

from patch_graph_timing import patch_features, global_knn, recall
from patch_graph_denoising import (window_candidates, nndescent_knn, from_table,
                                   from_knn, best_denoise, psnr)
from patch_graph_benchmark import load_image, noisy_dist, neighbour_error, pca_knn, IMAGES

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / "cache"
TRAIN_IMAGES = ["rocket", "moon", "grass", "gravel", "hubble_deep_field",
                "immunohistochemistry", "cell", "clock"]


def timed(fn, *args):
    t = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t


def train_model(a):
    key = f"learned_in{a.in_patch}_p{a.patch}_d{a.dim}_n{a.samples}_h{a.hidden}.joblib"
    path = CACHE / key
    if path.exists() and not a.retrain:
        print(f"Loading trained model from cache/{key}")
        return joblib.load(path)

    rng = np.random.default_rng(123)
    X, C = [], []
    per_copy = a.samples // (len(TRAIN_IMAGES) * a.copies)
    for name in TRAIN_IMAGES:
        clean = load_image(name, a.size)
        Fc = patch_features(clean, a.patch)
        for _ in range(a.copies):
            sigma = rng.uniform(0.05, 0.25)
            noisy = (clean + sigma * rng.standard_normal(clean.shape)).astype(np.float32)
            idx = rng.choice(len(Fc), per_copy, replace=False)
            X.append(patch_features(noisy, a.in_patch)[idx])
            C.append(Fc[idx])
    X, C = np.concatenate(X), np.concatenate(C)

    pca = PCA(n_components=a.dim, random_state=0).fit(C)
    Y = pca.transform(C)
    print(f"Training on {len(X)} patches from {len(TRAIN_IMAGES)} images "
          f"(input {a.in_patch}x{a.in_patch}, code {a.dim} numbers) ...")
    t = time.perf_counter()
    net = MLPRegressor(hidden_layer_sizes=(a.hidden, a.hidden), batch_size=512,
                       learning_rate_init=1e-3, max_iter=a.epochs, early_stopping=True,
                       n_iter_no_change=5, random_state=0)
    net.fit(X, Y)
    print(f"Training took {time.perf_counter() - t:.0f} s, {net.n_iter_} epochs, "
          f"validation R^2 {net.best_validation_score_:.3f}")
    CACHE.mkdir(exist_ok=True)
    joblib.dump(net, path)
    return net


def learned_knn(net, noisy, a, search):
    Z = net.predict(patch_features(noisy, a.in_patch)).astype(np.float32)
    return search(Z, a.k)


def run_one(net, clean, noisy, a, lams):
    H, W = clean.shape
    n = H * W
    F = patch_features(noisy, a.patch)

    (D, J), t_window = timed(window_candidates, F, H, W, a.window)
    part = np.argpartition(D, a.k, axis=1)[:, :a.k]
    iw, dw = np.take_along_axis(J, part, 1), np.take_along_axis(D, part, 1)
    knn_keep = np.zeros_like(D, dtype=bool)
    np.put_along_axis(knn_keep, part, True, axis=1)
    h = float(np.median(dw))

    (ig, dg), t_global = timed(global_knn, F, a.k)
    (ip, _), t_pca = timed(pca_knn, F, a.k, a.dim)
    (il, _), t_learn = timed(learned_knn, net, noisy, a, global_knn)
    (ila, _), t_learn_nnd = timed(learned_knn, net, noisy, a, nndescent_knn)
    (io, _), t_oracle = timed(global_knn, patch_features(clean, a.patch), a.k)

    graphs = {
        "window kNN": (from_table(D, J, knn_keep, n, h), t_window, recall(ig, iw)),
        "global kNN exact": (from_knn(ig, dg, n, h), t_global, 1.0),
        f"global kNN PCA-{a.dim}": (from_knn(ip, noisy_dist(F, ip), n, h), t_pca, recall(ig, ip)),
        f"learned-{a.dim} exact search": (from_knn(il, noisy_dist(F, il), n, h), t_learn, recall(ig, il)),
        f"learned-{a.dim} NN-Descent": (from_knn(ila, noisy_dist(F, ila), n, h), t_learn_nnd, recall(ig, ila)),
        "oracle: clean patches": (from_knn(io, noisy_dist(F, io), n, h), t_oracle, recall(ig, io)),
    }
    out = []
    for name, (Wg, t_build, rec) in graphs.items():
        p, lam, _ = best_denoise(Wg, clean, noisy, lams, a.iters)
        out.append(dict(graph=name, build_s=t_build, recall=rec, lam=lam, psnr=p,
                        nerr=neighbour_error(Wg, clean)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=IMAGES)
    ap.add_argument("--noises", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--patch", type=int, default=5)
    ap.add_argument("--in_patch", type=int, default=11)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    lams = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]

    overlap = set(a.images) & set(TRAIN_IMAGES)
    if overlap:
        raise SystemExit(f"Test images overlap with training images: {overlap}")

    net = train_model(a)
    nndescent_knn(np.random.default_rng(0).random((500, a.dim), dtype=np.float32), a.k)

    rows = []
    for name in a.images:
        clean = load_image(name, a.size)
        for noise in a.noises:
            for seed in range(a.seeds):
                rng = np.random.default_rng(seed)
                noisy = (clean + noise * rng.standard_normal(clean.shape)).astype(np.float32)
                results = run_one(net, clean, noisy, a, lams)
                base = next(r["psnr"] for r in results if r["graph"] == "window kNN")
                for r in results:
                    rows.append(dict(image=name, noise=noise, seed=seed, graph=r["graph"],
                                     build_s=round(r["build_s"], 3),
                                     recall_vs_exact=round(r["recall"], 3),
                                     neighbour_error=round(r["nerr"], 4),
                                     best_lambda=r["lam"], psnr_db=round(r["psnr"], 2),
                                     vs_window_knn_db=round(r["psnr"] - base, 2)))
                learned = next(r["psnr"] for r in results if r["graph"].startswith("learned"))
                print(f"{name:<10} noise {noise}  seed {seed}  window kNN {base:.2f} dB  "
                      f"learned {learned:.2f} dB  ({learned - base:+.2f})")

    out_csv = PROJECT / "patch_graph_learned.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    graphs = list(dict.fromkeys(r["graph"] for r in rows))
    for noise in a.noises:
        sub = [r for r in rows if r["noise"] == noise]
        print(f"\nSummary, noise {noise}: mean over {len(a.images)} images x {a.seeds} seeds "
              f"(test images never seen in training)")
        print(f"  {'graph':<26} {'build':>7} {'recall':>7} {'nb.error':>8} {'PSNR':>7} {'vs window kNN':>14}")
        for g in graphs:
            gr = [r for r in sub if r["graph"] == g]
            print(f"  {g:<26} {np.mean([r['build_s'] for r in gr]):6.2f}s "
                  f"{np.mean([r['recall_vs_exact'] for r in gr]):7.3f} "
                  f"{np.mean([r['neighbour_error'] for r in gr]):8.4f} "
                  f"{np.mean([r['psnr_db'] for r in gr]):6.2f} "
                  f"{np.mean([r['vs_window_knn_db'] for r in gr]):+13.2f}")
        print(f"\n  PSNR per image (mean over seeds), noise {noise}")
        print(f"  {'graph':<26} " + " ".join(f"{im[:9]:>9}" for im in a.images))
        for g in graphs:
            vals = [np.mean([r["psnr_db"] for r in sub if r["graph"] == g and r["image"] == im])
                    for im in a.images]
            print(f"  {g:<26} " + " ".join(f"{v:9.2f}" for v in vals))
    print(f"\nSaved {out_csv.name}")


if __name__ == "__main__":
    main()