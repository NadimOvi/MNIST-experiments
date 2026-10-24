import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from patch_graph_timing import patch_features, global_knn, recall
from patch_graph_denoising import (window_candidates, nndescent_knn, from_table,
                                   from_knn, best_denoise, psnr)

PROJECT = Path(__file__).resolve().parent.parent
IMAGES = ["camera", "astronaut", "coffee", "chelsea", "coins", "brick"]


def load_image(name, size):
    from skimage import data
    from skimage.color import rgb2gray
    from skimage.transform import resize
    img = getattr(data, name)()
    if img.ndim == 3:
        img = rgb2gray(img[..., :3])
    img = img.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0
    h, w = img.shape
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return resize(img, (size, size), anti_aliasing=True).astype(np.float32)


def timed(fn, *args):
    t = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t


def pca_knn(F, k, dim):
    Z = PCA(n_components=dim, random_state=0).fit_transform(F).astype(np.float32)
    return global_knn(Z, k)


def noisy_dist(F, idx):
    return np.sqrt(((F[:, None, :] - F[idx]) ** 2).sum(-1)).astype(np.float32)


def neighbour_error(Wg, clean):
    c = clean.ravel()
    coo = Wg.tocoo()
    return float(np.abs(c[coo.row] - c[coo.col]).mean())


def run_one(clean, noisy, a, lams):
    H, W = clean.shape
    n = H * W
    F = patch_features(noisy, a.patch)

    (D, J), t_window = timed(window_candidates, F, H, W, a.window)
    part = np.argpartition(D, a.k, axis=1)[:, :a.k]
    iw, dw = np.take_along_axis(J, part, 1), np.take_along_axis(D, part, 1)
    knn_keep = np.zeros_like(D, dtype=bool)
    np.put_along_axis(knn_keep, part, True, axis=1)
    h = float(np.median(dw))
    eps = float(np.quantile(dw.max(axis=1), a.eps_q))

    (ig, dg), t_global = timed(global_knn, F, a.k)
    (ip, dp), t_pca = timed(pca_knn, F, a.k, a.pca_dim)
    (ia, da), t_nnd = timed(nndescent_knn, F, a.k)
    (io, do), t_oracle = timed(global_knn, patch_features(clean, a.patch), a.k)

    graphs = {
        "window kNN": (from_table(D, J, knn_keep, n, h), t_window, recall(ig, iw)),
        "window eps-ball": (from_table(D, J, D <= eps, n, h), t_window, None),
        "global kNN exact": (from_knn(ig, dg, n, h), t_global, 1.0),
        "global kNN NN-Descent": (from_knn(ia, da, n, h), t_nnd, recall(ig, ia)),
        f"global kNN PCA-{a.pca_dim}": (from_knn(ip, noisy_dist(F, ip), n, h), t_pca, recall(ig, ip)),
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
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--eps_q", type=float, default=0.9)
    ap.add_argument("--pca_dim", type=int, default=8)
    a = ap.parse_args()
    lams = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]

    nndescent_knn(np.random.default_rng(0).random((500, a.patch ** 2), dtype=np.float32), a.k)

    rows = []
    for name in a.images:
        clean = load_image(name, a.size)
        for noise in a.noises:
            for seed in range(a.seeds):
                rng = np.random.default_rng(seed)
                noisy = (clean + noise * rng.standard_normal(clean.shape)).astype(np.float32)
                noisy_psnr = psnr(clean, noisy)
                results = run_one(clean, noisy, a, lams)
                base = next(r["psnr"] for r in results if r["graph"] == "window kNN")
                for r in results:
                    rows.append(dict(image=name, noise=noise, seed=seed, graph=r["graph"],
                                     build_s=round(r["build_s"], 3),
                                     recall_vs_exact="" if r["recall"] is None else round(r["recall"], 3),
                                     neighbour_error=round(r["nerr"], 4),
                                     best_lambda=r["lam"], noisy_psnr=round(noisy_psnr, 2),
                                     psnr_db=round(r["psnr"], 2),
                                     vs_window_knn_db=round(r["psnr"] - base, 2)))
                best = max(results, key=lambda r: r["psnr"] if not r["graph"].startswith("oracle") else -1)
                print(f"{name:<10} noise {noise}  seed {seed}  best real graph: "
                      f"{best['graph']} ({best['psnr']:.2f} dB)")

    out_csv = PROJECT / "patch_graph_benchmark.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    graphs = list(dict.fromkeys(r["graph"] for r in rows))
    for noise in a.noises:
        print(f"\nSummary, noise {noise}: mean over {len(a.images)} images x {a.seeds} seeds")
        print(f"  {'graph':<24} {'build':>7} {'recall':>7} {'nb.error':>8} {'PSNR':>7} {'vs window kNN':>14} {'wins':>6}")
        sub = [r for r in rows if r["noise"] == noise]
        runs = {(r["image"], r["seed"]) for r in sub}
        for g in graphs:
            gr = [r for r in sub if r["graph"] == g]
            rec = [r["recall_vs_exact"] for r in gr if r["recall_vs_exact"] != ""]
            wins = sum(1 for run in runs if all(
                r["psnr_db"] <= next(x["psnr_db"] for x in gr if (x["image"], x["seed"]) == run)
                for r in sub if (r["image"], r["seed"]) == run and not r["graph"].startswith("oracle")))
            print(f"  {g:<24} {np.mean([r['build_s'] for r in gr]):6.2f}s "
                  f"{(f'{np.mean(rec):.3f}' if rec else '-'):>7} "
                  f"{np.mean([r['neighbour_error'] for r in gr]):8.4f} "
                  f"{np.mean([r['psnr_db'] for r in gr]):6.2f} "
                  f"{np.mean([r['vs_window_knn_db'] for r in gr]):+13.2f} "
                  f"{(str(wins) + '/' + str(len(runs))) if not g.startswith('oracle') else '-':>6}")

        print(f"\n  PSNR per image (mean over seeds), noise {noise}")
        print(f"  {'graph':<24} " + " ".join(f"{im[:9]:>9}" for im in a.images))
        for g in graphs:
            vals = [np.mean([r["psnr_db"] for r in sub if r["graph"] == g and r["image"] == im])
                    for im in a.images]
            print(f"  {g:<24} " + " ".join(f"{v:9.2f}" for v in vals))
    print(f"\nSaved {out_csv.name}")


if __name__ == "__main__":
    main()