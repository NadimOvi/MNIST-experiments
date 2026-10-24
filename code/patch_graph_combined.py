import argparse
import csv
import time
from pathlib import Path

import numpy as np

from patch_graph_timing import patch_features, global_knn
from patch_graph_denoising import window_candidates, nndescent_knn, from_knn, best_denoise
from patch_graph_benchmark import load_image, noisy_dist, neighbour_error, IMAGES
from patch_graph_learned import train_model, TRAIN_IMAGES

PROJECT = Path(__file__).resolve().parent.parent


def sorted_window(D, J, k):
    part = np.argpartition(D, k, axis=1)[:, :k]
    d = np.take_along_axis(D, part, 1)
    order = np.argsort(d, axis=1)
    return np.take_along_axis(np.take_along_axis(J, part, 1), order, 1)


def knn_graph(F, idx, n, h):
    return from_knn(idx, noisy_dist(F, idx), n, h)


def run_one(net, clean, noisy, a, lams):
    H, W = clean.shape
    n = H * W
    F = patch_features(noisy, a.patch)

    t = time.perf_counter()
    D, J = window_candidates(F, H, W, a.window)
    w20 = sorted_window(D, J, 2 * a.k)
    t_window = time.perf_counter() - t
    h = float(np.median(noisy_dist(F, w20[:, :a.k])))

    t = time.perf_counter()
    Z = net.predict(patch_features(noisy, a.in_patch)).astype(np.float32)
    l20, _ = nndescent_knn(Z, 2 * a.k)
    t_learned = time.perf_counter() - t

    t = time.perf_counter()
    DZ, JZ = window_candidates(Z, H, W, a.window)
    wl = sorted_window(DZ, JZ, a.k)
    t_learned_window = time.perf_counter() - t

    io, _ = global_knn(patch_features(clean, a.patch), a.k)

    k, half = a.k, a.k // 2
    window10, window20 = knn_graph(F, w20[:, :k], n, h), knn_graph(F, w20, n, h)
    learned10, learned20 = knn_graph(F, l20[:, :k], n, h), knn_graph(F, l20, n, h)
    graphs = {
        f"window kNN ({k})": (window10, t_window),
        f"window kNN ({2 * k})": (window20, t_window),
        f"learned global ({k})": (learned10, t_learned),
        f"learned global ({2 * k})": (learned20, t_learned),
        f"combined {half}+{half}": (knn_graph(F, w20[:, :half], n, h).maximum(
            knn_graph(F, l20[:, :half], n, h)), t_window + t_learned),
        f"combined {k}+{k}": (window10.maximum(learned10), t_window + t_learned),
        f"window kNN, learned codes ({k})": (knn_graph(F, wl, n, h), t_learned_window),
        "oracle: clean patches": (knn_graph(F, io, n, h), 0.0),
    }
    out = []
    for name, (Wg, t_build) in graphs.items():
        p, lam, _ = best_denoise(Wg, clean, noisy, lams, a.iters)
        deg = np.diff(Wg.indptr)
        out.append(dict(graph=name, build_s=t_build, degree=float(deg.mean()), lam=lam,
                        psnr=p, nerr=neighbour_error(Wg, clean)))
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
                base = results[0]["psnr"]
                for r in results:
                    rows.append(dict(image=name, noise=noise, seed=seed, graph=r["graph"],
                                     build_s=round(r["build_s"], 3),
                                     mean_degree=round(r["degree"], 1),
                                     neighbour_error=round(r["nerr"], 4),
                                     best_lambda=r["lam"], psnr_db=round(r["psnr"], 2),
                                     vs_window_knn_db=round(r["psnr"] - base, 2)))
                best = max((r for r in results if not r["graph"].startswith("oracle")),
                           key=lambda r: r["psnr"])
                print(f"{name:<10} noise {noise}  seed {seed}  best: {best['graph']} "
                      f"({best['psnr']:.2f} dB, {best['psnr'] - base:+.2f} vs window kNN)")

    out_csv = PROJECT / "patch_graph_combined.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    graphs = list(dict.fromkeys(r["graph"] for r in rows))
    for noise in a.noises:
        sub = [r for r in rows if r["noise"] == noise]
        runs = sorted({(r["image"], r["seed"]) for r in sub})
        print(f"\nSummary, noise {noise}: mean over {len(a.images)} images x {a.seeds} seeds")
        print(f"  {'graph':<32} {'build':>7} {'degree':>7} {'nb.error':>8} {'PSNR':>7} "
              f"{'vs window':>10} {'wins':>6}")
        for g in graphs:
            gr = [r for r in sub if r["graph"] == g]
            wins = sum(1 for run in runs if not g.startswith("oracle") and
                       max((r for r in sub if (r["image"], r["seed"]) == run
                            and not r["graph"].startswith("oracle")),
                           key=lambda r: r["psnr_db"])["graph"] == g)
            print(f"  {g:<32} {np.mean([r['build_s'] for r in gr]):6.2f}s "
                  f"{np.mean([r['mean_degree'] for r in gr]):7.1f} "
                  f"{np.mean([r['neighbour_error'] for r in gr]):8.4f} "
                  f"{np.mean([r['psnr_db'] for r in gr]):6.2f} "
                  f"{np.mean([r['vs_window_knn_db'] for r in gr]):+9.2f} "
                  f"{(str(wins) + '/' + str(len(runs))) if not g.startswith('oracle') else '-':>6}")
        print(f"\n  PSNR per image (mean over seeds), noise {noise}")
        print(f"  {'graph':<32} " + " ".join(f"{im[:9]:>9}" for im in a.images))
        for g in graphs:
            vals = [np.mean([r["psnr_db"] for r in sub if r["graph"] == g and r["image"] == im])
                    for im in a.images]
            print(f"  {g:<32} " + " ".join(f"{v:9.2f}" for v in vals))
    print(f"\nSaved {out_csv.name}")


if __name__ == "__main__":
    main()