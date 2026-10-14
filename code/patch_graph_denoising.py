import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

from patch_graph_timing import load_digits, patch_features, global_knn, recall

PROJECT = Path(__file__).resolve().parent.parent


def make_pair(size, image, noise, rng, fake):
    if image == "camera":
        from skimage import data
        from skimage.transform import resize
        clean = resize(data.camera() / 255.0, (size, size), anti_aliasing=True)
    else:
        side = int(np.ceil(size / 28))
        d = load_digits(side * side, rng, fake)
        mosaic = d.reshape(side, side, 28, 28).transpose(0, 2, 1, 3).reshape(28 * side, 28 * side)
        clean = mosaic[:size, :size]
    noisy = clean + noise * rng.standard_normal(clean.shape)
    return clean.astype(np.float32), noisy.astype(np.float32)


def window_candidates(F, H, W, s):
    G = F.reshape(H, W, -1)
    offsets = [(dy, dx) for dy in range(-s, s + 1) for dx in range(-s, s + 1)
               if (dy, dx) != (0, 0)]
    n = H * W
    D = np.full((n, len(offsets)), np.inf, dtype=np.float32)
    J = np.zeros((n, len(offsets)), dtype=np.int64)
    base = np.arange(n)
    for o, (dy, dx) in enumerate(offsets):
        y0, y1 = max(0, -dy), min(H, H - dy)
        x0, x1 = max(0, -dx), min(W, W - dx)
        diff = G[y0:y1, x0:x1] - G[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        dist = np.full((H, W), np.inf, dtype=np.float32)
        dist[y0:y1, x0:x1] = np.sqrt((diff ** 2).sum(-1))
        D[:, o] = dist.ravel()
        J[:, o] = base + dy * W + dx
    return D, J


def nndescent_knn(F, k):
    from pynndescent import NNDescent
    idx, d = NNDescent(F, n_neighbors=k + 1, random_state=0, n_jobs=-1).neighbor_graph
    n = len(F)
    keep = idx != np.arange(n)[:, None]
    keep[keep.all(axis=1), -1] = False
    return idx[keep].reshape(n, k), d[keep].reshape(n, k)


def to_graph(rows, cols, dist, n, h):
    D = csr_matrix((dist, (rows, cols)), shape=(n, n))
    D = D.maximum(D.T)
    D.data = np.exp(-(D.data ** 2) / h ** 2)
    return D


def from_table(D, J, keep, n, h):
    rows = np.repeat(np.arange(n), D.shape[1]).reshape(D.shape)[keep]
    return to_graph(rows, J[keep], D[keep], n, h)


def from_knn(idx, d, n, h):
    k = idx.shape[1]
    return to_graph(np.repeat(np.arange(n), k), idx.ravel(), d.ravel(), n, h)


def diffuse(W, f0, lam, iters):
    gamma = 2.0 * W
    denom = lam + np.asarray(gamma.sum(axis=1)).ravel()
    f = f0.copy()
    for _ in range(iters):
        f = (lam * f0 + gamma @ f) / denom
    return f


def psnr(clean, x):
    return 10 * np.log10(1.0 / np.mean((clean - x) ** 2))


def best_denoise(W, clean, noisy, lams, iters):
    best = None
    for lam in lams:
        out = diffuse(W, noisy.ravel(), lam, iters).reshape(clean.shape)
        p = psnr(clean, out)
        if best is None or p > best[0]:
            best = (p, lam, out)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[256])
    ap.add_argument("--image", choices=["mnist", "camera"], default="mnist")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--patch", type=int, default=5)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--eps_q", type=float, default=0.9)
    ap.add_argument("--no_nndescent", action="store_true")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    lams = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    rng = np.random.default_rng(0)

    if not a.no_nndescent:
        nndescent_knn(rng.random((500, a.patch ** 2), dtype=np.float32), a.k)

    rows, first = [], None
    for size in a.sizes:
        clean, noisy = make_pair(size, a.image, a.noise, rng, a.fake)
        H, W = clean.shape
        n = H * W
        F = patch_features(noisy, a.patch)

        t = time.perf_counter()
        D, J = window_candidates(F, H, W, a.window)
        t_window = time.perf_counter() - t

        part = np.argpartition(D, a.k, axis=1)[:, :a.k]
        iw, dw = np.take_along_axis(J, part, 1), np.take_along_axis(D, part, 1)
        knn_keep = np.zeros_like(D, dtype=bool)
        np.put_along_axis(knn_keep, part, True, axis=1)
        h = float(np.median(dw))
        eps = float(np.quantile(dw.max(axis=1), a.eps_q))

        t = time.perf_counter()
        ig, dg = global_knn(F, a.k)
        t_global = time.perf_counter() - t

        graphs = {
            "window all (11x11)": (from_table(D, J, np.isfinite(D), n, h), t_window, None),
            "window kNN": (from_table(D, J, knn_keep, n, h), t_window, recall(ig, iw)),
            "window eps-ball": (from_table(D, J, D <= eps, n, h), t_window, None),
            "global kNN exact": (from_knn(ig, dg, n, h), t_global, 1.0),
        }
        if not a.no_nndescent:
            t = time.perf_counter()
            ia, da = nndescent_knn(F, a.k)
            t_nnd = time.perf_counter() - t
            graphs["global kNN NN-Descent"] = (from_knn(ia, da, n, h), t_nnd, recall(ig, ia))

        noisy_psnr = psnr(clean, noisy)
        images = {"clean": clean, f"noisy ({noisy_psnr:.1f} dB)": noisy}
        print(f"\n{H}x{W} {a.image}  noisy PSNR {noisy_psnr:.2f} dB  h {h:.3f}  eps {eps:.3f}")
        print(f"  {'graph':<22} {'build':>8} {'degree':>7} {'min':>4} {'max':>5} "
              f"{'isolated':>8} {'recall':>7} {'lambda':>6} {'PSNR':>7}")
        for name, (Wg, t_build, rec) in graphs.items():
            p, lam, out = best_denoise(Wg, clean, noisy, lams, a.iters)
            deg = np.diff(Wg.indptr)
            row = dict(image=f"{H}x{W} {a.image}", graph=name, build_s=round(t_build, 3),
                       edges=int(Wg.nnz), mean_degree=round(float(deg.mean()), 1),
                       min_degree=int(deg.min()), max_degree=int(deg.max()),
                       isolated=round(float((deg == 0).mean()), 3),
                       recall_vs_exact=None if rec is None else round(rec, 3),
                       best_lambda=lam, psnr_db=round(p, 2),
                       gain_db=round(p - noisy_psnr, 2))
            rows.append(row)
            images[f"{name} ({p:.1f} dB)"] = out
            rec_txt = "-" if rec is None else f"{rec:.3f}"
            print(f"  {name:<22} {t_build:7.2f}s {row['mean_degree']:7.1f} {row['min_degree']:>4} "
                  f"{row['max_degree']:>5} {row['isolated']:>8.3f} {rec_txt:>7} {lam:>6} {p:6.2f}dB")
        if first is None:
            first = images

    out_csv = PROJECT / "patch_graph_denoising.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, len(first), figsize=(2.6 * len(first), 3))
    for ax, (title, img) in zip(axes, first.items()):
        ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    (PROJECT / "figures").mkdir(exist_ok=True)
    fig.savefig(PROJECT / "figures" / "patch_graph_denoising.png", dpi=200)
    print(f"\nSaved {out_csv.name} and figures/patch_graph_denoising.png")


if __name__ == "__main__":
    main()