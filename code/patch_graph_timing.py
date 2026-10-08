import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / "cache"



def load_digits(n, rng, fake=False):
    if fake:

        x = np.linspace(-1, 1, 28)
        yy, xx = np.meshgrid(x, x, indexing="ij")
        c = rng.uniform(-0.5, 0.5, (n, 2, 1, 1))
        r = rng.uniform(0.2, 0.5, (n, 1, 1))
        return (((yy - c[:, 0]) ** 2 + (xx - c[:, 1]) ** 2) < r ** 2).astype(np.float32)
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "mnist.npy"
    if not f.exists():
        from sklearn.datasets import fetch_openml
        X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        np.save(f, (X / 255.0).astype(np.float32))
    X = np.load(f)
    return X[rng.choice(len(X), n, replace=False)].reshape(-1, 28, 28)


def make_image(side, noise, rng, fake):
    d = load_digits(side * side, rng, fake)
    img = d.reshape(side, side, 28, 28).transpose(0, 2, 1, 3).reshape(28 * side, 28 * side)
    noisy = (img + noise * rng.standard_normal(img.shape)).astype(np.float32)
    return noisy, img > 0.5


def patch_features(img, p=5):
    r = p // 2
    padded = np.pad(img, r, mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(padded, (p, p))
    return np.ascontiguousarray(win.reshape(img.size, p * p))



def global_knn(F, k):
    d, idx = NearestNeighbors(n_neighbors=k + 1, algorithm="brute",
                              n_jobs=-1).fit(F).kneighbors(F)
    n = len(F)
    keep = idx != np.arange(n)[:, None]
    keep[keep.all(axis=1), -1] = False
    return idx[keep].reshape(n, k), d[keep].reshape(n, k)


def window_knn(F, H, W, k, s):
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
    part = np.argpartition(D, k, axis=1)[:, :k]
    return np.take_along_axis(J, part, 1), np.take_along_axis(D, part, 1)



def weighted_graph(idx, d):
    n, k = idx.shape
    D = csr_matrix((d.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    D = D.maximum(D.T)
    sigma = np.median(D.data)
    D.data = np.exp(-(D.data ** 2) / sigma ** 2)
    return D


def recall(idx_true, idx_test):
    return float((idx_test[:, :, None] == idx_true[:, None, :]).any(-1).mean())


def timed(fn, *args):
    t = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sides", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--patch", type=int, default=5)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    rows = []
    for side in a.sides:
        img, digit = make_image(side, a.noise, rng, a.fake)
        fg = digit.ravel()
        H, W = img.shape
        F = patch_features(img, a.patch)
        N = len(F)

        (ig, dg), t_global = timed(global_knn, F, a.k)
        (iw, dw), t_window = timed(window_knn, F, H, W, a.k, a.window)
        Wg, t_weights = timed(weighted_graph, ig, dg)


        y, x = np.divmod(np.arange(N), W)
        ny, nx = np.divmod(ig, W)
        cheb = np.maximum(abs(ny - y[:, None]), abs(nx - x[:, None]))

        row = dict(
            image=f"{H}x{W}", N=N,
            global_s=round(t_global, 3), window_s=round(t_window, 3),
            weights_s=round(t_weights, 3),
            dense_W_MB=round(N * N * 4 / 1e6, 1),
            sparse_W_MB=round((Wg.data.nbytes + Wg.indices.nbytes + Wg.indptr.nbytes) / 1e6, 2),
            window_recall=round(recall(ig, iw), 3),
            outside_window=round(float((cheb > a.window).mean()), 3),
            median_px_dist=int(np.median(cheb)),
            dist_ratio=round(float(dw.mean() / dg.mean()), 3),
            digit_pixels=round(float(fg.mean()), 3),
            outside_digit=round(float((cheb[fg] > a.window).mean()), 3),
            ratio_digit=round(float(dw[fg].mean() / dg[fg].mean()), 3),
        )
        rows.append(row)
        print(row)


    out_csv = PROJECT / "patch_graph_timing.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


    N = np.array([r["N"] for r in rows], float)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in [("global_s", "global search"), ("window_s", "window search"),
                       ("weights_s", "weight computation")]:
        t = np.array([r[key] for r in rows], float)
        slope = np.polyfit(np.log(N), np.log(np.maximum(t, 1e-4)), 1)[0] if len(N) > 1 else np.nan
        ax.loglog(N, t, "o-", label=f"{label} (slope {slope:.2f})")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("number of pixels N")
    ax.set_ylabel("time (s)")
    ax.set_title("Cost of building the patch graph")
    ax.legend()
    fig.tight_layout()
    (PROJECT / "figures").mkdir(exist_ok=True)
    fig.savefig(PROJECT / "figures" / "patch_graph_timing.png", dpi=200)

    print("\nSummary")
    print(f"{'image':>9} {'N':>7} {'global':>8} {'window':>8} {'weights':>8} "
          f"{'dense W':>9} {'outside':>8} {'ratio':>6} {'out.digit':>9} {'r.digit':>7}")
    for r in rows:
        print(f"{r['image']:>9} {r['N']:>7} {r['global_s']:>7}s {r['window_s']:>7}s "
              f"{r['weights_s']:>7}s {r['dense_W_MB']:>7}MB {r['outside_window']:>8} "
              f"{r['dist_ratio']:>6} {r['outside_digit']:>9} {r['ratio_digit']:>7}")
    print(f"\nSaved {out_csv.name} and figures/patch_graph_timing.png")


if __name__ == "__main__":
    main()