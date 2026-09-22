"""
Baseline v2: graph regularization on MNIST.
Elmoataz, Lezoray & Bougleux, IEEE TIP 2008, Section V.

v2 adds: k-NN graphs, sigma scaling, relative lambda, and diagnostics.

Usage:
    python baseline_mnist_v2.py --n 1000 --knn 10 --sigma-scale 0.25
    python baseline_mnist_v2.py --n 1000 --sweep
"""

import argparse
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix

SEED = 0


def load_mnist(n, seed=SEED):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float64) / 255.0
    y = y.astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def build_graph(X, knn=None, sigma_scale=1.0):
    """g2 weights, optionally sparsified to a symmetric k-NN graph.

    sigma = sigma_scale * median(nonzero distances).
    Smaller sigma_scale -> more discriminative weights.
    """
    t0 = time.perf_counter()
    D = cdist(X, X, metric="euclidean")
    sigma = sigma_scale * np.median(D[D > 0])
    W = np.exp(-(D ** 2) / (sigma ** 2))
    np.fill_diagonal(W, 0.0)

    if knn is not None:
        n = W.shape[0]
        mask = np.zeros_like(W, dtype=bool)
        # indices of the knn largest weights in each row
        nbrs = np.argpartition(-W, kth=knn, axis=1)[:, :knn]
        mask[np.arange(n)[:, None], nbrs] = True
        W = W * mask
        W = np.maximum(W, W.T)        # symmetrize: union of k-NN

    return W, sigma, time.perf_counter() - t0


def diffuse(W, f0, lam, iters=10):
    """Eq. (27) with p = 2 (gamma reduces to w)."""
    denom = lam + W.sum(axis=1, keepdims=True)
    f = f0.copy()
    for _ in range(iters):
        f = (lam * f0 + W @ f) / denom
    return f


def cluster_accuracy(y_true, y_pred, k=10):
    C = confusion_matrix(y_true, y_pred, labels=np.arange(k))
    row, col = linear_sum_assignment(-C)
    return C[row, col].sum() / len(y_true)


def kmeans_score(X, y, runs=10, k=10):
    scores = [
        cluster_accuracy(y, KMeans(n_clusters=k, n_init=10, random_state=r).fit(X).labels_, k)
        for r in range(runs)
    ]
    return float(np.mean(scores)), float(np.std(scores))


def run_once(X, y, knn, sigma_scale, lam_rel, iters, runs, verbose=True):
    W, sigma, t_build = build_graph(X, knn=knn, sigma_scale=sigma_scale)
    rowsum = W.sum(axis=1)
    lam = lam_rel * rowsum.mean()          # lambda relative to typical row sum

    Xreg = diffuse(W, X, lam=lam, iters=iters)
    change = np.linalg.norm(Xreg - X) / np.linalg.norm(X)

    raw_m, raw_s = kmeans_score(X, y, runs=runs)
    reg_m, reg_s = kmeans_score(Xreg, y, runs=runs)

    if verbose:
        print(f"  sigma={sigma:.3f}  mean rowsum={rowsum.mean():.2f}  "
              f"lam={lam:.3f}  build={t_build:.3f}s")
        print(f"  relative change in data: {change:.4f}")
        print(f"  k-means raw         : {raw_m:.3f} +/- {raw_s:.3f}")
        print(f"  k-means regularized : {reg_m:.3f} +/- {reg_s:.3f}")

    return dict(knn=knn, sigma_scale=sigma_scale, sigma=sigma, lam=lam,
                t_build=t_build, change=change, raw=raw_m, reg=reg_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--knn", type=int, default=None,
                    help="k for k-NN graph; omit for complete graph")
    ap.add_argument("--sigma-scale", type=float, default=1.0)
    ap.add_argument("--lam-rel", type=float, default=0.1,
                    help="lambda as a fraction of the mean row sum")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--sweep", action="store_true",
                    help="grid over knn and sigma_scale")
    args = ap.parse_args()

    X, y = load_mnist(args.n)
    print(f"n = {args.n}, dim = {X.shape[1]}\n")

    if not args.sweep:
        run_once(X, y, args.knn, args.sigma_scale, args.lam_rel,
                 args.iters, args.runs)
        return

    rows = []
    for knn in [5, 10, 20, 50, None]:
        for ss in [0.1, 0.25, 0.5, 1.0]:
            label = f"knn={knn if knn else 'complete':<8} sigma_scale={ss}"
            print(label)
            rows.append(run_once(X, y, knn, ss, args.lam_rel,
                                 args.iters, args.runs))
            print()

    print("\n--- summary (sorted by regularized accuracy) ---")
    for r in sorted(rows, key=lambda d: -d["reg"]):
        k = r["knn"] if r["knn"] else "complete"
        print(f"knn={str(k):<9} sigma_scale={r['sigma_scale']:<5} "
              f"raw={r['raw']:.3f}  reg={r['reg']:.3f}  "
              f"delta={r['reg'] - r['raw']:+.3f}  change={r['change']:.3f}")


if __name__ == "__main__":
    main()