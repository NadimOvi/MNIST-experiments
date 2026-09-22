"""
Graph regularization baseline on MNIST.

Each MNIST image is one graph vertex. The graph is complete, with Gaussian
g2 weights. Diffusion follows Elmoataz, Lezoray & Bougleux (IEEE TIP 2008),
Equation (27), for p=2.

Examples:
    python baseline_mnist.py --n 1000
    python baseline_mnist.py --n 2000 --lam 0.01 --iters 10
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


SEED = 0


def load_mnist(n, seed=SEED):
    """Fetch MNIST and return n random images as vectors with values in [0, 1]."""
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
    )

    X = X.astype(np.float64) / 255.0
    y = y.astype(int)

    if n > len(X):
        raise ValueError(f"Requested {n} images, but MNIST contains only {len(X)}.")

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=n, replace=False)
    return X[indices], y[indices]


def build_graph(X, k=20):
    """Build a symmetric sparse k-nearest-neighbour graph with Gaussian weights."""
    start = time.perf_counter()

    distances, indices = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
    ).fit(X).kneighbors(X)

    n = len(X)
    rows, cols, values = [], [], []

    for i in range(n):
        for distance, j in zip(distances[i], indices[i]):
            if i != j:  # exclude the image itself
                rows.append(i)
                cols.append(j)
                values.append(distance)

    D = csr_matrix((values, (rows, cols)), shape=(n, n))
    D = D.maximum(D.T)  # make the graph symmetric

    sigma = np.median(D.data)
    W = D.copy()
    W.data = np.exp(-(W.data ** 2) / (sigma ** 2))

    return W, sigma, time.perf_counter() - start


def diffuse(W, f0, lam=0.01, iters=10):
    """
    Apply Equation (27) for p=2.

    In the paper's definition, gamma = 2 * W for p=2.
    """
    gamma = 2.0 * W
    weighted_degree = gamma.sum(axis=1, keepdims=True)
    denominator = lam + weighted_degree

    f = f0.copy()

    for step in range(iters):
        f_new = (lam * f0 + gamma @ f) / denominator

        relative_change = (
            np.linalg.norm(f_new - f) / (np.linalg.norm(f) + 1e-12)
        )
        f = f_new

        if step == 0 or step == iters - 1:
            print(
                f"iteration {step + 1}: "
                f"relative change = {relative_change:.6f}"
            )

    total_change = (
        np.linalg.norm(f - f0) / (np.linalg.norm(f0) + 1e-12)
    )
    print(f"total change after {iters} iterations = {total_change:.6f}")

    return f


def cluster_accuracy(y_true, y_pred, k=10):
    """Calculate clustering accuracy after best matching clusters to digits."""
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(k))
    row_indices, col_indices = linear_sum_assignment(-matrix)
    return matrix[row_indices, col_indices].sum() / len(y_true)


def kmeans_score(X, y, runs=10, k=10):
    """Return mean and standard deviation of k-means clustering accuracy."""
    scores = []

    for run in range(runs):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=run,
        ).fit(X)

        scores.append(cluster_accuracy(y, model.labels_, k))

    return float(np.mean(scores)), float(np.std(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="number of images (graph vertices)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.01,
        help="fidelity parameter",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="number of diffusion iterations",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="number of k-means runs",
    )
    args = parser.parse_args()

    if args.lam < 0:
        parser.error("--lam must be non-negative")
    if args.iters < 1:
        parser.error("--iters must be at least 1")
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    X, y = load_mnist(args.n)
    print(f"n = {args.n}, image dimension = {X.shape[1]}")

    W, sigma, build_time = build_graph(X)
    weights = W[np.triu_indices_from(W, k=1)]

    print(f"graph built in {build_time:.3f} seconds")
    print(f"sigma = {sigma:.3f}")
    print(f"W memory = {W.nbytes / 1e6:.1f} MB")
    print(
        "weight percentiles (10%, 50%, 90%) = "
        f"{np.percentile(weights, [10, 50, 90])}"
    )
    print(f"average weighted degree = {W.sum(axis=1).mean():.3f}")

    start_time = time.perf_counter()
    X_regularized = diffuse(W, X, lam=args.lam, iters=args.iters)
    diffusion_time = time.perf_counter() - start_time
    print(f"diffusion done in {diffusion_time:.3f} seconds")

    raw_mean, raw_std = kmeans_score(X, y, runs=args.runs)
    regularized_mean, regularized_std = kmeans_score(
        X_regularized,
        y,
        runs=args.runs,
    )

    print(
        f"\nk-means accuracy, raw: "
        f"{raw_mean:.4f} +/- {raw_std:.4f}"
    )
    print(
        f"k-means accuracy, regularized: "
        f"{regularized_mean:.4f} +/- {regularized_std:.4f}"
    )

    results_file = Path("results.csv")
    write_header = not results_file.exists()

    with results_file.open("a", encoding="utf-8") as file:
        if write_header:
            file.write(
                "n,lambda,iters,graph_seconds,diffusion_seconds,"
                "raw_accuracy,regularized_accuracy\n"
            )

        file.write(
            f"{args.n},{args.lam},{args.iters},"
            f"{build_time:.4f},{diffusion_time:.4f},"
            f"{raw_mean:.4f},{regularized_mean:.4f}\n"
        )


if __name__ == "__main__":
    main()