import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pynndescent import NNDescent
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors


SEED = 2


def load_mnist(n, seed=SEED):
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
    )

    X = X.astype(np.float32) / 255.0
    y = y.astype(int)

    if n > len(X):
        raise ValueError(
            f"Requested {n} images, but MNIST contains only {len(X)}."
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=n, replace=False)
    return X[indices], y[indices]


def clean_neighbors(indices, distances, k):
    result_indices = []
    result_distances = []

    for i in range(len(indices)):
        row_indices = []
        row_distances = []

        for j, distance in zip(indices[i], distances[i]):
            if j != i:
                row_indices.append(int(j))
                row_distances.append(float(distance))

            if len(row_indices) == k:
                break

        result_indices.append(row_indices)
        result_distances.append(row_distances)

    return result_indices, result_distances


def build_weighted_graph(neighbor_indices, neighbor_distances, n):
    rows = []
    cols = []
    values = []

    for i, (indices, distances) in enumerate(
        zip(neighbor_indices, neighbor_distances)
    ):
        rows.extend([i] * len(indices))
        cols.extend(indices)
        values.extend(distances)

    D = csr_matrix((values, (rows, cols)), shape=(n, n))
    D = D.maximum(D.T)

    if D.nnz == 0:
        raise ValueError("The graph has no edges.")

    sigma = float(np.median(D.data))

    if sigma <= 0:
        raise ValueError("The median edge distance must be positive.")

    W = D.copy()
    W.data = np.exp(-(W.data ** 2) / (sigma ** 2))

    return W, sigma


def exact_graph(X, k):
    start = time.perf_counter()

    distances, indices = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1,
    ).fit(X).kneighbors(X)

    neighbor_indices, neighbor_distances = clean_neighbors(
        indices,
        distances,
        k,
    )
    W, sigma = build_weighted_graph(
        neighbor_indices,
        neighbor_distances,
        len(X),
    )

    return W, sigma, neighbor_indices, time.perf_counter() - start


def approximate_graph(X, k):
    start = time.perf_counter()

    index = NNDescent(
        X,
        n_neighbors=k + 1,
        metric="euclidean",
        random_state=SEED,
        n_jobs=-1,
    )

    indices, distances = index.neighbor_graph
    neighbor_indices, neighbor_distances = clean_neighbors(
        indices,
        distances,
        k,
    )
    W, sigma = build_weighted_graph(
        neighbor_indices,
        neighbor_distances,
        len(X),
    )

    return W, sigma, neighbor_indices, time.perf_counter() - start


def graph_memory_mb(W):
    total_bytes = W.data.nbytes + W.indices.nbytes + W.indptr.nbytes
    return total_bytes / 1e6


def recall_at_k(exact_indices, approximate_indices, k):
    recalls = []

    for exact, approximate in zip(exact_indices, approximate_indices):
        exact_set = set(exact)
        approximate_set = set(approximate)
        recalls.append(len(exact_set & approximate_set) / k)

    return float(np.mean(recalls))


def diffuse(W, f0, lam=5.0, iters=10):
    gamma = 2.0 * W
    weighted_degree = np.asarray(gamma.sum(axis=1)).reshape(-1, 1)
    denominator = lam + weighted_degree
    f = f0.copy()

    for _ in range(iters):
        f = (lam * f0 + gamma @ f) / denominator

    return f


def cluster_accuracy(y_true, y_pred, k=10):
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(k))
    rows, cols = linear_sum_assignment(-matrix)
    return matrix[rows, cols].sum() / len(y_true)


def kmeans_score(X, y, runs=10, k=10):
    scores = []

    for run in range(runs):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=run,
        ).fit(X)

        scores.append(cluster_accuracy(y, model.labels_, k))

    return float(np.mean(scores)), float(np.std(scores))


def save_results_and_figures(
    n,
    k,
    exact_time,
    cold_time,
    warm_time,
    recall,
    exact_W,
    approximate_W,
    raw_accuracy=None,
    exact_accuracy=None,
    approximate_accuracy=None,
):
    project_dir = Path(__file__).resolve().parent.parent
    results_path = project_dir / "graph_comparison.csv"
    figures_dir = project_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "n",
        "k",
        "exact_seconds",
        "approx_cold_seconds",
        "approx_warm_seconds",
        "recall_at_k",
        "exact_memory_mb",
        "approx_memory_mb",
        "raw_accuracy",
        "exact_accuracy",
        "approx_accuracy",
    ]

    row = {
        "n": n,
        "k": k,
        "exact_seconds": exact_time,
        "approx_cold_seconds": cold_time,
        "approx_warm_seconds": warm_time,
        "recall_at_k": recall,
        "exact_memory_mb": graph_memory_mb(exact_W),
        "approx_memory_mb": graph_memory_mb(approximate_W),
        "raw_accuracy": raw_accuracy if raw_accuracy is not None else "",
        "exact_accuracy": exact_accuracy if exact_accuracy is not None else "",
        "approx_accuracy": (
            approximate_accuracy if approximate_accuracy is not None else ""
        ),
    }

    write_header = not results_path.exists() or results_path.stat().st_size == 0

    with results_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    with results_path.open("r", newline="", encoding="utf-8") as file:
        rows = [
            item for item in csv.DictReader(file)
            if int(item["k"]) == k
        ]

    sizes = sorted({int(item["n"]) for item in rows})

    def median_for(size, column):
        values = [
            float(item[column])
            for item in rows
            if int(item["n"]) == size and item[column] != ""
        ]
        return float(np.median(values)) if values else np.nan

    plt.figure()
    plt.plot(
        sizes,
        [median_for(size, "exact_seconds") for size in sizes],
        marker="o",
        label="Exact",
    )
    plt.plot(
        sizes,
        [median_for(size, "approx_cold_seconds") for size in sizes],
        marker="o",
        label="Approximate cold",
    )
    plt.plot(
        sizes,
        [median_for(size, "approx_warm_seconds") for size in sizes],
        marker="o",
        label="Approximate warm",
    )
    plt.xlabel("Number of images")
    plt.ylabel("Build time (seconds)")
    plt.title(f"Graph construction time, k={k}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "graph_build_time.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(
        sizes,
        [median_for(size, "exact_memory_mb") for size in sizes],
        marker="o",
        label="Exact",
    )
    plt.plot(
        sizes,
        [median_for(size, "approx_memory_mb") for size in sizes],
        marker="o",
        label="Approximate",
    )
    plt.xlabel("Number of images")
    plt.ylabel("Graph memory (MB)")
    plt.title(f"Graph memory, k={k}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "graph_memory.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(
        sizes,
        [median_for(size, "recall_at_k") for size in sizes],
        marker="o",
    )
    plt.xlabel("Number of images")
    plt.ylabel(f"Neighbour recall@{k}")
    plt.title(f"Approximate graph recall, k={k}")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(figures_dir / "graph_recall.png", dpi=200)
    plt.close()

    quality_rows = [
        item for item in rows
        if item["raw_accuracy"] != ""
        and item["exact_accuracy"] != ""
        and item["approx_accuracy"] != ""
    ]

    if quality_rows:
        quality_sizes = sorted({int(item["n"]) for item in quality_rows})

        def median_quality(size, column):
            values = [
                float(item[column])
                for item in quality_rows
                if int(item["n"]) == size
            ]
            return float(np.median(values))

        plt.figure()
        plt.plot(
            quality_sizes,
            [median_quality(size, "raw_accuracy") for size in quality_sizes],
            marker="o",
            label="Raw data",
        )
        plt.plot(
            quality_sizes,
            [median_quality(size, "exact_accuracy") for size in quality_sizes],
            marker="o",
            label="Exact graph",
        )
        plt.plot(
            quality_sizes,
            [
                median_quality(size, "approx_accuracy")
                for size in quality_sizes
            ],
            marker="o",
            label="Approximate graph",
        )
        plt.xlabel("Number of images")
        plt.ylabel("Clustering accuracy")
        plt.title(f"Clustering results, k={k}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "clustering_accuracy.png", dpi=200)
        plt.close()

    print(f"\nSaved results to: {results_path}")
    print(f"Saved figures to: {figures_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--lam", type=float, default=5.0)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--graph-only", action="store_true")
    args = parser.parse_args()

    if args.n < 2:
        parser.error("--n must be at least 2")
    if args.k < 1 or args.k >= args.n:
        parser.error("--k must be at least 1 and smaller than --n")
    if args.lam < 0:
        parser.error("--lam must be non-negative")
    if args.iters < 1:
        parser.error("--iters must be at least 1")
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    X, y = load_mnist(args.n)
    print(f"n = {args.n}, dimension = {X.shape[1]}")

    exact_W, exact_sigma, exact_indices, exact_time = exact_graph(X, args.k)
    print("\nExact k-NN")
    print(f"build time = {exact_time:.3f} seconds")
    print(f"sigma = {exact_sigma:.3f}")
    print(f"memory = {graph_memory_mb(exact_W):.2f} MB")
    print(f"edges = {exact_W.nnz}")

    approximate_W, approximate_sigma, approximate_indices, cold_time = (
        approximate_graph(X, args.k)
    )
    print("\nApproximate k-NN cold start")
    print(f"build time = {cold_time:.3f} seconds")

    approximate_W, approximate_sigma, approximate_indices, warm_time = (
        approximate_graph(X, args.k)
    )
    print("\nApproximate k-NN warmed up")
    print(f"build time = {warm_time:.3f} seconds")
    print(f"sigma = {approximate_sigma:.3f}")
    print(f"memory = {graph_memory_mb(approximate_W):.2f} MB")
    print(f"edges = {approximate_W.nnz}")

    recall = recall_at_k(exact_indices, approximate_indices, args.k)
    print(f"\napproximate neighbour recall@{args.k} = {recall:.4f}")

    if args.graph_only:
        save_results_and_figures(
            args.n,
            args.k,
            exact_time,
            cold_time,
            warm_time,
            recall,
            exact_W,
            approximate_W,
        )
        return

    start = time.perf_counter()
    X_exact = diffuse(exact_W, X, lam=args.lam, iters=args.iters)
    exact_diffusion_time = time.perf_counter() - start

    start = time.perf_counter()
    X_approximate = diffuse(
        approximate_W,
        X,
        lam=args.lam,
        iters=args.iters,
    )
    approximate_diffusion_time = time.perf_counter() - start

    print(f"\nExact graph diffusion time = {exact_diffusion_time:.3f} seconds")
    print(
        "Approximate graph diffusion time = "
        f"{approximate_diffusion_time:.3f} seconds"
    )

    raw_mean, raw_std = kmeans_score(X, y, runs=args.runs)
    exact_mean, exact_std = kmeans_score(X_exact, y, runs=args.runs)
    approximate_mean, approximate_std = kmeans_score(
        X_approximate,
        y,
        runs=args.runs,
    )

    print(f"\nRaw k-means: {raw_mean:.4f} +/- {raw_std:.4f}")
    print(f"Exact graph: {exact_mean:.4f} +/- {exact_std:.4f}")
    print(
        f"Approximate graph: "
        f"{approximate_mean:.4f} +/- {approximate_std:.4f}"
    )

    save_results_and_figures(
        args.n,
        args.k,
        exact_time,
        cold_time,
        warm_time,
        recall,
        exact_W,
        approximate_W,
        raw_mean,
        exact_mean,
        approximate_mean,
    )


if __name__ == "__main__":
    main()