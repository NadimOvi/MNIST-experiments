import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

# (csv file, graph name in that file, label in the figure)
METHODS = [
    ("patch_graph_benchmark.csv", "global kNN exact", "global kNN exact"),
    ("patch_graph_benchmark.csv", "global kNN PCA-8", "global kNN PCA-8"),
    ("patch_graph_learned.csv", "learned-8 NN-Descent", "learned global, NN-Descent"),
    ("patch_graph_combined.csv", "combined 5+5", "combined 5+5"),
    ("patch_graph_combined.csv", "window kNN, learned codes (10)", "window kNN, learned codes"),
]
ORACLE = ("patch_graph_combined.csv", "oracle: clean patches")


def load(name):
    with (PROJECT / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def gains(rows, graph, noise, images):
    """Mean PSNR gain over window kNN per image (mean over seeds)."""
    out = []
    for im in images:
        v = [float(r["vs_window_knn_db"]) for r in rows
             if r["graph"] == graph and r["image"] == im and float(r["noise"]) == noise]
        if not v:
            raise SystemExit(f"no rows for graph '{graph}', image {im}, noise {noise}")
        out.append(np.mean(v))
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noises", type=float, nargs="+", default=[0.1, 0.2])
    a = p.parse_args()

    tables = {f: load(f) for f in {m[0] for m in METHODS} | {ORACLE[0]}}
    combined = tables[ORACLE[0]]
    images = list(dict.fromkeys(r["image"] for r in combined))
    labels = [m[2] for m in METHODS]
    G = {noise: np.array([gains(tables[f], g, noise, images) for f, g, _ in METHODS])
         for noise in a.noises}
    O = {noise: gains(combined, ORACLE[1], noise, images) for noise in a.noises}

    figures = PROJECT / "figures"
    figures.mkdir(exist_ok=True)
    x = np.arange(len(images))
    width = 0.8 / len(METHODS)
    colors = plt.cm.tab10(np.arange(len(METHODS)))

    # Figure 1: PSNR gain over window kNN, per image
    fig, axes = plt.subplots(1, len(a.noises), figsize=(6 * len(a.noises), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, noise in zip(axes, a.noises):
        for i, lab in enumerate(labels):
            ax.bar(x + (i - (len(METHODS) - 1) / 2) * width, G[noise][i], width,
                   label=lab, color=colors[i])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, images)
        ax.set_title(f"noise sigma = {noise}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("PSNR gain over window kNN (dB)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    out1 = figures / "learned_gain_per_image.png"
    fig.savefig(out1, dpi=200)
    plt.close(fig)

    # Figure 2: share of the oracle headroom (oracle gain over window kNN) that each method reaches
    fig, axes = plt.subplots(1, len(a.noises), figsize=(7 * len(a.noises), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, noise in zip(axes, a.noises):
        for i, lab in enumerate(labels):
            ax.bar(x + (i - (len(METHODS) - 1) / 2) * width, 100 * G[noise][i] / O[noise], width,
                   label=lab, color=colors[i])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, [f"{im}\n+{o:.1f} dB" for im, o in zip(images, O[noise])])
        ax.set_title(f"noise sigma = {noise}  (below each image: oracle gain)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("% of oracle gain reached")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    out2 = figures / "learned_headroom_per_image.png"
    fig.savefig(out2, dpi=200)
    plt.close(fig)

    for noise in a.noises:
        print(f"\nnoise {noise}: gain over window kNN (dB), mean over seeds")
        print(f"  {'graph':<28} " + " ".join(f"{im[:9]:>9}" for im in images) + f" {'mean':>7}")
        for lab, g in zip(labels, G[noise]):
            print(f"  {lab:<28} " + " ".join(f"{v:+9.2f}" for v in g) + f" {g.mean():+7.2f}")
        print(f"  {'oracle: clean patches':<28} " + " ".join(f"{v:+9.2f}" for v in O[noise])
              + f" {O[noise].mean():+7.2f}")
    print(f"\nSaved {out1.relative_to(PROJECT)} and {out2.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
