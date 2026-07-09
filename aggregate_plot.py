"""Aggregate all per-trial results and render the final boxplot figures.

Reads every ``results/trial_*.npz`` produced by the array job, reassembles the
raw-trial dictionaries, and writes two boxplot PNGs (noisy and noiseless). This is
the ``plt.show()`` section of the original script, turned into a headless,
file-writing step suitable for a compute node.
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")  # headless backend for cluster nodes
import matplotlib.pyplot as plt
import numpy as np

from spo_core import degrees_to_test, NUM_TASKS

RESULTS_DIR = "results"
OUT_DIR = "figures"


def load_results():
    raw_l2 = {"noiseless": {d: [] for d in degrees_to_test},
              "noisy": {d: [] for d in degrees_to_test}}
    raw_spo = {"noiseless": {d: [] for d in degrees_to_test},
               "noisy": {d: [] for d in degrees_to_test}}

    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "trial_*.npz")))
    if not files:
        raise SystemExit(f"No result files found in '{RESULTS_DIR}/'. Run the array job first.")

    for f in files:
        data = np.load(f, allow_pickle=True)
        regime = str(data["regime"])
        deg = int(data["degree"])
        raw_l2[regime][deg].append(float(data["l2_pct"]))
        raw_spo[regime][deg].append(float(data["spo_pct"]))

    n_loaded = len(files)
    if n_loaded != NUM_TASKS:
        print(f"WARNING: loaded {n_loaded}/{NUM_TASKS} trials -- some array tasks may "
              f"have failed. Plotting with what is available.")
    return raw_l2, raw_spo


def render_unified_figure(l2_dict, spo_dict, title_string):
    plt.figure(figsize=(13, 6.5))

    positions = np.arange(len(degrees_to_test)) * 4.0
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue = L2, Green = SPO+

    legend_handles = []
    for idx, deg in enumerate(degrees_to_test):
        b2 = plt.boxplot(l2_dict[deg], positions=[positions[idx] - 0.5], widths=0.6,
                         patch_artist=True,
                         boxprops=dict(facecolor=colors[0], color='#2c3e50'), whis=1.5)
        b_spo = plt.boxplot(spo_dict[deg], positions=[positions[idx] + 0.5], widths=0.6,
                            patch_artist=True,
                            boxprops=dict(facecolor=colors[2], color='#2c3e50'), whis=1.5)
        if idx == 0:
            legend_handles = [b2["boxes"][0], b_spo["boxes"][0]]

    plt.xticks(positions, [f"Degree {d}" for d in degrees_to_test], fontsize=10)
    plt.xlabel("Model Misspecification Complexity (True Polynomial Degree)",
               fontsize=11, fontweight='bold')
    plt.ylabel("Normalized Operational Regret Percentage (%)",
               fontsize=11, fontweight='bold')
    plt.title(title_string, fontsize=13, fontweight='bold', pad=15)
    plt.legend(legend_handles,
               ['L2 Loss (Lasso Regression)', 'SPO+ Loss (Decision-Aware Regularized)'],
               loc='upper left', fontsize=10)
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()


def main():
    raw_l2, raw_spo = load_results()
    os.makedirs(OUT_DIR, exist_ok=True)

    render_unified_figure(
        raw_l2["noisy"], raw_spo["noisy"],
        "Figure 1: Side-by-Side Operational Regret Under Noisy Scoring Environments (H = 0.5)")
    noisy_path = os.path.join(OUT_DIR, "regret_boxplot_noisy.png")
    plt.savefig(noisy_path, dpi=150)
    print(f"wrote {noisy_path}")

    render_unified_figure(
        raw_l2["noiseless"], raw_spo["noiseless"],
        "Figure 2: Side-by-Side Operational Regret Under Perfect Noiseless Environments")
    noiseless_path = os.path.join(OUT_DIR, "regret_boxplot_noiseless.png")
    plt.savefig(noiseless_path, dpi=150)
    print(f"wrote {noiseless_path}")


if __name__ == "__main__":
    main()
