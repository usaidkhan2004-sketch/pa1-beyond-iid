"""
task4/plot_task4.py
Generates publication figures for Task 4:
  1. task4_score_distributions.png (MSP, MLS, Mahalanobis multi-panel)
  2. task4_roc_curves.png (ROC curves comparing scores and models)
Uses cached score arrays from task4/results/osr_cache.pt
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.3,
})

FIG_DIR = Path("task4/results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_score_distributions(cache):
    scores_dict = cache["scores_dict"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    metrics = [
        ("MSP", r"Maximum Softmax Probability ($u_{\mathrm{MSP}}$)", axes[0]),
        ("MLS", r"Maximum Logit Score ($u_{\mathrm{MLS}}$)", axes[1]),
        ("Mahalanobis", r"Mahalanobis Distance ($u_{\mathrm{Mah}}$)", axes[2]),
    ]

    for name, title, ax in metrics:
        d = scores_dict[name]
        tau = np.percentile(d["val"], 95)
        
        bins = np.linspace(
            min(d["test"].min(), d["near"].min(), d["far"].min()),
            max(d["test"].max(), d["near"].max(), d["far"].max()),
            45
        )

        ax.hist(d["test"], bins=bins, density=True, alpha=0.5, color="#1f77b4", label="Known (CIFAR-10)")
        ax.hist(d["near"], bins=bins, density=True, alpha=0.5, color="#ff7f0e", label="Near Unknown")
        ax.hist(d["far"], bins=bins, density=True, alpha=0.5, color="#2ca02c", label="Far Unknown")

        ax.axvline(tau, color="red", linestyle="--", linewidth=1.5, label=r"95% Val ($\tau$)")
        ax.set_title(title)
        ax.set_xlabel("Unknownness Score")
        ax.set_ylabel("Density")
        ax.legend(frameon=True, loc="upper right")
        ax.grid(True)

    plt.tight_layout()
    out_path = FIG_DIR / "task4_score_distributions.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_roc_curves(cache):
    scores_dict = cache["scores_dict"]
    models_mls = cache["models_mls"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    # Left: Compare post-hoc scores on Frozen Vanilla (Near vs Far)
    colors = {"MSP": "#1f77b4", "MLS": "#2ca02c", "Energy": "#ff7f0e", "Mahalanobis": "#d62728"}
    for name in ["MSP", "MLS", "Energy", "Mahalanobis"]:
        d = scores_dict[name]
        # Near Unknown ROC
        y_true_near = np.concatenate([np.zeros(len(d["test"])), np.ones(len(d["near"]))])
        y_score_near = np.concatenate([d["test"], d["near"]])
        fpr, tpr, _ = roc_curve(y_true_near, y_score_near)
        roc_auc = auc(fpr, tpr) * 100.0
        ax1.plot(fpr, tpr, color=colors[name], label=f"{name} (AUC={roc_auc:.1f}%)")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax1.set_title("Near Unknown ROC (Post-hoc Scores)")
    ax1.set_xlabel("False Positive Rate (Incorrect Acceptance)")
    ax1.set_ylabel("True Positive Rate (Correct Rejection)")
    ax1.legend(frameon=True, loc="lower right")
    ax1.grid(True)

    # Right: Compare Models (Vanilla, GCSC, PROSER) with MLS on Near Unknowns
    m_colors = {"Vanilla": "#1f77b4", "GCSC": "#9467bd", "PROSER": "#2ca02c"}
    for m_name, d in models_mls.items():
        y_true = np.concatenate([np.zeros(len(d["test"])), np.ones(len(d["near"]))])
        y_score = np.concatenate([d["test"], d["near"]])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr) * 100.0
        ax2.plot(fpr, tpr, color=m_colors[m_name], label=f"{m_name} MLS (AUC={roc_auc:.1f}%)")

    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax2.set_title("Near Unknown ROC (Model Comparison via MLS)")
    ax2.set_xlabel("False Positive Rate (Incorrect Acceptance)")
    ax2.set_ylabel("True Positive Rate (Correct Rejection)")
    ax2.legend(frameon=True, loc="lower right")
    ax2.grid(True)

    plt.tight_layout()
    out_path = FIG_DIR / "task4_roc_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    cache_path = Path("task4/results/osr_cache.pt")
    if not cache_path.is_file():
        print(f"Error: {cache_path} not found. Run task4/evaluate_all.py first.")
        return

    print("Generating Task 4 publication visualizations from cache...")
    cache = torch.load(cache_path, map_location="cpu")
    plot_score_distributions(cache)
    plot_roc_curves(cache)
    print("All Task 4 plots generated successfully.")


if __name__ == "__main__":
    main()