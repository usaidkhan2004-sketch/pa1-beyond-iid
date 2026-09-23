"""
task3/plot_task3.py
Generates polished publication-ready figures for Task 3 Domain Generalization:
  1. task3_training_curves.png
  2. task3_controlled_study.png
  3. task3_sketch_per_class.png
  4. task3_diagnostics_tradeoff.png
Outputs saved to task3/results/figures/
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set aesthetic publication defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.3,
})

FIG_DIR = Path("task3/results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_curves():
    dan_path = Path("task3/checkpoints/dan_dg_history.json")
    sam_path = Path("task3/checkpoints/sam_history.json")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: DAN-DG (lambda=1.0) collapse dynamics
    if dan_path.is_file():
        with open(dan_path, "r") as f:
            dan_hist = json.load(f)
        epochs = dan_hist["epochs"]
        axes[0].plot(epochs, dan_hist["cls_loss"], "o-", color="#d62728", label="Cls Loss (CE)")
        axes[0].plot(epochs, dan_hist["mmd_loss"], "s--", color="#ff7f0e", label="MMD Penalty")
        axes[0].plot(epochs, np.array(dan_hist["source_val_acc"]) / 100.0, "^-.", color="#1f77b4", label="Src Val Acc (/100)")
        axes[0].axhline(y=1.946, color="black", linestyle=":", alpha=0.7, label=r"Chance CE ($-\ln 1/7$)")
        axes[0].set_title(r"DAN-DG ($\lambda_{\mathrm{DG}}=1.0$) Dynamics")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Metric Value")
        axes[0].set_ylim(0.0, 2.2)
        axes[0].legend(frameon=True, loc="center right")
        axes[0].grid(True)

    # Right: SAM (rho=0.05) convergence
    if sam_path.is_file():
        with open(sam_path, "r") as f:
            sam_hist = json.load(f)
        epochs = sam_hist["epochs"]
        axes[1].plot(epochs, sam_hist["train_loss"], "o-", color="#2ca02c", label="SAM Train Loss")
        ax1_val = axes[1].twinx()
        ax1_val.plot(epochs, sam_hist["source_val_f1"], "d-.", color="#9467bd", label="Src Val Macro-F1 (%)")
        axes[1].set_title(r"SAM ($\rho=0.05$) Convergence")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("SAM Training Loss", color="#2ca02c")
        ax1_val.set_ylabel("Mean Source Val Macro-F1 (%)", color="#9467bd")
        axes[1].grid(True)

    plt.tight_layout()
    out_file = FIG_DIR / "task3_training_curves.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_controlled_study():
    study_path = Path("task3/results/dan_dg_controlled_study.json")
    if not study_path.is_file():
        print(f"Skipping controlled study plot: {study_path} not found.")
        return

    with open(study_path, "r") as f:
        data = json.load(f)

    lambdas = [0.1, 1.0, 10.0]
    keys = ["lambda_0.1", "lambda_1.0", "lambda_10.0"]

    src_acc = [data[k]["mean_source_acc"] for k in keys]
    sep_acc = [data[k]["source_separability"] for k in keys]
    tgt_acc = [data[k]["sketch_acc"] for k in keys]

    x = np.arange(len(lambdas))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.8, 4.5))

    rects1 = ax.bar(x - width, src_acc, width, label="Mean Source Val Acc", color="#1f77b4")
    rects2 = ax.bar(x, sep_acc, width, label="Domain Separability (Chance=33.3%)", color="#ff7f0e")
    rects3 = ax.bar(x + width, tgt_acc, width, label="Sketch Target Acc", color="#2ca02c")

    ax.axhline(y=33.33, color="black", linestyle="--", alpha=0.5, label="Chance Separability (33.33%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(r"Controlled Study: Effect of MMD Weight ($\lambda_{\mathrm{DG}}$) in DAN-DG")
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\lambda=0.1$", r"$\lambda=1.0$", r"$\lambda=10.0$"])
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y")

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_file = FIG_DIR / "task3_controlled_study.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_sketch_per_class():
    per_class_path = Path("task3/results/task3_per_class_sketch.json")
    if not per_class_path.is_file():
        print(f"Skipping per-class plot: {per_class_path} not found.")
        return

    with open(per_class_path, "r") as f:
        data = json.load(f)

    classes = [f"Class {i}" for i in range(7)]
    class_keys = [f"class_{i}" for i in range(7)]
    pacs_names = ["Dog", "Elephant", "Giraffe", "Guitar", "Horse", "House", "Person"]

    erm_acc = [data["ERM"][k] for k in class_keys]
    sam_acc = [data["SAM"][k] for k in class_keys]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    rects1 = ax.bar(x - width/2, erm_acc, width, label="ERM Baseline", color="#7f7f7f")
    rects2 = ax.bar(x + width/2, sam_acc, width, label=r"SAM ($\rho=0.05$)", color="#2ca02c")

    ax.set_ylabel("Accuracy on Unseen Sketch (%)")
    ax.set_title("Per-Class Generalization on Sketch Target (ERM vs. SAM)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{pacs_names[i]}\n({classes[i]})" for i in range(7)])
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, loc="upper left")
    ax.grid(axis="y")

    for i in range(len(classes)):
        diff = sam_acc[i] - erm_acc[i]
        color = "green" if diff >= 0 else "red"
        sign = "+" if diff > 0 else ""
        ax.annotate(f"{sign}{diff:.1f}%",
                    xy=(x[i] + width/2, sam_acc[i]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=color, weight="bold")

    plt.tight_layout()
    out_file = FIG_DIR / "task3_sketch_per_class.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_diagnostics_tradeoff():
    table_path = Path("task3/results/task3_comparison_table.json")
    if not table_path.is_file():
        print(f"Skipping diagnostics tradeoff plot: {table_path} not found.")
        return

    with open(table_path, "r") as f:
        data = json.load(f)

    models = ["ERM", "SAM"]
    colors = ["#1f77b4", "#2ca02c"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.2))

    # Sharpness vs Sketch Accuracy
    sharpness = [data[m]["sharpness_proxy"] for m in models]
    sketch_acc = [data[m]["sketch_acc"] for m in models]

    ax1.scatter(sharpness, sketch_acc, color=colors, s=120, zorder=5)
    for i, m in enumerate(models):
        if m == "SAM":
            ax1.annotate(f"{m}\n({sketch_acc[i]:.2f}%)", (sharpness[i], sketch_acc[i]),
                         textcoords="offset points", xytext=(12, -8), ha="left", weight="bold")
        else:
            ax1.annotate(f"{m}\n({sketch_acc[i]:.2f}%)", (sharpness[i], sketch_acc[i]),
                         textcoords="offset points", xytext=(-12, 10), ha="right", weight="bold")

    ax1.set_xlabel(r"Local Sharpness Proxy ($\Delta_{\mathrm{sharp}}$) (Lower = Flatter)")
    ax1.set_ylabel("Sketch Target Accuracy (%)")
    ax1.set_title("Sharpness vs. Out-of-Distribution Transfer")
    ax1.set_ylim(64, 76)
    ax1.grid(True)

    # Domain Separability vs Sketch Accuracy
    separability = [data[m]["source_separability"] for m in models]
    ax2.scatter(separability, sketch_acc, color=colors, s=120, zorder=5)
    for i, m in enumerate(models):
        if m == "SAM":
            ax2.annotate(f"{m}\n({sketch_acc[i]:.2f}%)", (separability[i], sketch_acc[i]),
                         textcoords="offset points", xytext=(12, -8), ha="left", weight="bold")
        else:
            ax2.annotate(f"{m}\n({sketch_acc[i]:.2f}%)", (separability[i], sketch_acc[i]),
                         textcoords="offset points", xytext=(-12, 10), ha="right", weight="bold")

    ax2.set_xlabel("Source Domain Separability (%)")
    ax2.set_ylabel("Sketch Target Accuracy (%)")
    ax2.set_title("Domain Separability vs. Target Transfer")
    ax2.set_ylim(64, 76)
    ax2.grid(True)

    plt.tight_layout()
    out_file = FIG_DIR / "task3_diagnostics_tradeoff.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def main():
    print("Regenerating polished Task 3 figures...")
    plot_training_curves()
    plot_controlled_study()
    plot_sketch_per_class()
    plot_diagnostics_tradeoff()
    print("All polished Task 3 plots successfully updated.")


if __name__ == "__main__":
    main()