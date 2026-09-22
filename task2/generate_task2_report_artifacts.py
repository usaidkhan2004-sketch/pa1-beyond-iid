"""
task2/generate_task2_report_artifacts.py
Aggregates all Task 2 experimental runs and generates required evidence:
1. Master Comparison Table (Markdown & LaTeX)
2. Training & Alignment Loss Curves (DAN, DANN, CDAN)
3. Per-Class Accuracy Delta Comparison
4. Controlled Study Trade-off Curves
5. Confusion Matrices for Source-Only vs Adaptation Methods
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CHECKPOINTS_DIR = Path("task2/checkpoints")
FIGURES_DIR = Path("task2/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


def load_json(filename):
    path = CHECKPOINTS_DIR / filename
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def generate_master_table():
    methods = ["source_only", "dan", "dann", "cdan"]
    display_names = {
        "source_only": "Source-Only (ERM)",
        "dan": "DAN (MMD)",
        "dann": "DANN (GRL)",
        "cdan": "CDAN (Multilinear)",
    }

    print("\n" + "=" * 115)
    print("                      TASK 2 MASTER COMPARATIVE EVALUATION TABLE")
    print("=" * 115)
    header = (
        f"{'Method':<20} | {'Photo Acc/F1':<14} | {'Art Acc/F1':<14} | {'Cartoon Acc/F1':<15} | "
        f"{'Mean Src Acc/F1':<16} | {'Sketch Acc':<11} | {'Sketch F1':<10} | {'Acc Delta':<10} | {'Dom Sep':<8}"
    )
    print(header)
    print("-" * 115)

    table_data = []
    for m in methods:
        data = load_json(f"{m}_metrics.json")
        if not data:
            continue

        p_acc, p_f1 = data["source_domains"]["photo"]["acc"], data["source_domains"]["photo"]["macro_f1"]
        a_acc, a_f1 = data["source_domains"]["art_painting"]["acc"], data["source_domains"]["art_painting"]["macro_f1"]
        c_acc, c_f1 = data["source_domains"]["cartoon"]["acc"], data["source_domains"]["cartoon"]["macro_f1"]
        m_s_acc, m_s_f1 = data["mean_source_acc"], data["mean_source_f1"]
        t_acc, t_f1 = data["target_acc"], data["target_f1"]
        delta = data["target_accuracy_change"]
        sep = data["domain_separability"]

        row = (
            f"{display_names[m]:<20} | "
            f"{p_acc:>5.1f}/{p_f1:<5.1f} | "
            f"{a_acc:>5.1f}/{a_f1:<5.1f} | "
            f"{c_acc:>5.1f}/{c_f1:<5.1f} | "
            f"{m_s_acc:>6.2f}/{m_s_f1:<6.2f} | "
            f"{t_acc:>9.2f}% | "
            f"{t_f1:>8.2f}% | "
            f"{delta:>+8.2f}% | "
            f"{sep:>6.2f}%"
        )
        print(row)
        table_data.append(data)
    print("=" * 115)


def plot_loss_curves():
    """Generates classification and domain/MMD alignment loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # DAN Curves
    dan_hist = load_json("dan_history.json")
    if dan_hist:
        ax = axes[0]
        epochs = dan_hist["epochs"]
        ax.plot(epochs, dan_hist["cls_loss"], label="Classification Loss", color="#1f77b4", lw=2)
        ax.plot(epochs, dan_hist["mmd_loss"], label="MMD Penalty", color="#ff7f0e", lw=2, linestyle="--")
        ax.set_title("DAN Optimization History", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Magnitude")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # DANN Curves
    dann_hist = load_json("dann_history.json")
    if dann_hist:
        ax = axes[1]
        epochs = dann_hist["epochs"]
        ax.plot(epochs, dann_hist["cls_loss"], label="Classification Loss", color="#1f77b4", lw=2)
        ax.plot(epochs, dann_hist["domain_loss"], label="Domain Loss", color="#2ca02c", lw=2, linestyle="--")
        ax.set_yscale("log")
        ax.set_title("DANN Optimization History (Log-Scale)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Log Scale)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # CDAN Curves
    cdan_hist = load_json("cdan_history.json")
    if cdan_hist:
        ax = axes[2]
        epochs = cdan_hist["epochs"]
        ax.plot(epochs, cdan_hist["cls_loss"], label="Classification Loss", color="#1f77b4", lw=2)
        ax.plot(epochs, cdan_hist["domain_loss"], label="Conditional Dom Loss", color="#d62728", lw=2, linestyle="--")
        ax.set_yscale("log")
        ax.set_title("CDAN Optimization History (Log-Scale)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Log Scale)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    out_path = FIGURES_DIR / "task2_loss_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_per_class_deltas():
    """Plots per-class target accuracy differences relative to Source-Only baseline."""
    base_data = load_json("source_only_metrics.json")
    dan_data = load_json("dan_metrics.json")
    cdan_data = load_json("cdan_metrics.json")

    if not (base_data and dan_data and cdan_data):
        return

    base_per_class = base_data["target_per_class_acc"]
    dan_deltas = [dan_data["target_per_class_acc"][c] - base_per_class[c] for c in CLASS_NAMES]
    cdan_deltas = [cdan_data["target_per_class_acc"][c] - base_per_class[c] for c in CLASS_NAMES]

    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, dan_deltas, width, label="DAN vs Baseline", color="#ff7f0e", alpha=0.85)
    ax.bar(x + width / 2, cdan_deltas, width, label="CDAN vs Baseline", color="#2ca02c", alpha=0.85)

    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_ylabel("Accuracy Change (%)", fontsize=11)
    ax.set_title("Per-Class Target Accuracy Change Relative to Source-Only Baseline", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    plt.tight_layout()
    out_path = FIGURES_DIR / "task2_per_class_deltas.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_controlled_study():
    """Generates trade-off curves for alignment strength study."""
    study_data = load_json("controlled_study_summary.json")
    if not study_data:
        return

    lambdas = [0.1, 1.0, 10.0]
    keys = ["0.1", "1.0", "10.0"]

    src_f1 = [study_data[k]["mean_source_f1"] for k in keys]
    target_acc = [study_data[k]["target_acc"] for k in keys]
    dom_sep = [study_data[k]["domain_separability"] for k in keys]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xscale("log")
    ax1.plot(lambdas, src_f1, marker="o", color="#1f77b4", lw=2, label="Mean Source Val F1")
    ax1.plot(lambdas, target_acc, marker="s", color="#2ca02c", lw=2, label="Target (Sketch) Acc")
    ax1.set_xlabel("MMD Loss Weight ($\\lambda_{\\mathrm{MMD}}$)", fontsize=11)
    ax1.set_ylabel("Classification Performance (%)", fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(lambdas, dom_sep, marker="^", color="#d62728", lw=2, ls="--", label="Domain Separability (70/30)")
    ax2.axhline(50.0, color="gray", ls=":", label="Chance Separability (50%)")
    ax2.set_ylabel("Domain Separability (%)", color="#d62728", fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")

    plt.title("Step 6: Alignment Strength Trade-Off ($\\lambda_{\\mathrm{MMD}}$ Study)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = FIGURES_DIR / "task2_controlled_study.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_confusion_matrices():
    """Plots side-by-side normalized confusion matrices on Target Sketch."""
    base_data = load_json("source_only_metrics.json")
    dan_data = load_json("dan_metrics.json")

    if not (base_data and dan_data):
        return

    cm_base = np.array(base_data["confusion_matrix"])
    cm_dan = np.array(dan_data["confusion_matrix"])

    # Row-normalize to percentages
    cm_base_norm = cm_base.astype("float") / cm_base.sum(axis=1)[:, np.newaxis] * 100.0
    cm_dan_norm = cm_dan.astype("float") / cm_dan.sum(axis=1)[:, np.newaxis] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm_base_norm, annot=True, fmt=".1f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0], cbar=False)
    axes[0].set_title(f"Source-Only Baseline (Sketch Acc: {base_data['target_acc']:.1f}%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True Class")

    sns.heatmap(cm_dan_norm, annot=True, fmt=".1f", cmap="Oranges", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1], cbar=False)
    axes[1].set_title(f"DAN (MMD) (Sketch Acc: {dan_data['target_acc']:.1f}%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True Class")

    plt.tight_layout()
    out_path = FIGURES_DIR / "task2_confusion_matrices.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    generate_master_table()
    plot_loss_curves()
    plot_per_class_deltas()
    plot_controlled_study()
    plot_confusion_matrices()
    print("\nAll Task 2 artifacts and figures generated in task2/figures/")


if __name__ == "__main__":
    main()