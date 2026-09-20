import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import STL10

from task1.data.transforms import apply_translation

DATA_DIR = REPO_ROOT / "data"
SUBSET_PATH = REPO_ROOT / "task1" / "data" / "test_subset_seed6304.json"
RESULTS_DIR = REPO_ROOT / "task1" / "results"
METRICS_PATH = RESULTS_DIR / "translation_metrics.json"


def plot_translation_curves():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    deltas = [0, 8, 16, 32]
    models = data["models"]

    palette = {
        "resnet50_probe": ("#1f77b4", "o-", "ResNet-50 (Probe)"),
        "vit_b16_probe": ("#2ca02c", "s-", "ViT-B/16 (Probe)"),
        "clip_vit_b32_probe": ("#ff7f0e", "^-", "CLIP ViT-B/32 (Probe)"),
        "clip_vit_b32_zero_shot": ("#d62728", "d--", "CLIP ViT-B/32 (Zero-Shot)"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Accuracy vs Displacement
    for key, (color, marker, label) in palette.items():
        accs = [models[key]["deltas"][str(d)]["accuracy"] for d in deltas]
        axes[0].plot(deltas, accs, marker, color=color, label=label, linewidth=2, markersize=7)

    axes[0].set_title("Top-1 Accuracy vs. Displacement", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Displacement $\delta$ (pixels)", fontsize=11)
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].set_xticks(deltas)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=9)

    # Panel 2: Consistency vs Displacement
    for key, (color, marker, label) in palette.items():
        cons = [models[key]["deltas"][str(d)]["consistency"] for d in deltas]
        axes[1].plot(deltas, cons, marker, color=color, label=label, linewidth=2, markersize=7)

    axes[1].set_title("Prediction Consistency vs. Displacement", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Displacement $\delta$ (pixels)", fontsize=11)
    axes[1].set_ylabel("Consistency (%)", fontsize=11)
    axes[1].set_xticks(deltas)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    out_curve = RESULTS_DIR / "translation_curves.png"
    plt.savefig(out_curve, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved translation metric curves to: {out_curve}")


def plot_translation_samples():
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        indices = json.load(f)["selected_indices"]

    dataset = STL10(root=str(DATA_DIR), split="test", download=False)
    sample_indices = [indices[0], indices[50]]

    fig, axes = plt.subplots(len(sample_indices), 5, figsize=(15, 3.2 * len(sample_indices)))
    cols = ["Original ($\delta=0$)", "North ($\delta=16$)", "South ($\delta=16$)", "West ($\delta=16$)", "East ($\delta=16$)"]

    for row_idx, real_idx in enumerate(sample_indices):
        raw = dataset.data[real_idx]
        img = Image.fromarray(np.transpose(raw, (1, 2, 0)))

        shifted_images = [
            img,
            apply_translation(img, shift_px=16, direction="north"),
            apply_translation(img, shift_px=16, direction="south"),
            apply_translation(img, shift_px=16, direction="west"),
            apply_translation(img, shift_px=16, direction="east"),
        ]

        for col_idx in range(5):
            ax = axes[row_idx, col_idx] if len(sample_indices) > 1 else axes[col_idx]
            ax.imshow(shifted_images[col_idx])
            if row_idx == 0:
                ax.set_title(cols[col_idx], fontsize=11, fontweight="bold")
            ax.axis("off")

    plt.tight_layout()
    out_sample = RESULTS_DIR / "translation_samples.png"
    plt.savefig(out_sample, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved translation qualitative samples to: {out_sample}")


if __name__ == "__main__":
    plot_translation_curves()
    plot_translation_samples()