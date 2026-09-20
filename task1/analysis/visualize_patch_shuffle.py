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

from task1.data.transforms import apply_patch_shuffle

DATA_DIR = REPO_ROOT / "data"
SUBSET_PATH = REPO_ROOT / "task1" / "data" / "test_subset_seed6304.json"
RESULTS_DIR = REPO_ROOT / "task1" / "results"


def plot_shuffle_samples(num_samples: int = 4):
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        indices = json.load(f)["selected_indices"]

    dataset = STL10(root=str(DATA_DIR), split="test", download=False)
    classes = dataset.classes

    selected_indices = indices[:num_samples]
    fig, axes = plt.subplots(2, num_samples, figsize=(3.2 * num_samples, 6.5))

    for col_idx, real_idx in enumerate(selected_indices):
        raw = dataset.data[real_idx]
        label = classes[int(dataset.labels[real_idx])]
        clean_img = Image.fromarray(np.transpose(raw, (1, 2, 0)))
        shuff_img = apply_patch_shuffle(clean_img, grid_size=4, seed=6304)

        axes[0, col_idx].imshow(clean_img)
        axes[0, col_idx].set_title(f"Clean: {label}", fontsize=11, fontweight="bold")
        axes[0, col_idx].axis("off")

        axes[1, col_idx].imshow(shuff_img)
        axes[1, col_idx].set_title(f"4x4 Shuffled (Seed 6304)", fontsize=10)
        axes[1, col_idx].axis("off")

    plt.tight_layout()
    out_path = RESULTS_DIR / "patch_shuffle_samples.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved patch shuffle qualitative samples to: {out_path}")


if __name__ == "__main__":
    plot_shuffle_samples()