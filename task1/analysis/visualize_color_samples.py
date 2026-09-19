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

from task1.data.transforms import apply_grayscale, apply_hue_rotation

DATA_DIR = REPO_ROOT / "data"
SUBSET_PATH = REPO_ROOT / "task1" / "data" / "test_subset_seed6304.json"
OUTPUT_DIR = REPO_ROOT / "task1" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_color_visualization(num_samples: int = 4):
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        indices = json.load(f)["selected_indices"]

    dataset = STL10(root=str(DATA_DIR), split="test", download=False)
    classes = dataset.classes

    # Pick samples across different classes
    seen_classes = set()
    selected_indices = []
    for idx in indices:
        label = int(dataset.labels[idx])
        if label not in seen_classes:
            seen_classes.add(label)
            selected_indices.append(idx)
        if len(selected_indices) == num_samples:
            break

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

    for row_idx, real_idx in enumerate(selected_indices):
        raw_arr = dataset.data[real_idx]
        target_name = classes[int(dataset.labels[real_idx])]

        # Transpose (3, 96, 96) -> (96, 96, 3)
        orig_img = Image.fromarray(np.transpose(raw_arr, (1, 2, 0)))
        gray_img = apply_grayscale(orig_img)
        hue_img = apply_hue_rotation(orig_img, hue_factor=0.5)

        images = [orig_img, gray_img, hue_img]
        titles = [f"Clean: {target_name}", "Grayscale", "Hue Rotation (180°)"]

        for col_idx in range(3):
            ax = axes[row_idx, col_idx] if num_samples > 1 else axes[col_idx]
            ax.imshow(images[col_idx])
            if row_idx == 0:
                ax.set_title(titles[col_idx], fontsize=12, fontweight="bold")
            else:
                if col_idx == 0:
                    ax.set_title(titles[0], fontsize=11)
            ax.axis("off")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "color_interventions_samples.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved qualitative color figure to: {out_file}")


if __name__ == "__main__":
    generate_color_visualization()