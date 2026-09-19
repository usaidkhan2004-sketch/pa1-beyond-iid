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

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "task1" / "results"
CUE_DIR = RESULTS_DIR / "cue_conflicts"
ACCEPTED_DIR = CUE_DIR / "accepted"
REJECTED_DIR = CUE_DIR / "rejected"
META_PATH = CUE_DIR / "cue_conflict_metadata.json"


def plot_triplets(num_examples: int = 5):
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    samples = meta["samples"]

    dataset = STL10(root=str(DATA_DIR), split="test", download=False)

    # Pick one sample from different pairs
    seen_pairs = set()
    selected = []
    for s in samples:
        if s["pair"] not in seen_pairs:
            seen_pairs.add(s["pair"])
            selected.append(s)
        if len(selected) == num_examples:
            break

    fig, axes = plt.subplots(num_examples, 3, figsize=(9, 3 * num_examples))

    for row_idx, item in enumerate(selected):
        c_raw = dataset.data[item["content_idx"]]
        s_raw = dataset.data[item["style_idx"]]

        c_img = Image.fromarray(np.transpose(c_raw, (1, 2, 0)))
        s_img = Image.fromarray(np.transpose(s_raw, (1, 2, 0)))
        conf_img = Image.open(ACCEPTED_DIR / item["filename"]).convert("RGB")

        images = [c_img, s_img, conf_img]
        titles = [
            f"Content (Shape): {item['content_class']}",
            f"Style (Texture): {item['style_class']}",
            f"Cue Conflict: {item['content_class']}/{item['style_class']}",
        ]

        for col_idx in range(3):
            ax = axes[row_idx, col_idx] if num_examples > 1 else axes[col_idx]
            ax.imshow(images[col_idx])
            if row_idx == 0:
                ax.set_title(titles[col_idx], fontsize=11, fontweight="bold")
            else:
                ax.set_title(titles[col_idx], fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    out_path = RESULTS_DIR / "cue_conflict_triplets.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved qualitative conflict triplets to: {out_path}")


def plot_rejections():
    rejected_files = list(REJECTED_DIR.glob("*.png"))
    if not rejected_files:
        print("No rejected images to visualize.")
        return

    num_to_show = min(4, len(rejected_files))
    fig, axes = plt.subplots(1, num_to_show, figsize=(3.5 * num_to_show, 3.5))

    for i in range(num_to_show):
        ax = axes[i] if num_to_show > 1 else axes
        img = Image.open(rejected_files[i]).convert("RGB")
        ax.imshow(img)
        title_str = rejected_files[i].stem.replace("rejected_", "")
        ax.set_title(title_str, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    out_path = RESULTS_DIR / "cue_conflict_rejections.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved rejection failure samples to: {out_path}")


if __name__ == "__main__":
    plot_triplets()
    plot_rejections()