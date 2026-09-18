import json
from pathlib import Path
import numpy as np
from torchvision.datasets import STL10

SEED = 6304
TOTAL_SAMPLES = 500
DATA_DIR = Path("data")
OUTPUT_PATH = Path("task1/data/test_subset_seed6304.json")


def make_test_subset():
    # 1. Download/load official STL-10 test split
    print(f"Loading official STL-10 test partition from '{DATA_DIR}'...")
    dataset = STL10(root=str(DATA_DIR), split="test", download=True)

    labels = np.array(dataset.labels)
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)

    samples_per_class = TOTAL_SAMPLES // num_classes
    print(
        f"Found {num_classes} classes. Selecting {samples_per_class} images per class using seed {SEED}..."
    )

    rng = np.random.RandomState(SEED)
    selected_indices = []

    # 2. Stratified sampling: draw exactly 50 indices per class
    for cls in sorted(unique_classes):
        cls_indices = np.where(labels == cls)[0]
        chosen = rng.choice(cls_indices, size=samples_per_class, replace=False)
        selected_indices.extend(chosen.tolist())

    # Sort indices so order is completely deterministic
    selected_indices.sort()

    metadata = {
        "seed": SEED,
        "dataset": "STL-10",
        "split": "test",
        "num_classes": int(num_classes),
        "total_selected": len(selected_indices),
        "samples_per_class": int(samples_per_class),
        "class_names": dataset.classes,
        "selected_indices": selected_indices,
    }

    # 3. Save identifiers to disk
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully saved {len(selected_indices)} indices to {OUTPUT_PATH}")


if __name__ == "__main__":
    make_test_subset()