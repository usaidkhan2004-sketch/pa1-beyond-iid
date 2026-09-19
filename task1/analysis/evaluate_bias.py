import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
from sklearn.metrics import f1_score

# Resolve repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task1.models.backbones import (
    get_device,
    load_resnet50,
    load_vit_b16,
    load_clip_vit_b32,
)

DATA_DIR = REPO_ROOT / "data"
SUBSET_PATH = REPO_ROOT / "task1" / "data" / "test_subset_seed6304.json"
CHECKPOINT_DIR = REPO_ROOT / "task1" / "results" / "checkpoints"
RESULTS_DIR = REPO_ROOT / "task1" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Metric Computation
# =====================================================================
def compute_classification_metrics(
    preds: np.ndarray, targets: np.ndarray, confs: np.ndarray
) -> Dict[str, float]:
    """Computes Top-1 Accuracy, Macro-F1, and Mean Maximum Confidence."""
    acc = (preds == targets).mean() * 100.0
    macro_f1 = f1_score(targets, preds, average="macro") * 100.0
    mean_conf = confs.mean() * 100.0
    return {
        "top1_accuracy": round(float(acc), 2),
        "macro_f1": round(float(macro_f1), 2),
        "mean_max_confidence": round(float(mean_conf), 2),
    }


# =====================================================================
# Model Loading Helpers
# =====================================================================
def load_probing_system(
    model_name: str, device: torch.device
) -> Tuple[nn.Module, nn.Linear, object]:
    """Loads a frozen backbone along with its trained linear head."""
    factories = {
        "resnet50": (load_resnet50, 2048),
        "vit_b16": (load_vit_b16, 768),
        "clip_vit_b32": (load_clip_vit_b32, 512),
    }
    loader_fn, dim = factories[model_name]
    backbone, transform, _ = loader_fn()
    backbone = backbone.to(device).eval()

    head = nn.Linear(dim, 10).to(device)
    ckpt_path = CHECKPOINT_DIR / f"{model_name}_linear_head.pt"
    head.load_state_dict(torch.load(ckpt_path, map_location=device))
    head.eval()

    return backbone, head, transform


# =====================================================================
# Inference Engines
# =====================================================================
def run_linear_probe_inference(
    backbone: nn.Module, head: nn.Linear, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs forward pass over loader returning predictions, targets, and max probabilities."""
    all_preds, all_targets, all_confs = [], [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            feats = backbone(images)
            logits = head(feats)
            probs = F.softmax(logits, dim=-1)

            confs, preds = probs.max(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_confs.extend(confs.cpu().numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_confs)


def run_clip_zeroshot_inference(
    clip_wrapper: object,
    class_names: List[str],
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs zero-shot inference using prompt 'a photo of a {class}'."""
    text_weights = clip_wrapper.get_zero_shot_weights(class_names)
    logit_scale = clip_wrapper.model.logit_scale.exp().item()

    all_preds, all_targets, all_confs = [], [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            img_feats = clip_wrapper(images)

            # Scaled cosine similarity
            logits = logit_scale * (img_feats @ text_weights)
            probs = F.softmax(logits, dim=-1)

            confs, preds = probs.max(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_confs.extend(confs.cpu().numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_confs)


# =====================================================================
# Part 1: Clean Baseline Evaluation
# =====================================================================
def evaluate_clean_baseline(indices: list, device: torch.device) -> dict:
    """Evaluates the 3 linear probe models and zero-shot CLIP on clean 500 images."""
    print("--- Running Clean Baseline Evaluation ---")
    results = {}

    # 1. Probing models
    for model_name in ["resnet50", "vit_b16", "clip_vit_b32"]:
        print(f"Evaluating {model_name} (Linear Probe)...")
        backbone, head, transform = load_probing_system(model_name, device)

        dataset = STL10(root=str(DATA_DIR), split="test", transform=transform, download=False)
        loader = DataLoader(Subset(dataset, indices), batch_size=64, shuffle=False, num_workers=0)

        preds, targets, confs = run_linear_probe_inference(backbone, head, loader, device)
        metrics = compute_classification_metrics(preds, targets, confs)
        results[f"{model_name}_probe"] = metrics
        print(f"  {model_name}: {metrics}")

    # 2. OpenCLIP Zero-Shot
    print("Evaluating OpenCLIP ViT-B-32 (Zero-Shot)...")
    clip_wrapper, transform, _ = load_clip_vit_b32()
    dataset = STL10(root=str(DATA_DIR), split="test", transform=transform, download=False)
    loader = DataLoader(Subset(dataset, indices), batch_size=64, shuffle=False, num_workers=0)

    preds, targets, confs = run_clip_zeroshot_inference(clip_wrapper, dataset.classes, loader, device)
    zero_shot_metrics = compute_classification_metrics(preds, targets, confs)
    results["clip_vit_b32_zero_shot"] = zero_shot_metrics
    print(f"  clip_vit_b32_zero_shot: {zero_shot_metrics}")

    # Save to disk
    out_path = RESULTS_DIR / "clean_baseline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved clean baseline metrics to {out_path}")
    return results


# =====================================================================
# Entry Point & CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Task 1 Bias Evaluation Suite")
    parser.add_argument(
        "--mode",
        type=str,
        default="clean",
        choices=["clean", "color", "translation", "shuffle", "all"],
        help="Evaluation experiment to run",
    )
    args = parser.parse_args()

    device = get_device()
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        indices = json.load(f)["selected_indices"]

    if args.mode == "clean":
        evaluate_clean_baseline(indices, device)
    else:
        print(f"Intervention '{args.mode}' evaluation routines will be added as we progress through Task 1!")


if __name__ == "__main__":
    main()