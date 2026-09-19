import sys
from pathlib import Path

# Resolve repository root (PA1)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from typing import Dict, Tuple, List, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10
from sklearn.metrics import f1_score
from PIL import Image

from task1.models.backbones import (
    get_device,
    load_resnet50,
    load_vit_b16,
    load_clip_vit_b32,
)
from task1.data.transforms import apply_grayscale, apply_hue_rotation

DATA_DIR = REPO_ROOT / "data"
SUBSET_PATH = REPO_ROOT / "task1" / "data" / "test_subset_seed6304.json"
CHECKPOINT_DIR = REPO_ROOT / "task1" / "results" / "checkpoints"
RESULTS_DIR = REPO_ROOT / "task1" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Dataset Wrapper for Interventions
# =====================================================================
class TransformedSubset(Dataset):
    """Wraps selected indices of STL-10, applying an image-level intervention
    (on PIL image) before passing to the backbone transform.
    """
    def __init__(
        self,
        base_dataset: STL10,
        indices: List[int],
        intervention_fn: Callable[[Image.Image], Image.Image] = None,
        backbone_transform: Callable = None,
    ):
        self.base_dataset = base_dataset
        self.indices = indices
        self.intervention_fn = intervention_fn
        self.backbone_transform = backbone_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, target = self.base_dataset.data[real_idx], int(self.base_dataset.labels[real_idx])
        # STL-10 numpy array shape is (3, 96, 96); transpose to (96, 96, 3) for PIL
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.intervention_fn is not None:
            img = self.intervention_fn(img)

        if self.backbone_transform is not None:
            img = self.backbone_transform(img)

        return img, target


# =====================================================================
# Metric Computation
# =====================================================================
def compute_classification_metrics(
    preds: np.ndarray, targets: np.ndarray, confs: np.ndarray
) -> Dict[str, float]:
    acc = (preds == targets).mean() * 100.0
    macro_f1 = f1_score(targets, preds, average="macro") * 100.0
    mean_conf = confs.mean() * 100.0
    return {
        "top1_accuracy": round(float(acc), 2),
        "macro_f1": round(float(macro_f1), 2),
        "mean_max_confidence": round(float(mean_conf), 2),
    }


def compute_deltas(perturbed: dict, baseline: dict) -> dict:
    """Calculates performance change: Baseline - Perturbed (positive value means a drop)."""
    return {
        "acc_drop": round(baseline["top1_accuracy"] - perturbed["top1_accuracy"], 2),
        "f1_drop": round(baseline["macro_f1"] - perturbed["macro_f1"], 2),
        "conf_drop": round(baseline["mean_max_confidence"] - perturbed["mean_max_confidence"], 2),
    }


# =====================================================================
# Model Loading Helpers
# =====================================================================
def load_probing_system(
    model_name: str, device: torch.device
) -> Tuple[nn.Module, nn.Linear, object]:
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
    text_weights = clip_wrapper.get_zero_shot_weights(class_names)
    logit_scale = clip_wrapper.model.logit_scale.exp().item()

    all_preds, all_targets, all_confs = [], [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            img_feats = clip_wrapper(images)

            logits = logit_scale * (img_feats @ text_weights)
            probs = F.softmax(logits, dim=-1)

            confs, preds = probs.max(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_confs.extend(confs.cpu().numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_confs)


# =====================================================================
# Part 1: Clean Baseline
# =====================================================================
def evaluate_clean_baseline(indices: list, device: torch.device) -> dict:
    print("--- Running Clean Baseline Evaluation ---")
    raw_dataset = STL10(root=str(DATA_DIR), split="test", download=False)
    results = {}

    for model_name in ["resnet50", "vit_b16", "clip_vit_b32"]:
        print(f"Evaluating {model_name} (Linear Probe)...")
        backbone, head, transform = load_probing_system(model_name, device)
        ds = TransformedSubset(raw_dataset, indices, intervention_fn=None, backbone_transform=transform)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

        preds, targets, confs = run_linear_probe_inference(backbone, head, loader, device)
        metrics = compute_classification_metrics(preds, targets, confs)
        results[f"{model_name}_probe"] = metrics
        print(f"  {model_name}: {metrics}")

    print("Evaluating OpenCLIP ViT-B-32 (Zero-Shot)...")
    clip_wrapper, transform, _ = load_clip_vit_b32()
    ds = TransformedSubset(raw_dataset, indices, intervention_fn=None, backbone_transform=transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    preds, targets, confs = run_clip_zeroshot_inference(clip_wrapper, raw_dataset.classes, loader, device)
    zero_shot_metrics = compute_classification_metrics(preds, targets, confs)
    results["clip_vit_b32_zero_shot"] = zero_shot_metrics
    print(f"  clip_vit_b32_zero_shot: {zero_shot_metrics}")

    out_path = RESULTS_DIR / "clean_baseline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Clean baseline metrics saved to: {out_path}\n")
    return results


# =====================================================================
# Part 2: Color Bias Evaluation
# =====================================================================
def evaluate_color_bias(indices: list, device: torch.device) -> dict:
    baseline_path = RESULTS_DIR / "clean_baseline_metrics.json"
    if not baseline_path.exists():
        raise FileNotFoundError("Baseline metrics missing. Run with '--mode clean' first.")

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_results = json.load(f)

    raw_dataset = STL10(root=str(DATA_DIR), split="test", download=False)
    interventions = {
        "grayscale": apply_grayscale,
        "hue_rotation": lambda img: apply_hue_rotation(img, hue_factor=0.5),
    }

    results = {}
    models_to_eval = ["resnet50", "vit_b16", "clip_vit_b32"]

    for interv_name, interv_fn in interventions.items():
        print(f"\n=== Evaluating Color Intervention: {interv_name.upper()} ===")
        results[interv_name] = {}

        # 1. Probed backbones
        for model_name in models_to_eval:
            key = f"{model_name}_probe"
            print(f"Evaluating {key}...")
            backbone, head, transform = load_probing_system(model_name, device)

            ds = TransformedSubset(raw_dataset, indices, intervention_fn=interv_fn, backbone_transform=transform)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

            preds, targets, confs = run_linear_probe_inference(backbone, head, loader, device)
            metrics = compute_classification_metrics(preds, targets, confs)
            deltas = compute_deltas(metrics, baseline_results[key])

            results[interv_name][key] = {"metrics": metrics, "relative_drop": deltas}
            print(f"  Metrics: {metrics}")
            print(f"  Drops vs Clean: {deltas}")

        # 2. Zero-Shot CLIP
        key = "clip_vit_b32_zero_shot"
        print(f"Evaluating {key}...")
        clip_wrapper, transform, _ = load_clip_vit_b32()

        ds = TransformedSubset(raw_dataset, indices, intervention_fn=interv_fn, backbone_transform=transform)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

        preds, targets, confs = run_clip_zeroshot_inference(clip_wrapper, raw_dataset.classes, loader, device)
        metrics = compute_classification_metrics(preds, targets, confs)
        deltas = compute_deltas(metrics, baseline_results[key])

        results[interv_name][key] = {"metrics": metrics, "relative_drop": deltas}
        print(f"  Metrics: {metrics}")
        print(f"  Drops vs Clean: {deltas}")

    out_path = RESULTS_DIR / "color_bias_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nColor bias results successfully saved to: {out_path}")
    return results


# =====================================================================
# CLI Entry Point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Task 1 Bias Evaluation Suite")
    parser.add_argument(
        "--mode",
        type=str,
        default="color",
        choices=["clean", "color", "translation", "shuffle", "all"],
        help="Experiment to run",
    )
    args = parser.parse_args()

    device = get_device()
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        indices = json.load(f)["selected_indices"]

    if args.mode == "clean":
        evaluate_clean_baseline(indices, device)
    elif args.mode == "color":
        evaluate_color_bias(indices, device)


if __name__ == "__main__":
    main()