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
from task1.data.transforms import (
    apply_grayscale,
    apply_hue_rotation,
    apply_translation,
    apply_patch_shuffle,
)

DATA_DIR = REPO_ROOT / "data"
SUBSET_PATH = REPO_ROOT / "task1" / "data" / "test_subset_seed6304.json"
CHECKPOINT_DIR = REPO_ROOT / "task1" / "results" / "checkpoints"
RESULTS_DIR = REPO_ROOT / "task1" / "results"
CUE_CONFLICT_DIR = RESULTS_DIR / "cue_conflicts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Dataset Wrappers
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
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.intervention_fn is not None:
            img = self.intervention_fn(img)

        if self.backbone_transform is not None:
            img = self.backbone_transform(img)

        return img, target


class CueConflictDataset(Dataset):
    """Loads accepted cue-conflict images and pairs them with both content (shape)
    and style (texture) class target indices.
    """
    def __init__(self, metadata_path: Path, img_dir: Path, backbone_transform: Callable = None):
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)["samples"]
        self.img_dir = img_dir
        self.backbone_transform = backbone_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = self.img_dir / item["filename"]
        img = Image.open(img_path).convert("RGB")

        if self.backbone_transform is not None:
            img = self.backbone_transform(img)

        shape_label = item["content_label"]
        texture_label = item["style_label"]
        return img, shape_label, texture_label


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
    return {
        "acc_drop": round(baseline["top1_accuracy"] - perturbed["top1_accuracy"], 2),
        "f1_drop": round(baseline["macro_f1"] - perturbed["macro_f1"], 2),
        "conf_drop": round(baseline["mean_max_confidence"] - perturbed["mean_max_confidence"], 2),
    }


def compute_shape_bias_metrics(
    preds: np.ndarray, shape_targets: np.ndarray, texture_targets: np.ndarray
) -> Dict[str, float]:
    n_shape = int((preds == shape_targets).sum())
    n_texture = int((preds == texture_targets).sum())
    n_total = len(preds)

    decisive_total = n_shape + n_texture
    shape_bias = (n_shape / decisive_total * 100.0) if decisive_total > 0 else 0.0
    coverage = (decisive_total / n_total * 100.0) if n_total > 0 else 0.0

    return {
        "n_shape": n_shape,
        "n_texture": n_texture,
        "n_other": n_total - decisive_total,
        "total_evaluated": n_total,
        "shape_bias_pct": round(float(shape_bias), 2),
        "coverage_pct": round(float(coverage), 2),
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
# Part 2: Color Bias
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

    print(f"\nColor bias results saved to: {out_path}")
    return results


# =====================================================================
# Part 3: Shape vs. Texture (Cue Conflicts)
# =====================================================================
def evaluate_cue_conflicts(device: torch.device) -> dict:
    meta_path = CUE_CONFLICT_DIR / "cue_conflict_metadata.json"
    accepted_dir = CUE_CONFLICT_DIR / "accepted"

    if not meta_path.exists() or not accepted_dir.exists():
        raise FileNotFoundError(
            f"Cue conflict dataset not found at {CUE_CONFLICT_DIR}. "
            f"Run 'task1/data/make_cue_conflicts.py' first."
        )

    print("\n--- Evaluating Shape vs. Texture Cue Conflicts ---")
    raw_dataset = STL10(root=str(DATA_DIR), split="test", download=False)
    classes = raw_dataset.classes

    results = {}
    models_to_eval = ["resnet50", "vit_b16", "clip_vit_b32"]

    for model_name in models_to_eval:
        key = f"{model_name}_probe"
        print(f"Evaluating {key}...")
        backbone, head, transform = load_probing_system(model_name, device)
        ds = CueConflictDataset(meta_path, accepted_dir, backbone_transform=transform)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

        all_preds, shape_targets, texture_targets = [], [], []
        with torch.no_grad():
            for images, s_lbls, t_lbls in loader:
                images = images.to(device)
                logits = head(backbone(images))
                preds = logits.argmax(dim=-1).cpu().numpy()

                all_preds.extend(preds)
                shape_targets.extend(s_lbls.numpy())
                texture_targets.extend(t_lbls.numpy())

        bias_metrics = compute_shape_bias_metrics(
            np.array(all_preds), np.array(shape_targets), np.array(texture_targets)
        )
        results[key] = bias_metrics
        print(f"  {key}: Shape Bias = {bias_metrics['shape_bias_pct']}%, Coverage = {bias_metrics['coverage_pct']}%")

    key = "clip_vit_b32_zero_shot"
    print(f"Evaluating {key}...")
    clip_wrapper, transform, _ = load_clip_vit_b32()
    text_weights = clip_wrapper.get_zero_shot_weights(classes)

    ds = CueConflictDataset(meta_path, accepted_dir, backbone_transform=transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    all_preds, shape_targets, texture_targets = [], [], []
    with torch.no_grad():
        for images, s_lbls, t_lbls in loader:
            images = images.to(device)
            feats = clip_wrapper(images)
            logits = feats @ text_weights
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            shape_targets.extend(s_lbls.numpy())
            texture_targets.extend(t_lbls.numpy())

    bias_metrics = compute_shape_bias_metrics(
        np.array(all_preds), np.array(shape_targets), np.array(texture_targets)
    )
    results[key] = bias_metrics
    print(f"  {key}: Shape Bias = {bias_metrics['shape_bias_pct']}%, Coverage = {bias_metrics['coverage_pct']}%")

    out_path = RESULTS_DIR / "cue_conflict_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nCue-conflict bias metrics saved to: {out_path}\n")
    return results


# =====================================================================
# Part 4: Spatial Translation Invariance
# =====================================================================
def evaluate_translation(indices: list, device: torch.device) -> dict:
    print("\n--- Evaluating Spatial Translation Invariance ---")
    raw_dataset = STL10(root=str(DATA_DIR), split="test", download=False)
    classes = raw_dataset.classes

    deltas = [0, 8, 16, 32]
    directions = ["north", "south", "east", "west"]

    results = {
        "displacements": deltas,
        "models": {},
    }

    def run_eval(model_type: str, model_obj, head_obj, transform_fn, interv_fn):
        ds = TransformedSubset(raw_dataset, indices, intervention_fn=interv_fn, backbone_transform=transform_fn)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        if model_type == "probe":
            preds, targets, _ = run_linear_probe_inference(model_obj, head_obj, loader, device)
        else:
            preds, targets, _ = run_clip_zeroshot_inference(model_obj, classes, loader, device)
        return preds, targets

    all_configs = [
        ("resnet50_probe", "probe", "resnet50"),
        ("vit_b16_probe", "probe", "vit_b16"),
        ("clip_vit_b32_probe", "probe", "clip_vit_b32"),
        ("clip_vit_b32_zero_shot", "zero_shot", "clip_vit_b32"),
    ]

    for model_key, mode_type, model_name in all_configs:
        print(f"\nEvaluating {model_key} across translations...")
        if mode_type == "probe":
            backbone, head, transform = load_probing_system(model_name, device)
            clip_wrapper = None
        else:
            clip_wrapper, transform, _ = load_clip_vit_b32()
            backbone, head = None, None

        clean_preds, targets = run_eval(
            mode_type,
            backbone if mode_type == "probe" else clip_wrapper,
            head,
            transform,
            interv_fn=None,
        )
        clean_acc = round(float((clean_preds == targets).mean() * 100.0), 2)

        model_results = {
            "deltas": {
                "0": {
                    "accuracy": clean_acc,
                    "consistency": 100.0,
                }
            }
        }

        for delta in [8, 16, 32]:
            dir_accuracies = []
            dir_consistencies = []

            for direction in directions:
                interv_fn = lambda img, d=delta, dr=direction: apply_translation(img, shift_px=d, direction=dr)
                preds, _ = run_eval(
                    mode_type,
                    backbone if mode_type == "probe" else clip_wrapper,
                    head,
                    transform,
                    interv_fn=interv_fn,
                )

                acc = (preds == targets).mean() * 100.0
                consistency = (preds == clean_preds).mean() * 100.0

                dir_accuracies.append(acc)
                dir_consistencies.append(consistency)

            mean_acc = round(float(np.mean(dir_accuracies)), 2)
            mean_cons = round(float(np.mean(dir_consistencies)), 2)

            model_results["deltas"][str(delta)] = {
                "accuracy": mean_acc,
                "consistency": mean_cons,
                "per_direction": {
                    d: {"accuracy": round(float(a), 2), "consistency": round(float(c), 2)}
                    for d, a, c in zip(directions, dir_accuracies, dir_consistencies)
                },
            }

        results["models"][model_key] = model_results

    out_path = RESULTS_DIR / "translation_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nTranslation metrics saved to: {out_path}\n")
    return results


# =====================================================================
# Part 5: Patch Structure (4x4 Shuffling)
# =====================================================================
def evaluate_patch_shuffle(indices: list, device: torch.device) -> dict:
    baseline_path = RESULTS_DIR / "clean_baseline_metrics.json"
    if not baseline_path.exists():
        raise FileNotFoundError("Baseline metrics missing. Run with '--mode clean' first.")

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_results = json.load(f)

    print("\n--- Evaluating Patch Structure (4x4 Patch Shuffling, Seed 6304) ---")
    raw_dataset = STL10(root=str(DATA_DIR), split="test", download=False)
    classes = raw_dataset.classes

    results = {}
    models_to_eval = ["resnet50", "vit_b16", "clip_vit_b32"]

    # Fixed seed 6304 produces the identical non-identity permutation across all models
    shuffle_fn = lambda img: apply_patch_shuffle(img, grid_size=4, seed=6304)

    for model_name in models_to_eval:
        key = f"{model_name}_probe"
        print(f"Evaluating {key} on shuffled patches...")
        backbone, head, transform = load_probing_system(model_name, device)

        # 1. Clean reference predictions for consistency calculation
        clean_ds = TransformedSubset(raw_dataset, indices, intervention_fn=None, backbone_transform=transform)
        clean_loader = DataLoader(clean_ds, batch_size=64, shuffle=False, num_workers=0)
        clean_preds, targets, _ = run_linear_probe_inference(backbone, head, clean_loader, device)

        # 2. Shuffled predictions
        shuff_ds = TransformedSubset(raw_dataset, indices, intervention_fn=shuffle_fn, backbone_transform=transform)
        shuff_loader = DataLoader(shuff_ds, batch_size=64, shuffle=False, num_workers=0)
        shuff_preds, _, shuff_confs = run_linear_probe_inference(backbone, head, shuff_loader, device)

        metrics = compute_classification_metrics(shuff_preds, targets, shuff_confs)
        deltas = compute_deltas(metrics, baseline_results[key])
        consistency = round(float((shuff_preds == clean_preds).mean() * 100.0), 2)

        results[key] = {
            "metrics": metrics,
            "relative_drop": deltas,
            "consistency": consistency,
        }
        print(f"  {key}: Acc = {metrics['top1_accuracy']}% (Drop: -{deltas['acc_drop']}%), Consistency = {consistency}%")

    key = "clip_vit_b32_zero_shot"
    print(f"Evaluating {key} on shuffled patches...")
    clip_wrapper, transform, _ = load_clip_vit_b32()

    # 1. Clean reference predictions
    clean_ds = TransformedSubset(raw_dataset, indices, intervention_fn=None, backbone_transform=transform)
    clean_loader = DataLoader(clean_ds, batch_size=64, shuffle=False, num_workers=0)
    clean_preds, targets, _ = run_clip_zeroshot_inference(clip_wrapper, classes, clean_loader, device)

    # 2. Shuffled predictions
    shuff_ds = TransformedSubset(raw_dataset, indices, intervention_fn=shuffle_fn, backbone_transform=transform)
    shuff_loader = DataLoader(shuff_ds, batch_size=64, shuffle=False, num_workers=0)
    shuff_preds, _, shuff_confs = run_clip_zeroshot_inference(clip_wrapper, classes, shuff_loader, device)

    metrics = compute_classification_metrics(shuff_preds, targets, shuff_confs)
    deltas = compute_deltas(metrics, baseline_results[key])
    consistency = round(float((shuff_preds == clean_preds).mean() * 100.0), 2)

    results[key] = {
        "metrics": metrics,
        "relative_drop": deltas,
        "consistency": consistency,
    }
    print(f"  {key}: Acc = {metrics['top1_accuracy']}% (Drop: -{deltas['acc_drop']}%), Consistency = {consistency}%")

    out_path = RESULTS_DIR / "patch_shuffle_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nPatch shuffle metrics saved to: {out_path}\n")
    return results


# =====================================================================
# CLI Entry Point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Task 1 Bias Evaluation Suite")
    parser.add_argument(
        "--mode",
        type=str,
        default="shuffle",
        choices=["clean", "color", "cue_conflict", "translation", "shuffle", "all"],
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
    elif args.mode == "cue_conflict":
        evaluate_cue_conflicts(device)
    elif args.mode == "translation":
        evaluate_translation(indices, device)
    elif args.mode == "shuffle":
        evaluate_patch_shuffle(indices, device)
    elif args.mode == "all":
        evaluate_clean_baseline(indices, device)
        evaluate_color_bias(indices, device)
        evaluate_cue_conflicts(device)
        evaluate_translation(indices, device)
        evaluate_patch_shuffle(indices, device)


if __name__ == "__main__":
    main()