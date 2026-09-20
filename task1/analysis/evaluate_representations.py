import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10
from PIL import Image

from task1.models.backbones import (
    get_device,
    load_resnet50,
    load_vit_b16,
    load_clip_vit_b32,
)
from task1.data.transforms import (
    apply_grayscale,
    apply_translation,
    apply_patch_shuffle,
)

DATA_DIR = REPO_ROOT / "data"
SUBSET_PATH = REPO_ROOT / "task1" / "data" / "test_subset_seed6304.json"
RESULTS_DIR = REPO_ROOT / "task1" / "results"
CUE_DIR = RESULTS_DIR / "cue_conflicts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Feature Extraction Helpers
# =====================================================================
def extract_features(
    model: nn.Module,
    images: torch.Tensor,
    is_clip: bool = False,
) -> torch.Tensor:
    with torch.no_grad():
        if is_clip:
            feats = model(images)
        else:
            feats = model(images)
    return feats


def compute_cosine_stability(f_clean: torch.Tensor, f_trans: torch.Tensor) -> float:
    """Computes I_T = (1/N) * sum_i (f(x_i)^T f(T(x_i))) / (||f(x_i)||_2 * ||f(T(x_i))||_2)"""
    u = F.normalize(f_clean, p=2, dim=-1)
    v = F.normalize(f_trans, p=2, dim=-1)
    cos_sim = (u * v).sum(dim=-1).mean().item()
    return round(float(cos_sim), 4)


# =====================================================================
# Main Stability Pipeline
# =====================================================================
def evaluate_stability():
    device = get_device()
    print(f"Using compute device: {device}")

    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        indices = json.load(f)["selected_indices"]

    raw_dataset = STL10(root=str(DATA_DIR), split="test", download=False)

    # Load cue conflict metadata
    meta_path = CUE_DIR / "cue_conflict_metadata.json"
    accepted_dir = CUE_DIR / "accepted"
    has_cue = meta_path.exists() and accepted_dir.exists()
    cue_samples = []
    if has_cue:
        with open(meta_path, "r", encoding="utf-8") as f:
            cue_samples = json.load(f)["samples"]

    models = {
        "resnet50": (load_resnet50, False),
        "vit_b16": (load_vit_b16, False),
        "clip_vit_b32": (load_clip_vit_b32, True),
    }

    stability_results = {}

    for model_name, (loader_fn, is_clip) in models.items():
        print(f"\n--- Extracting Representations: {model_name} ---")
        model_obj, transform, _ = loader_fn()
        model_obj = model_obj.to(device).eval()

        # 1. Clean Reference Features (N=500)
        clean_feats_list = []
        for real_idx in indices:
            raw = raw_dataset.data[real_idx]
            img = Image.fromarray(np.transpose(raw, (1, 2, 0)))
            tensor = transform(img).unsqueeze(0).to(device)
            feat = extract_features(model_obj, tensor, is_clip=is_clip)
            clean_feats_list.append(feat.cpu())
        clean_feats = torch.cat(clean_feats_list, dim=0)

        # 2. Grayscale (N=500)
        gray_feats_list = []
        for real_idx in indices:
            raw = raw_dataset.data[real_idx]
            img = apply_grayscale(Image.fromarray(np.transpose(raw, (1, 2, 0))))
            tensor = transform(img).unsqueeze(0).to(device)
            feat = extract_features(model_obj, tensor, is_clip=is_clip)
            gray_feats_list.append(feat.cpu())
        gray_feats = torch.cat(gray_feats_list, dim=0)
        i_gray = compute_cosine_stability(clean_feats, gray_feats)

        # 3. Spatial Translation (delta=16 px, average over 4 directions)
        trans_stabilities = []
        for direction in ["north", "south", "east", "west"]:
            trans_feats_list = []
            for real_idx in indices:
                raw = raw_dataset.data[real_idx]
                img = apply_translation(
                    Image.fromarray(np.transpose(raw, (1, 2, 0))),
                    shift_px=16,
                    direction=direction,
                )
                tensor = transform(img).unsqueeze(0).to(device)
                feat = extract_features(model_obj, tensor, is_clip=is_clip)
                trans_feats_list.append(feat.cpu())
            t_feats = torch.cat(trans_feats_list, dim=0)
            trans_stabilities.append(compute_cosine_stability(clean_feats, t_feats))
        i_trans = round(float(np.mean(trans_stabilities)), 4)

        # 4. Patch Shuffle (4x4, seed 6304)
        shuff_feats_list = []
        for real_idx in indices:
            raw = raw_dataset.data[real_idx]
            img = apply_patch_shuffle(
                Image.fromarray(np.transpose(raw, (1, 2, 0))),
                grid_size=4,
                seed=6304,
            )
            tensor = transform(img).unsqueeze(0).to(device)
            feat = extract_features(model_obj, tensor, is_clip=is_clip)
            shuff_feats_list.append(feat.cpu())
        shuff_feats = torch.cat(shuff_feats_list, dim=0)
        i_shuff = compute_cosine_stability(clean_feats, shuff_feats)

        # 5. Cue Conflict (Stylized vs. Content Image)
        i_cue = None
        if has_cue and len(cue_samples) > 0:
            content_feats_list = []
            stylized_feats_list = []
            for s in cue_samples:
                # Load clean content image
                content_idx = s.get("content_idx")
                if content_idx is not None:
                    raw_c = raw_dataset.data[content_idx]
                    content_img = Image.fromarray(np.transpose(raw_c, (1, 2, 0)))
                else:
                    content_img = Image.open(accepted_dir / s["filename"]).convert("RGB")

                stylized_img = Image.open(accepted_dir / s["filename"]).convert("RGB")

                c_tensor = transform(content_img).unsqueeze(0).to(device)
                s_tensor = transform(stylized_img).unsqueeze(0).to(device)

                c_feat = extract_features(model_obj, c_tensor, is_clip=is_clip)
                s_feat = extract_features(model_obj, s_tensor, is_clip=is_clip)

                content_feats_list.append(c_feat.cpu())
                stylized_feats_list.append(s_feat.cpu())

            c_feats_all = torch.cat(content_feats_list, dim=0)
            s_feats_all = torch.cat(stylized_feats_list, dim=0)
            i_cue = compute_cosine_stability(c_feats_all, s_feats_all)

        stability_results[model_name] = {
            "grayscale": i_gray,
            "translation_delta16": i_trans,
            "patch_shuffle": i_shuff,
            "cue_conflict": i_cue,
        }
        print(f"  Stability for {model_name}: {stability_results[model_name]}")

    out_json = RESULTS_DIR / "representation_stability.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stability_results, f, indent=2)

    print("\n" + "=" * 80)
    print("REPRESENTATION STABILITY (I_T) SUMMARY")
    print("=" * 80)
    header = f"| {'Backbone Architecture':<22} | {'Grayscale':<12} | {'Translation (16px)':<20} | {'Patch Shuffle':<15} | {'Cue Conflict':<14} |"
    print(header)
    print("|" + "-" * 24 + "|" + "-" * 14 + "|" + "-" * 22 + "|" + "-" * 17 + "|" + "-" * 16 + "|")
    for m, res in stability_results.items():
        cue_val = f"{res['cue_conflict']:.4f}" if res['cue_conflict'] is not None else "N/A"
        row = (
            f"| {m:<22} "
            f"| {res['grayscale']:>10.4f} "
            f"| {res['translation_delta16']:>18.4f} "
            f"| {res['patch_shuffle']:>13.4f} "
            f"| {cue_val:>12} |"
        )
        print(row)
    print("=" * 80 + "\n")
    print(f"Results saved to: {out_json}")


if __name__ == "__main__":
    evaluate_stability()