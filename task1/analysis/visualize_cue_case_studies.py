import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.datasets import STL10

from task1.models.backbones import (
    get_device,
    load_resnet50,
    load_vit_b16,
    load_clip_vit_b32,
)
from task1.analysis.evaluate_bias import load_probing_system

RESULTS_DIR = REPO_ROOT / "task1" / "results"
CUE_DIR = RESULTS_DIR / "cue_conflicts"
ACCEPTED_DIR = CUE_DIR / "accepted"
META_PATH = CUE_DIR / "cue_conflict_metadata.json"


def find_and_plot_case_studies():
    device = get_device()
    with open(META_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    # Load backbones and heads
    r_backbone, r_head, r_trans = load_probing_system("resnet50", device)
    v_backbone, v_head, v_trans = load_probing_system("vit_b16", device)
    c_backbone, c_head, c_trans = load_probing_system("clip_vit_b32", device)
    clip_zs, cz_trans, _ = load_clip_vit_b32()

    raw_ds = STL10(root=str(REPO_ROOT / "data"), split="test", download=False)
    classes = raw_ds.classes
    text_weights = clip_zs.get_zero_shot_weights(classes)

    predictions = []
    for s in samples:
        img_path = ACCEPTED_DIR / s["filename"]
        img = Image.open(img_path).convert("RGB")

        with torch.no_grad():
            r_pred = r_head(r_backbone(r_trans(img).unsqueeze(0).to(device))).argmax().item()
            v_pred = v_head(v_backbone(v_trans(img).unsqueeze(0).to(device))).argmax().item()
            c_pred = c_head(c_backbone(c_trans(img).unsqueeze(0).to(device))).argmax().item()
            cz_pred = (clip_zs(cz_trans(img).unsqueeze(0).to(device)) @ text_weights).argmax().item()

        predictions.append({
            "sample": s,
            "img": img,
            "shape": s["content_label"],
            "texture": s["style_label"],
            "shape_name": s["content_class"],
            "texture_name": s["style_class"],
            "r": r_pred,
            "v": v_pred,
            "c": c_pred,
            "cz": cz_pred,
        })

    # Select distinct cases
    shape_consensus = None
    texture_bias_case = None
    disagreement_case = None

    for p in predictions:
        # Case 1: Models agree on Shape
        if shape_consensus is None and p["r"] == p["shape"] and p["v"] == p["shape"] and p["c"] == p["shape"]:
            shape_consensus = p

        # Case 2: ViT chooses Shape while ResNet or CLIP probe falls for Texture
        if texture_bias_case is None and p["v"] == p["shape"] and p["r"] == p["texture"]:
            texture_bias_case = p

        # Case 3: Models disagree or predict neither
        if disagreement_case is None and p["v"] != p["r"] and (p["cz"] != p["shape"]):
            disagreement_case = p

        if shape_consensus and texture_bias_case and disagreement_case:
            break

    cases = [
        ("Shape Agreement (Robust)", shape_consensus or predictions[0]),
        ("ViT Shape vs. ResNet Texture Bias", texture_bias_case or predictions[1]),
        ("Model Disagreement / Indecision", disagreement_case or predictions[2]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    for idx, (title, c) in enumerate(cases):
        ax = axes[idx]
        ax.imshow(c["img"])
        info = (
            f"Ground Truth:\n"
            f"• Shape: {c['shape_name']} | Texture: {c['texture_name']}\n\n"
            f"Predictions:\n"
            f"• ResNet-50: {classes[c['r']]}\n"
            f"• ViT-B/16: {classes[c['v']]}\n"
            f"• CLIP Probe: {classes[c['c']]}\n"
            f"• CLIP Zero-Shot: {classes[c['cz']]}"
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel(info, fontsize=9.5, labelpad=8, ha="left")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_path = RESULTS_DIR / "cue_conflict_case_studies.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved cue conflict case studies to: {out_path}")


if __name__ == "__main__":
    find_and_plot_case_studies()