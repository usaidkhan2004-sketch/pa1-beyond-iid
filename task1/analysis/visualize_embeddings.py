import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
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


def normalize_features(arr: np.ndarray) -> np.ndarray:
    """L2-normalizes feature vectors so Euclidean distance corresponds to Cosine distance."""
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def generate_backbone_visualizations():
    device = get_device()
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        indices = json.load(f)["selected_indices"]

    raw_dataset = STL10(root=str(DATA_DIR), split="test", download=False)
    classes = raw_dataset.classes
    labels_clean = [int(raw_dataset.labels[i]) for i in indices]

    # Check cue conflict existence
    meta_path = CUE_DIR / "cue_conflict_metadata.json"
    accepted_dir = CUE_DIR / "accepted"
    has_cue = meta_path.exists() and accepted_dir.exists()
    cue_samples = []
    if has_cue:
        with open(meta_path, "r", encoding="utf-8") as f:
            cue_samples = json.load(f)["samples"]

    backbone_configs = [
        ("ResNet-50", "resnet50", load_resnet50, False),
        ("ViT-B/16", "vit_b16", load_vit_b16, False),
        ("CLIP ViT-B/32", "clip_vit_b32", load_clip_vit_b32, True),
    ]

    cmap = plt.get_cmap("tab10")

    for backbone_title, backbone_id, loader_fn, is_clip in backbone_configs:
        print(f"\n========================================================")
        print(f"Generating Joint t-SNE Projections: {backbone_title}")
        print(f"========================================================")

        model, transform, _ = loader_fn()
        model = model.to(device).eval()

        # 1. Clean embeddings
        print("Extracting clean feature embeddings...")
        clean_feats_list = []
        with torch.no_grad():
            for real_idx in indices:
                img = Image.fromarray(np.transpose(raw_dataset.data[real_idx], (1, 2, 0)))
                t_img = transform(img).unsqueeze(0).to(device)
                feat = model(t_img)
                clean_feats_list.append(feat.cpu())
        clean_feats = torch.cat(clean_feats_list, dim=0).numpy()

        # 2. Grayscale embeddings
        print("Extracting grayscale embeddings...")
        gray_feats_list = []
        with torch.no_grad():
            for real_idx in indices:
                img = apply_grayscale(Image.fromarray(np.transpose(raw_dataset.data[real_idx], (1, 2, 0))))
                t_img = transform(img).unsqueeze(0).to(device)
                feat = model(t_img)
                gray_feats_list.append(feat.cpu())
        gray_feats = torch.cat(gray_feats_list, dim=0).numpy()

        # 3. Translation embeddings (delta=16 px East)
        print("Extracting translation embeddings...")
        trans_feats_list = []
        with torch.no_grad():
            for real_idx in indices:
                img = apply_translation(
                    Image.fromarray(np.transpose(raw_dataset.data[real_idx], (1, 2, 0))),
                    shift_px=16,
                    direction="east",
                )
                t_img = transform(img).unsqueeze(0).to(device)
                feat = model(t_img)
                trans_feats_list.append(feat.cpu())
        trans_feats = torch.cat(trans_feats_list, dim=0).numpy()

        # 4. Patch Shuffle embeddings (4x4, seed 6304)
        print("Extracting patch shuffle embeddings...")
        shuff_feats_list = []
        with torch.no_grad():
            for real_idx in indices:
                img = apply_patch_shuffle(
                    Image.fromarray(np.transpose(raw_dataset.data[real_idx], (1, 2, 0))),
                    grid_size=4,
                    seed=6304,
                )
                t_img = transform(img).unsqueeze(0).to(device)
                feat = model(t_img)
                shuff_feats_list.append(feat.cpu())
        shuff_feats = torch.cat(shuff_feats_list, dim=0).numpy()

        # 5. Cue Conflict embeddings (if available)
        cue_c_feats, cue_s_feats, cue_labels = None, None, None
        if has_cue and len(cue_samples) > 0:
            print("Extracting cue-conflict embeddings...")
            c_list, s_list, lbl_list = [], [], []
            with torch.no_grad():
                for s in cue_samples:
                    content_idx = s.get("content_idx")
                    if content_idx is not None:
                        c_img = Image.fromarray(np.transpose(raw_dataset.data[content_idx], (1, 2, 0)))
                    else:
                        c_img = Image.open(accepted_dir / s["filename"]).convert("RGB")
                    s_img = Image.open(accepted_dir / s["filename"]).convert("RGB")

                    c_feat = model(transform(c_img).unsqueeze(0).to(device))
                    s_feat = model(transform(s_img).unsqueeze(0).to(device))

                    c_list.append(c_feat.cpu())
                    s_list.append(s_feat.cpu())
                    lbl_list.append(s["content_label"])
            cue_c_feats = torch.cat(c_list, dim=0).numpy()
            cue_s_feats = torch.cat(s_list, dim=0).numpy()
            cue_labels = lbl_list

        fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))

        interventions = [
            ("Grayscale", clean_feats, gray_feats, labels_clean, labels_clean),
            ("Translation (16px)", clean_feats, trans_feats, labels_clean, labels_clean),
            ("Patch Shuffle (4x4)", clean_feats, shuff_feats, labels_clean, labels_clean),
        ]
        if cue_c_feats is not None:
            interventions.append(("Cue Conflict", cue_c_feats, cue_s_feats, cue_labels, cue_labels))

        for ax_idx, (name, f_clean_cond, f_trans_cond, clean_lbls, trans_lbls) in enumerate(interventions):
            print(f"Fitting joint t-SNE for: {backbone_title} - {name}...")

            combined_raw = np.vstack([f_clean_cond, f_trans_cond])
            combined_features = normalize_features(combined_raw)
            n_clean = len(f_clean_cond)

            tsne = TSNE(
                n_components=2,
                perplexity=30,
                random_state=6304,
                max_iter=1000,
                init="pca",
                learning_rate="auto",
            )
            coords = tsne.fit_transform(combined_features)

            clean_2d = coords[:n_clean]
            trans_2d = coords[n_clean:]

            ax = axes[ax_idx]

            # Plot Clean: Circles
            for c_idx in range(10):
                mask = np.array(clean_lbls) == c_idx
                if np.any(mask):
                    ax.scatter(
                        clean_2d[mask, 0],
                        clean_2d[mask, 1],
                        c=[cmap(c_idx)],
                        marker="o",
                        s=28,
                        alpha=0.55,
                        label=classes[c_idx] if ax_idx == 0 else "",
                    )

            # Plot Transformed: Triangles
            for c_idx in range(10):
                mask = np.array(trans_lbls) == c_idx
                if np.any(mask):
                    ax.scatter(
                        trans_2d[mask, 0],
                        trans_2d[mask, 1],
                        c=[cmap(c_idx)],
                        marker="^",
                        s=32,
                        alpha=0.85,
                        edgecolors="black",
                        linewidths=0.3,
                    )

            ax.set_title(f"{name}", fontsize=13, fontweight="bold", pad=8)
            ax.set_xticks([])
            ax.set_yticks([])

        # Custom marker proxies for the legend
        from matplotlib.lines import Line2D
        marker_proxies = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="Clean"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markeredgecolor="k", markersize=8, label="Transformed"),
        ]
        leg1 = axes[0].legend(handles=marker_proxies, loc="upper right", fontsize=8.5, framealpha=0.9)
        axes[0].add_artist(leg1)
        axes[0].legend(loc="lower left", fontsize=7.5, ncol=2, framealpha=0.9)

        fig.suptitle(
            f"Joint Latent Space Manifold: {backbone_title} (Cosine t-SNE | Perplexity: 30, Seed: 6304)",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        out_png = RESULTS_DIR / f"tsne_{backbone_id}.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {out_png}")

    print("\nAll representation visualizations generated successfully.")


if __name__ == "__main__":
    generate_backbone_visualizations()