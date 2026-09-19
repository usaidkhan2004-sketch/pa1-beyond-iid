import sys
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import vgg19, VGG19_Weights
from torchvision.datasets import STL10

from task1.models.backbones import get_device

DATA_DIR = REPO_ROOT / "data"
SUBSET_PATH = REPO_ROOT / "task1" / "data" / "test_subset_seed6304.json"
OUTPUT_DIR = REPO_ROOT / "task1" / "results" / "cue_conflicts"
ACCEPTED_DIR = OUTPUT_DIR / "accepted"
REJECTED_DIR = OUTPUT_DIR / "rejected"

ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)
REJECTED_DIR.mkdir(parents=True, exist_ok=True)

# 5 Balanced Unordered Class Pairs
PAIRS = [
    ("cat", "dog"),
    ("car", "truck"),
    ("airplane", "bird"),
    ("deer", "horse"),
    ("monkey", "ship"),
]
TARGET_PER_DIRECTION = 20  # 10 directions * 20 = 200 valid images


# =====================================================================
# Style Transfer Engine (Self-Contained PyTorch VGG19)
# =====================================================================
class StyleLossNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = VGG19_Weights.DEFAULT
        vgg = vgg19(weights=weights).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        # Layers: conv1_1 (0), conv2_1 (5), conv3_1 (10), conv4_1 (19), conv5_1 (28)
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(0, 4)])   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])   # relu2_2
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 18)])  # relu3_4
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(18, 27)]) # relu4_4

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        h = (x - self.mean) / self.std
        h1 = self.slice1(h)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4]


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, h * w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def transfer_style(
    content_tensor: torch.Tensor,
    style_tensor: torch.Tensor,
    loss_net: nn.Module,
    device: torch.device,
    num_steps: int = 35,
    style_weight: float = 8e4,
    content_weight: float = 1.0,
) -> Image.Image:
    """Generates a cue-conflict image combining content silhouette and style texture."""
    target = content_tensor.clone().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([target], lr=0.04)

    with torch.no_grad():
        c_feats = loss_net(content_tensor.to(device))
        s_feats = loss_net(style_tensor.to(device))
        s_grams = [gram_matrix(sf) for sf in s_feats]

    for _ in range(num_steps):
        optimizer.zero_grad()
        t_feats = loss_net(target)

        # Content loss at relu3_4
        c_loss = F.mse_loss(t_feats[2], c_feats[2])

        # Style loss across all 4 layers
        s_loss = 0.0
        for tf, sg in zip(t_feats, s_grams):
            s_loss += F.mse_loss(gram_matrix(tf), sg)

        total_loss = (content_weight * c_loss) + (style_weight * s_loss)
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            target.clamp_(0.0, 1.0)

    res = target.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray((res * 255).astype(np.uint8))


# =====================================================================
# Visual Rejection Filter
# =====================================================================
def evaluate_rejection_rule(
    orig_content_pil: Image.Image, stylized_pil: Image.Image
) -> tuple[bool, str]:
    """Tests edge preservation and luminance range without using model predictions."""
    c_gray = np.array(orig_content_pil.convert("L"), dtype=np.float32) / 255.0
    s_gray = np.array(stylized_pil.convert("L"), dtype=np.float32) / 255.0

    # 1. Luminance standard deviation check
    lum_std = float(np.std(s_gray))
    if lum_std < 0.10:
        return False, f"low_contrast_{lum_std:.3f}"

    # 2. Sobel edge correlation check
    # Sobel kernels
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    def get_edges(arr):
        pad = np.pad(arr, 1, mode="edge")
        gx = (
            kx[0, 0] * pad[:-2, :-2] + kx[0, 2] * pad[:-2, 2:] +
            kx[1, 0] * pad[1:-1, :-2] + kx[1, 2] * pad[1:-1, 2:] +
            kx[2, 0] * pad[2:, :-2] + kx[2, 2] * pad[2:, 2:]
        )
        gy = (
            ky[0, 0] * pad[:-2, :-2] + ky[0, 1] * pad[:-2, 1:-1] + ky[0, 2] * pad[:-2, 2:] +
            ky[2, 0] * pad[2:, :-2] + ky[2, 1] * pad[2:, 1:-1] + ky[2, 2] * pad[2:, 2:]
        )
        return np.sqrt(gx**2 + gy**2)

    e_c = get_edges(c_gray).flatten()
    e_s = get_edges(s_gray).flatten()

    std_c = np.std(e_c)
    std_s = np.std(e_s)
    if std_c < 1e-6 or std_s < 1e-6:
        return False, "flat_edges"

    r_edge = float(np.corrcoef(e_c, e_s)[0, 1])
    if r_edge < 0.25:
        return False, f"edge_loss_{r_edge:.3f}"

    return True, f"accepted_corr_{r_edge:.3f}"


# =====================================================================
# Main Generator Loop
# =====================================================================
def main():
    device = get_device()
    print(f"Using compute device: {device}")

    # Clean existing directories for a fresh run
    if ACCEPTED_DIR.exists():
        shutil.rmtree(ACCEPTED_DIR)
    if REJECTED_DIR.exists():
        shutil.rmtree(REJECTED_DIR)
    ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load subset and map classes to pool of subset indices
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        subset_indices = json.load(f)["selected_indices"]

    dataset = STL10(root=str(DATA_DIR), split="test", download=False)
    classes = dataset.classes
    class_to_id = {c: i for i, c in enumerate(classes)}

    class_pools = defaultdict(list)
    for idx in subset_indices:
        label_idx = int(dataset.labels[idx])
        class_pools[classes[label_idx]].append(idx)

    # Pre-extract PIL images from subset
    subset_images = {}
    for idx in subset_indices:
        raw_arr = dataset.data[idx]
        subset_images[idx] = Image.fromarray(np.transpose(raw_arr, (1, 2, 0)))

    to_tensor = T.Compose([T.ToTensor()])
    loss_net = StyleLossNet().to(device)

    metadata = {
        "summary": {"total_accepted": 0, "total_rejected": 0},
        "pairs": {},
        "samples": [],
    }

    print("\n--- Starting Cue-Conflict Generation ---")
    image_counter = 0

    for cls_a, cls_b in PAIRS:
        pair_key = f"{cls_a}_{cls_b}"
        metadata["pairs"][pair_key] = {}
        directions = [(cls_a, cls_b), (cls_b, cls_a)]

        for c_cls, s_cls in directions:
            dir_key = f"{c_cls}_shape__{s_cls}_texture"
            print(f"\nGenerating: {dir_key} (Target: {TARGET_PER_DIRECTION})")

            c_pool = class_pools[c_cls]
            s_pool = class_pools[s_cls]

            accepted_in_dir = 0
            rejected_in_dir = 0
            candidate_idx = 0

            while accepted_in_dir < TARGET_PER_DIRECTION and candidate_idx < len(c_pool):
                c_idx = c_pool[candidate_idx]
                s_idx = s_pool[candidate_idx % len(s_pool)]
                candidate_idx += 1

                content_pil = subset_images[c_idx]
                style_pil = subset_images[s_idx]

                c_tensor = to_tensor(content_pil).unsqueeze(0)
                s_tensor = to_tensor(style_pil).unsqueeze(0)

                stylized_pil = transfer_style(c_tensor, s_tensor, loss_net, device)

                passed, reason = evaluate_rejection_rule(content_pil, stylized_pil)

                if passed:
                    accepted_in_dir += 1
                    image_counter += 1
                    fname = f"conflict_{image_counter:03d}_{c_cls}_shape_{s_cls}_texture.png"
                    stylized_pil.save(ACCEPTED_DIR / fname)

                    metadata["samples"].append({
                        "filename": fname,
                        "content_class": c_cls,
                        "content_label": class_to_id[c_cls],
                        "style_class": s_cls,
                        "style_label": class_to_id[s_cls],
                        "pair": pair_key,
                        "content_idx": c_idx,
                        "style_idx": s_idx,
                    })
                    print(f"  [{accepted_in_dir}/{TARGET_PER_DIRECTION}] Accepted: {fname} ({reason})")
                else:
                    rejected_in_dir += 1
                    r_fname = f"rejected_{c_cls}_{s_cls}_{candidate_idx}_{reason}.png"
                    stylized_pil.save(REJECTED_DIR / r_fname)
                    print(f"  Rejected candidate: {reason}")

            metadata["pairs"][pair_key][dir_key] = {
                "accepted": accepted_in_dir,
                "rejected": rejected_in_dir,
            }
            metadata["summary"]["total_accepted"] += accepted_in_dir
            metadata["summary"]["total_rejected"] += rejected_in_dir

    meta_path = OUTPUT_DIR / "cue_conflict_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=======================================================")
    print(f"Cue-Conflict Dataset Generation Complete!")
    print(f"Total Accepted: {metadata['summary']['total_accepted']}")
    print(f"Total Rejected: {metadata['summary']['total_rejected']}")
    print(f"Metadata saved to: {meta_path}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()