"""
task2/recompute_separability.py
Recomputes out-of-sample Domain Separability and Proxy A-Distance
using a standard 50/50 stratified split on pooled validation features.
Updates the metrics JSON files for Source-Only, DAN, and DANN.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from shared.pacs_protocol import get_pacs_dataloaders
from task2.models.backbone import SourceOnlyClassifier


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def extract_backbone_features(model, dataloader, device):
    model.eval()
    all_feats = []
    for images, _, _ in dataloader:
        images = images.to(device)
        feats = model.backbone(images)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def evaluate_proxy_a_distance(source_feats: np.ndarray, target_feats: np.ndarray):
    """
    Evaluates Domain Separability using a standard 50/50 train/test split.
    Proxy A-distance = 2 * (1 - 2 * error) on the holdout test set.
    """
    n_samples = min(len(source_feats), len(target_feats))
    np.random.seed(6304)
    idx_s = np.random.choice(len(source_feats), n_samples, replace=False)
    idx_t = np.random.choice(len(target_feats), n_samples, replace=False)

    X = np.vstack([source_feats[idx_s], target_feats[idx_t]])
    y = np.array([0] * n_samples + [1] * n_samples)

    # Standard 50/50 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=6304, stratify=y
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, random_state=6304),
    )
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test) * 100.0
    error = 1.0 - (test_acc / 100.0)
    proxy_a_dist = 2.0 * (1.0 - 2.0 * error)
    return test_acc, max(0.0, proxy_a_dist)


def update_model_metrics(checkpoint_name: str, metrics_json_name: str, val_loaders, test_loader, device):
    ckpt_path = Path("task2/checkpoints") / checkpoint_name
    json_path = Path("task2/checkpoints") / metrics_json_name

    if not ckpt_path.exists() or not json_path.exists():
        print(f"Skipping {checkpoint_name} (file not found)")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = SourceOnlyClassifier(pretrained=False, num_classes=7).to(device)

    # Load backbone weights cleanly
    state_dict = checkpoint["model_state_dict"]
    backbone_state = {
        k.replace("backbone.", ""): v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }
    model.backbone.load_state_dict(backbone_state)

    # Extract source validation and target test features
    source_feats = [
        extract_backbone_features(model, loader, device)
        for loader in val_loaders.values()
    ]
    s_feats_all = np.vstack(source_feats)
    t_feats = extract_backbone_features(model, test_loader, device)

    # Compute out-of-sample separability
    test_acc, pad = evaluate_proxy_a_distance(s_feats_all, t_feats)

    # Update JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    data["domain_separability_acc"] = test_acc
    data["proxy_a_distance"] = pad

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[{metrics_json_name}] -> Holdout Test Domain Acc: {test_acc:.2f}% | Proxy A-Distance: {pad:.4f}")


def main():
    device = get_device()
    print("Loading data loaders...")
    _, val_loaders, test_loader = get_pacs_dataloaders("data/PACS")

    models_to_update = [
        ("source_only_best.pth", "source_only_metrics.json"),
        ("dan_best.pth", "dan_metrics.json"),
        ("dann_best.pth", "dann_metrics.json"),
    ]

    print("\n--- Recomputing Holdout Domain Separability ---")
    for ckpt, m_json in models_to_update:
        update_model_metrics(ckpt, m_json, val_loaders, test_loader, device)


if __name__ == "__main__":
    main()