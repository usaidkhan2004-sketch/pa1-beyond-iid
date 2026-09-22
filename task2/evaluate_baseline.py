"""
task2/evaluate_baseline.py
Independent evaluation script for any saved checkpoint.
Computes per-domain source metrics, target performance, per-class accuracies,
confusion matrix, and the official 70/30 stratified domain separability score (seed 6304).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from shared.pacs_protocol import get_pacs_dataloaders
from task2.models.backbone import SourceOnlyClassifier


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def extract_features_and_preds(model, dataloader, device):
    model.eval()
    all_feats, all_preds, all_labels = [], [], []

    for images, labels, _ in dataloader:
        images = images.to(device)
        feats = model.backbone(images)
        logits = model.classifier(feats)
        preds = logits.argmax(dim=1)

        all_feats.append(feats.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_feats, axis=0),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def compute_domain_separability(source_feats: np.ndarray, target_feats: np.ndarray, seed: int = 6304):
    """
    Step 5 Protocol:
    Equal numbers of source-val and target features, 70/30 split using seed 6304,
    balanced LogisticRegression(C=1.0).
    """
    n_samples = min(len(source_feats), len(target_feats))
    np.random.seed(seed)
    idx_s = np.random.choice(len(source_feats), n_samples, replace=False)
    idx_t = np.random.choice(len(target_feats), n_samples, replace=False)

    X = np.vstack([source_feats[idx_s], target_feats[idx_t]])
    y = np.array([0] * n_samples + [1] * n_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)

    heldout_acc = float(clf.score(X_test, y_test) * 100.0)
    return heldout_acc


def main():
    device = get_device()
    checkpoint_path = "task2/checkpoints/source_only_best.pth"

    print("Loading PACS data protocol...")
    _, val_loaders_raw, test_loader_raw = get_pacs_dataloaders("data/PACS")

    val_loaders = {
        dom: DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=0)
        for dom, loader in val_loaders_raw.items()
    }
    test_loader = DataLoader(
        test_loader_raw.dataset, batch_size=64, shuffle=False, num_workers=0
    )

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SourceOnlyClassifier(pretrained=False, num_classes=7).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

    print("\n" + "=" * 50)
    print("--- 1. SOURCE VALIDATION BREAKDOWN ---")
    print("=" * 50)
    source_results = {}
    all_source_feats = []

    for dom_name, loader in val_loaders.items():
        feats, preds, labels = extract_features_and_preds(model, loader, device)
        all_source_feats.append(feats)
        acc = float((preds == labels).mean() * 100.0)
        f1 = float(f1_score(labels, preds, average="macro", zero_division=0) * 100.0)
        source_results[dom_name] = {"acc": acc, "macro_f1": f1}
        print(f"Domain: {dom_name:<15} | Acc: {acc:.2f}% | Macro-F1: {f1:.2f}%")

    mean_s_acc = float(np.mean([v["acc"] for v in source_results.values()]))
    mean_s_f1 = float(np.mean([v["macro_f1"] for v in source_results.values()]))
    print("-" * 50)
    print(f"Mean Source Val        | Acc: {mean_s_acc:.2f}% | Macro-F1: {mean_s_f1:.2f}%")

    print("\n" + "=" * 50)
    print("--- 2. TARGET (SKETCH) DETAILED PERFORMANCE ---")
    print("=" * 50)
    t_feats, t_preds, t_labels = extract_features_and_preds(model, test_loader, device)
    t_acc = float((t_preds == t_labels).mean() * 100.0)
    t_f1 = float(f1_score(t_labels, t_preds, average="macro", zero_division=0) * 100.0)
    print(f"Target Overall Acc:    {t_acc:.2f}%")
    print(f"Target Overall F1:     {t_f1:.2f}%\n")

    print("Per-Class Accuracy on Target (Sketch):")
    cm = confusion_matrix(t_labels, t_preds)
    per_class_acc = (cm.diagonal() / cm.sum(axis=1) * 100.0).tolist()
    for name, acc in zip(class_names, per_class_acc):
        print(f"  - {name:<10}: {acc:.2f}%")

    print("\n" + "=" * 50)
    print("--- 3. DOMAIN SEPARABILITY (70/30 HELDOUT) ---")
    print("=" * 50)
    s_feats_all = np.vstack(all_source_feats)
    dom_sep = compute_domain_separability(s_feats_all, t_feats, seed=6304)
    print(f"Domain Separability Score: {dom_sep:.2f}% (Chance is 50.0%)")

    metrics_record = {
        "source_domains": source_results,
        "mean_source_acc": mean_s_acc,
        "mean_source_f1": mean_s_f1,
        "target_acc": t_acc,
        "target_f1": t_f1,
        "target_accuracy_change": 0.0,
        "domain_separability": dom_sep,
        "target_per_class_acc": dict(zip(class_names, per_class_acc)),
        "confusion_matrix": cm.tolist(),
    }

    out_file = "task2/checkpoints/source_only_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics_record, f, indent=2)
    print(f"\nAll baseline metrics saved to: {out_file}")


if __name__ == "__main__":
    main()