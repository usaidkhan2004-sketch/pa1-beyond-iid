"""
task2/train_source_only.py
Trains ResNet-18 Source-Only baseline strictly following manual specifications:
- Seed 6304
- Frozen BatchNorm running stats (train scale/bias only)
- AdamW (lr=1e-4, weight_decay=1e-4)
- Early stopping after 5 epochs without improvement in mean source-val macro-F1 (min_delta=0.0)
- 70/30 Stratified Domain Separability evaluation with seed 6304 and balanced LR(C=1.0)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# macOS File Descriptor Limit Fix
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
except Exception:
    pass

import os
import json
import time
import random
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from shared.pacs_protocol import get_pacs_dataloaders
from task2.models.backbone import SourceOnlyClassifier, set_train_mode_with_frozen_bn


def set_seed(seed: int = 6304):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, dataloader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []

    for images, labels, _ in dataloader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = float((all_preds == all_labels).mean() * 100.0)
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100.0)
    return acc, macro_f1


@torch.no_grad()
def extract_features(model: nn.Module, dataloader, device: torch.device):
    model.eval()
    all_feats = []
    for images, _, _ in dataloader:
        images = images.to(device)
        feats = model.backbone(images)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def compute_domain_separability(source_feats: np.ndarray, target_feats: np.ndarray, seed: int = 6304):
    """
    Standard diagnostic per Step 5 of manual:
    Equal numbers of source-val and target features, 70/30 split (seed 6304),
    balanced Logistic Regression (C=1.0).
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


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return True
        elif current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def train_source_only(
    data_root: str = "data/PACS",
    max_epochs: int = 30,
    patience: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    save_dir: str = "task2/checkpoints",
):
    set_seed(6304)
    device = get_device()
    print(f"Using device: {device} | Seed: 6304 | Optimizer: AdamW (lr={lr}, wd={weight_decay})")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("Loading PACS data protocol...")
    train_iterator, val_loaders_raw, target_test_loader_raw = get_pacs_dataloaders(data_root)

    val_loaders = {
        dom: DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=0)
        for dom, loader in val_loaders_raw.items()
    }
    target_test_loader = DataLoader(
        target_test_loader_raw.dataset, batch_size=64, shuffle=False, num_workers=0
    )

    # 2. Instantiate Architecture
    model = SourceOnlyClassifier(pretrained=True, num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    early_stopping = EarlyStopping(patience=patience, min_delta=0.0)
    best_source_val_f1 = 0.0
    best_model_path = os.path.join(save_dir, "source_only_best.pth")

    print(f"\n--- Starting Source-Only Training (Max: {max_epochs} Epochs, Patience: {patience}) ---")
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # Freeze BatchNorm running stats per manual
        set_train_mode_with_frozen_bn(model)
        running_loss = 0.0
        total_samples = 0

        for x_s, y_s, _ in train_iterator:
            x_s = x_s.to(device)
            y_s = y_s.to(device)

            optimizer.zero_grad()
            logits_s = model(x_s)
            loss = criterion(logits_s, y_s)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_s.size(0)
            total_samples += y_s.size(0)

        epoch_loss = running_loss / total_samples

        # 3. Source Validation
        source_metrics = {
            dom: evaluate(model, loader, device)
            for dom, loader in val_loaders.items()
        }
        avg_val_acc = float(np.mean([m[0] for m in source_metrics.values()]))
        avg_val_f1 = float(np.mean([m[1] for m in source_metrics.values()]))

        # 4. Passive Target Evaluation
        target_acc, target_f1 = evaluate(model, target_test_loader, device)

        # 5. Early Stopping check based on mean source validation Macro F1
        improved = early_stopping.step(avg_val_f1)
        if improved:
            best_source_val_f1 = avg_val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "source_val_acc": avg_val_acc,
                    "source_val_f1": avg_val_f1,
                    "target_test_acc": target_acc,
                    "target_test_f1": target_f1,
                },
                best_model_path,
            )

        print(
            f"Epoch [{epoch:02d}/{max_epochs:02d}] "
            f"Loss: {epoch_loss:.4f} | "
            f"Source Val Acc: {avg_val_acc:.1f}% | "
            f"Source Val F1: {avg_val_f1:.1f}% | "
            f"Sketch Acc: {target_acc:.1f}% | "
            f"Sketch F1: {target_f1:.1f}%"
            f"{' (Checkpoint Saved)' if improved else f' (Patience: {early_stopping.counter}/{patience})'}"
        )

        if early_stopping.early_stop:
            print(f"\n[Early Stopping Triggered] No improvement in validation F1 for {patience} epochs.")
            break

    total_time = (time.time() - start_time) / 60.0
    print("\n--- Training Complete ---")
    print(f"Total time: {total_time:.2f} minutes")

    # =======================================================
    # Final Evaluation & Evidence Logging
    # =======================================================
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    source_results = {}
    source_feats_list = []
    for dom_name, loader in val_loaders.items():
        acc, f1 = evaluate(model, loader, device)
        source_results[dom_name] = {"acc": acc, "macro_f1": f1}
        source_feats_list.append(extract_features(model, loader, device))

    mean_s_acc = float(np.mean([v["acc"] for v in source_results.values()]))
    mean_s_f1 = float(np.mean([v["macro_f1"] for v in source_results.values()]))

    t_acc, t_f1 = evaluate(model, target_test_loader, device)
    t_feats = extract_features(model, target_test_loader, device)

    # Confusion matrix and per-class accuracies
    all_preds, all_labels = [], []
    for images, labels, _ in target_test_loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    class_names = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = (cm.diagonal() / cm.sum(axis=1) * 100.0).tolist()

    # Domain separability (70/30 stratified holdout, seed 6304, balanced LR C=1.0)
    s_feats_all = np.vstack(source_feats_list)
    dom_sep = compute_domain_separability(s_feats_all, t_feats, seed=6304)

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

    out_file = os.path.join(save_dir, "source_only_metrics.json")
    with open(out_file, "w") as f:
        json.dump(metrics_record, f, indent=2)

    print("\n" + "=" * 55)
    print("           SOURCE-ONLY BASELINE RESULTS (ADAMW)")
    print("=" * 55)
    for dom, m in source_results.items():
        print(f"Source [{dom:<13}]: Acc = {m['acc']:.2f}% | F1 = {m['macro_f1']:.2f}%")
    print("-" * 55)
    print(f"Mean Source Val Acc:   {mean_s_acc:.2f}%")
    print(f"Mean Source Val F1:    {mean_s_f1:.2f}%")
    print(f"Target (Sketch) Acc:   {t_acc:.2f}%")
    print(f"Target (Sketch) F1:    {t_f1:.2f}%")
    print(f"Domain Separability:   {dom_sep:.2f}% (Chance: 50.0%)")
    print("=" * 55)
    print(f"Metrics saved to: {out_file}")


if __name__ == "__main__":
    train_source_only()