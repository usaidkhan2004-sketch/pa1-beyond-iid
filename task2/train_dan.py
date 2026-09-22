"""
task2/train_dan.py
Trains ResNet-18 with Deep Adaptation Network (DAN) MMD feature alignment.
Complies with manual: Seed 6304, AdamW, frozen BN, 3 RBF kernels (0.5, 1, 2 median),
loss curves history, and 70/30 domain separability diagnostic.
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
from task2.models.backbone import ResNet18Backbone, TaskClassifier, set_train_mode_with_frozen_bn
from task2.models.adaptation_modules import MultipleKernelMaximumMeanDiscrepancy


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


class DANModel(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 7):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained)
        self.classifier = TaskClassifier(in_features=self.backbone.feature_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        return features, logits


@torch.no_grad()
def evaluate_loader(model: nn.Module, dataloader, device: torch.device):
    model.eval()
    all_feats, all_preds, all_labels = [], [], []

    for images, labels, _ in dataloader:
        images = images.to(device)
        feats, logits = model(images)
        preds = logits.argmax(dim=1)

        all_feats.append(feats.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

    feats_cat = np.concatenate(all_feats, axis=0)
    preds_cat = np.concatenate(all_preds, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)

    acc = float((preds_cat == labels_cat).mean() * 100.0)
    macro_f1 = float(f1_score(labels_cat, preds_cat, average="macro", zero_division=0) * 100.0)
    return acc, macro_f1, feats_cat, preds_cat, labels_cat


def compute_domain_separability(source_feats: np.ndarray, target_feats: np.ndarray, seed: int = 6304):
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

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def train_dan(
    data_root: str = "data/PACS",
    max_epochs: int = 30,
    patience: int = 5,
    lambda_mmd: float = 1.0,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    save_dir: str = "task2/checkpoints",
    history_filename: str = "dan_history.json",
    metrics_filename: str = "dan_metrics.json",
    ckpt_filename: str = "dan_best.pth",
):
    set_seed(6304)
    device = get_device()
    print(f"Using device: {device} | Seed: 6304 | lambda_mmd: {lambda_mmd} | Optimizer: AdamW")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("Loading PACS data protocol...")
    train_iterator, val_loaders_raw, target_test_loader_raw = get_pacs_dataloaders(data_root)

    val_loaders = {
        dom: DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=0)
        for dom, loader in val_loaders_raw.items()
    }
    target_test_loader = DataLoader(
        target_test_loader_raw.dataset, batch_size=64, shuffle=False, num_workers=0
    )

    model = DANModel(pretrained=True, num_classes=7).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_mmd = MultipleKernelMaximumMeanDiscrepancy(multipliers=(0.5, 1.0, 2.0))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    early_stopping = EarlyStopping(patience=patience, min_delta=0.0)
    best_source_val_f1 = 0.0
    best_model_path = os.path.join(save_dir, ckpt_filename)
    history_file = os.path.join(save_dir, history_filename)

    history = {
        "epochs": [],
        "cls_loss": [],
        "mmd_loss": [],
        "total_loss": [],
        "source_val_acc": [],
        "source_val_f1": [],
        "target_acc": [],
        "target_f1": [],
    }

    print(f"\n--- Starting DAN Training (lambda_mmd={lambda_mmd}, Max Epochs={max_epochs}) ---")
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        set_train_mode_with_frozen_bn(model)
        running_cls_loss = 0.0
        running_mmd_loss = 0.0
        running_total_loss = 0.0
        total_samples = 0

        for x_s, y_s, x_t in train_iterator:
            x_s, y_s, x_t = x_s.to(device), y_s.to(device), x_t.to(device)

            optimizer.zero_grad()

            f_s, logits_s = model(x_s)
            f_t, _ = model(x_t)

            loss_cls = criterion_cls(logits_s, y_s)
            loss_mmd = criterion_mmd(f_s, f_t)
            total_loss = loss_cls + lambda_mmd * loss_mmd

            total_loss.backward()
            optimizer.step()

            batch_size = x_s.size(0)
            running_cls_loss += loss_cls.item() * batch_size
            running_mmd_loss += loss_mmd.item() * batch_size
            running_total_loss += total_loss.item() * batch_size
            total_samples += batch_size

        epoch_cls_loss = running_cls_loss / total_samples
        epoch_mmd_loss = running_mmd_loss / total_samples
        epoch_total_loss = running_total_loss / total_samples

        # Validation on Source Domains
        val_accs, val_f1s = {}, {}
        for dom, loader in val_loaders.items():
            acc, f1, _, _, _ = evaluate_loader(model, loader, device)
            val_accs[dom] = acc
            val_f1s[dom] = f1

        mean_val_acc = float(np.mean(list(val_accs.values())))
        mean_val_f1 = float(np.mean(list(val_f1s.values())))

        # Passive Evaluation on Target (Sketch)
        target_acc, target_f1, _, _, _ = evaluate_loader(model, target_test_loader, device)

        history["epochs"].append(epoch)
        history["cls_loss"].append(epoch_cls_loss)
        history["mmd_loss"].append(epoch_mmd_loss)
        history["total_loss"].append(epoch_total_loss)
        history["source_val_acc"].append(mean_val_acc)
        history["source_val_f1"].append(mean_val_f1)
        history["target_acc"].append(target_acc)
        history["target_f1"].append(target_f1)

        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        improved = early_stopping.step(mean_val_f1)
        if improved:
            best_source_val_f1 = mean_val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "source_val_acc": mean_val_acc,
                    "source_val_f1": mean_val_f1,
                    "target_acc": target_acc,
                    "target_f1": target_f1,
                },
                best_model_path,
            )

        print(
            f"Epoch [{epoch:02d}/{max_epochs:02d}] "
            f"Loss(Cls: {epoch_cls_loss:.4f} | MMD: {epoch_mmd_loss:.4f}) | "
            f"Source Val Acc: {mean_val_acc:.1f}% | "
            f"Source Val F1: {mean_val_f1:.1f}% | "
            f"Sketch Acc: {target_acc:.1f}% | "
            f"Sketch F1: {target_f1:.1f}%"
            f"{' (Checkpoint Saved)' if improved else f' (Patience: {early_stopping.counter}/{patience})'}"
        )

        if early_stopping.early_stop:
            print(f"\n[Early Stopping Triggered] No validation F1 improvement for {patience} epochs.")
            break

    total_time = (time.time() - start_time) / 60.0
    print(f"\n--- DAN Training Finished in {total_time:.2f} minutes ---")

    # =======================================================
    # Final Detailed Evaluation of Best Checkpoint
    # =======================================================
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    source_metrics = {}
    source_feats_list = []
    for dom, loader in val_loaders.items():
        acc, f1, feats, _, _ = evaluate_loader(model, loader, device)
        source_metrics[dom] = {"acc": acc, "macro_f1": f1}
        source_feats_list.append(feats)

    mean_s_acc = float(np.mean([v["acc"] for v in source_metrics.values()]))
    mean_s_f1 = float(np.mean([v["macro_f1"] for v in source_metrics.values()]))

    t_acc, t_f1, t_feats, t_preds, t_labels = evaluate_loader(model, target_test_loader, device)

    class_names = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
    cm = confusion_matrix(t_labels, t_preds)
    per_class_acc = (cm.diagonal() / cm.sum(axis=1) * 100.0).tolist()

    s_feats_all = np.vstack(source_feats_list)
    dom_sep = compute_domain_separability(s_feats_all, t_feats, seed=6304)

    baseline_path = os.path.join(save_dir, "source_only_metrics.json")
    baseline_acc = 0.0
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline_acc = json.load(f).get("target_acc", 0.0)

    acc_change = t_acc - baseline_acc

    final_results = {
        "source_domains": source_metrics,
        "mean_source_acc": mean_s_acc,
        "mean_source_f1": mean_s_f1,
        "target_acc": t_acc,
        "target_f1": t_f1,
        "target_accuracy_change": acc_change,
        "domain_separability": dom_sep,
        "target_per_class_acc": dict(zip(class_names, per_class_acc)),
        "confusion_matrix": cm.tolist(),
    }

    metrics_file = os.path.join(save_dir, metrics_filename)
    with open(metrics_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print("\n" + "=" * 55)
    print("                    DAN RESULTS TABLE")
    print("=" * 55)
    for dom, m in source_metrics.items():
        print(f"Source [{dom:<13}]: Acc = {m['acc']:.2f}% | F1 = {m['macro_f1']:.2f}%")
    print("-" * 55)
    print(f"Mean Source Val Acc:   {mean_s_acc:.2f}%")
    print(f"Mean Source Val F1:    {mean_s_f1:.2f}%")
    print(f"Target (Sketch) Acc:   {t_acc:.2f}% (Change vs Baseline: {acc_change:+.2f}%)")
    print(f"Target (Sketch) F1:    {t_f1:.2f}%")
    print(f"Domain Separability:   {dom_sep:.2f}% (Chance: 50.0%)")
    print("=" * 55)
    print(f"All metrics recorded to: {metrics_file}")


if __name__ == "__main__":
    train_dan()