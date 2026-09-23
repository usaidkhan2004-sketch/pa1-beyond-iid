"""
task3/run_controlled_study.py
Executes Step 5 Controlled Design Study for Task 3:
Varies lambda_dg in {0.1, 1.0, 10.0} for DAN-DG.
Reuses existing lambda_dg=1.0 checkpoint and trains lambda_dg=0.1 and lambda_dg=10.0.
Computes:
  - Source Validation Acc & Macro-F1 (Mean and Worst-domain)
  - 3-Way Source-Domain Separability
  - Held-out Sketch Target Accuracy and Macro-F1
Outputs:
  - task3/results/dan_dg_controlled_study.json
"""

import itertools
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader

from shared.pacs import PACSDataset, get_pacs_transform
from shared.pacs_protocol import (
    DEFAULT_SEED,
    SOURCE_DOMAINS,
    TARGET_DOMAIN,
    build_or_load_splits,
)
from task2.models.backbone import (
    ResNet18Backbone,
    TaskClassifier,
    set_train_mode_with_frozen_bn,
)


def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class NativeMKMMD(nn.Module):
    def __init__(self, multipliers=(0.5, 1.0, 2.0)):
        super().__init__()
        self.multipliers = multipliers

    def forward(self, source_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        batch_size = source_feats.size(0)
        combined = torch.cat([source_feats, target_feats], dim=0)
        r = torch.sum(combined ** 2, dim=1, keepdim=True)
        pairwise_sq_dists = torch.clamp(
            r + r.t() - 2.0 * torch.matmul(combined, combined.t()), min=0.0
        )
        median_dist = torch.median(pairwise_sq_dists.detach())
        if median_dist.item() == 0:
            median_dist = torch.tensor(1.0, device=combined.device)

        total_kernel = torch.zeros_like(pairwise_sq_dists)
        for mult in self.multipliers:
            gamma = mult * median_dist + 1e-8
            total_kernel += torch.exp(-pairwise_sq_dists / gamma)

        k_ss = total_kernel[:batch_size, :batch_size]
        k_tt = total_kernel[batch_size:, batch_size:]
        k_st = total_kernel[:batch_size, batch_size:]
        return torch.mean(k_ss) + torch.mean(k_tt) - 2.0 * torch.mean(k_st)


class DANModel(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 7):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained)
        self.classifier = TaskClassifier(
            in_features=self.backbone.feature_dim, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        return features, logits


class SourceOnlyBalancedBatchIterator:
    def __init__(self, source_loaders: Dict[str, DataLoader], steps_per_epoch: int):
        self.source_loaders = source_loaders
        self.steps_per_epoch = steps_per_epoch
        self.source_iters = {
            d: itertools.cycle(loader) for d, loader in source_loaders.items()
        }

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            source_imgs = []
            source_labels = []
            for d in SOURCE_DOMAINS:
                batch = next(self.source_iters[d])
                source_imgs.append(batch[0])
                source_labels.append(batch[1])
            yield torch.cat(source_imgs, dim=0), torch.cat(source_labels, dim=0)


def get_dataloaders(
    dataset_root: str = "data/PACS",
    split_file_path: str = "shared/splits/pacs_sketch_seed6304.json",
    seed: int = DEFAULT_SEED,
):
    splits = build_or_load_splits(dataset_root, split_file_path, seed)
    train_transform = get_pacs_transform("train")
    eval_transform = get_pacs_transform("eval")

    source_train_loaders = {}
    total_source_samples = 0
    for d in SOURCE_DOMAINS:
        ds = PACSDataset(splits["sources"][d]["train"], transform=train_transform)
        total_source_samples += len(ds)
        source_train_loaders[d] = DataLoader(
            ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, drop_last=True
        )

    steps_per_epoch = total_source_samples // 24
    train_iterator = SourceOnlyBalancedBatchIterator(
        source_train_loaders, steps_per_epoch
    )

    val_loaders = {}
    for d in SOURCE_DOMAINS:
        ds = PACSDataset(splits["sources"][d]["val"], transform=eval_transform)
        val_loaders[d] = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    target_ds = PACSDataset(splits["target"][TARGET_DOMAIN]["all"], transform=eval_transform)
    target_loader = DataLoader(target_ds, batch_size=32, shuffle=False, num_workers=2)

    return train_iterator, val_loaders, target_loader


@torch.no_grad()
def evaluate_dataloader(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        imgs, targets = batch[0].to(device), batch[1]
        _, logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        if isinstance(targets, torch.Tensor):
            all_targets.extend(targets.cpu().numpy())
        else:
            all_targets.extend(targets)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = float(accuracy_score(all_targets, all_preds) * 100.0)
    f1 = float(f1_score(all_targets, all_preds, average="macro") * 100.0)
    return acc, f1


@torch.no_grad()
def extract_source_features(model: nn.Module, val_loaders: Dict[str, DataLoader], device: torch.device):
    model.eval()
    all_features, all_domain_labels = [], []
    for dom_idx, dom_name in enumerate(SOURCE_DOMAINS):
        for batch in val_loaders[dom_name]:
            imgs = batch[0].to(device)
            feats = model.backbone(imgs)
            all_features.append(feats.cpu().numpy())
            all_domain_labels.append(np.full(imgs.size(0), dom_idx))
    return np.concatenate(all_features, axis=0), np.concatenate(all_domain_labels, axis=0)


def compute_source_domain_separability(features: np.ndarray, domain_labels: np.ndarray, seed: int = DEFAULT_SEED):
    x_train, x_test, y_train, y_test = train_test_split(
        features, domain_labels, test_size=0.30, random_state=seed, stratify=domain_labels
    )
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver="lbfgs")
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return float(accuracy_score(y_test, y_pred) * 100.0)


def train_dan_dg_variant(
    lambda_dg: float,
    train_iterator: SourceOnlyBalancedBatchIterator,
    val_loaders: Dict[str, DataLoader],
    device: torch.device,
    save_path: str,
    max_epochs: int = 30,
    patience: int = 5,
):
    print(f"\n==========================================")
    print(f"Training DAN-DG variant: lambda_dg = {lambda_dg}")
    print(f"==========================================")

    model = DANModel(pretrained=True, num_classes=7).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_mmd = NativeMKMMD(multipliers=(0.5, 1.0, 2.0))
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_source_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        set_train_mode_with_frozen_bn(model)
        running_cls_loss = 0.0
        running_mmd_loss = 0.0
        total_samples = 0

        for x_s, y_s in train_iterator:
            x_s, y_s = x_s.to(device), y_s.to(device)
            optimizer.zero_grad()

            f_s, logits_s = model(x_s)
            loss_cls = criterion_cls(logits_s, y_s)

            mmd_pa = criterion_mmd(f_s[0:8], f_s[8:16])
            mmd_pc = criterion_mmd(f_s[0:8], f_s[16:24])
            mmd_ac = criterion_mmd(f_s[8:16], f_s[16:24])
            loss_mmd = (mmd_pa + mmd_pc + mmd_ac) / 3.0

            total_loss = loss_cls + lambda_dg * loss_mmd
            total_loss.backward()
            optimizer.step()

            batch_size = x_s.size(0)
            running_cls_loss += loss_cls.item() * batch_size
            running_mmd_loss += loss_mmd.item() * batch_size
            total_samples += batch_size

        epoch_cls_loss = running_cls_loss / total_samples
        epoch_mmd_loss = running_mmd_loss / total_samples

        # Source Validation Macro-F1
        val_f1s = []
        for d in SOURCE_DOMAINS:
            _, f1 = evaluate_dataloader(model, val_loaders[d], device)
            val_f1s.append(f1)
        mean_val_f1 = float(np.mean(val_f1s))

        improved = mean_val_f1 > best_source_val_f1
        if improved:
            best_source_val_f1 = mean_val_f1
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "lambda_dg": lambda_dg}, save_path)
        else:
            patience_counter += 1

        print(
            f"Epoch [{epoch:02d}/{max_epochs:02d}] | Cls Loss: {epoch_cls_loss:.4f} | MMD Loss: {epoch_mmd_loss:.4f} | "
            f"Src Val F1: {mean_val_f1:.2f}% {'-> [Best Saved]' if improved else f'(Patience: {patience_counter}/{patience})'}"
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break


def main():
    set_seed(DEFAULT_SEED)
    device = get_device()
    print(f"=== Running Task 3 Controlled Design Study on {device} ===")

    checkpoints_dir = Path("task3/checkpoints")
    results_dir = Path("task3/results")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_iterator, val_loaders, target_loader = get_dataloaders()

    study_configs = [
        (0.1, checkpoints_dir / "dan_dg_lambda0.1_best.pth"),
        (1.0, checkpoints_dir / "dan_dg_best.pth"),  # Reuses existing main checkpoint
        (10.0, checkpoints_dir / "dan_dg_lambda10.0_best.pth"),
    ]

    # Train variants if checkpoints do not exist
    for lambda_val, ckpt_path in study_configs:
        if lambda_val == 1.0 and ckpt_path.is_file():
            print(f"\nReusing existing main checkpoint for lambda_dg = 1.0: {ckpt_path}")
            continue

        if not ckpt_path.is_file():
            train_dan_dg_variant(
                lambda_dg=lambda_val,
                train_iterator=train_iterator,
                val_loaders=val_loaders,
                device=device,
                save_path=str(ckpt_path),
            )
        else:
            print(f"\nFound existing checkpoint for lambda_dg = {lambda_val}: {ckpt_path}")

    # Evaluate all three configurations
    print("\n=== Evaluating All Configurations for Controlled Study Table ===")
    study_results = {}

    for lambda_val, ckpt_path in study_configs:
        print(f"\n--- Evaluating lambda_dg = {lambda_val} ---")
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        model = DANModel(pretrained=False, num_classes=7).to(device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.eval()

        # 1. Source Validation Performance
        src_accs, src_f1s = {}, {}
        for d in SOURCE_DOMAINS:
            acc, f1 = evaluate_dataloader(model, val_loaders[d], device)
            src_accs[d], src_f1s[d] = acc, f1

        mean_src_acc = float(np.mean(list(src_accs.values())))
        mean_src_f1 = float(np.mean(list(src_f1s.values())))
        worst_src_acc = float(np.min(list(src_accs.values())))
        worst_src_f1 = float(np.min(list(src_f1s.values())))

        # 2. 3-Way Source Domain Separability Probe
        features, dom_labels = extract_source_features(model, val_loaders, device)
        sep_score = compute_source_domain_separability(features, dom_labels, seed=DEFAULT_SEED)

        # 3. Sketch Target Performance
        sketch_acc, sketch_f1 = evaluate_dataloader(model, target_loader, device)

        study_results[f"lambda_{lambda_val}"] = {
            "lambda_dg": lambda_val,
            "mean_source_acc": mean_src_acc,
            "mean_source_f1": mean_src_f1,
            "worst_source_acc": worst_src_acc,
            "worst_source_f1": worst_src_f1,
            "source_separability": sep_score,
            "sketch_acc": sketch_acc,
            "sketch_f1": sketch_f1,
        }

        print(f"  Mean Src Acc: {mean_src_acc:.2f}% | Worst Src Acc: {worst_src_acc:.2f}%")
        print(f"  Domain Separability: {sep_score:.2f}% (Chance=33.3%)")
        print(f"  Sketch Acc: {sketch_acc:.2f}% | Sketch F1: {sketch_f1:.2f}%")

    output_path = results_dir / "dan_dg_controlled_study.json"
    with open(output_path, "w") as f:
        json.dump(study_results, f, indent=4)
    print(f"\nSaved Controlled Study results to: {output_path}")


if __name__ == "__main__":
    main()