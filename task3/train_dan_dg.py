"""
task3/train_dan_dg.py
Trains DAN-DG (Domain Generalization) on PACS using pairwise source-domain MMD.
Objective:
    L_DAN_DG = L_cls + (lambda_dg / 3) * sum_{e < e'} MMD^2(F(X_e), F(X_e'))
Pairs:
    (Photo, Art Painting), (Photo, Cartoon), (Art Painting, Cartoon)
Strictly target-free: Zero Sketch data is loaded or accessed during training or validation.
Natively accelerated for Apple Silicon MPS (no CPU fallback).
"""

import itertools
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from shared.pacs import PACSDataset, get_pacs_transform
from shared.pacs_protocol import (
    DEFAULT_SEED,
    SOURCE_DOMAINS,
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
    """
    Multi-Kernel MMD implemented with native matrix ops:
        ||u - v||^2 = ||u||^2 + ||v||^2 - 2 * u @ v.T
    Avoids aten::_cdist_backward, enabling 100% native Apple Silicon GPU training.
    """

    def __init__(self, multipliers=(0.5, 1.0, 2.0)):
        super().__init__()
        self.multipliers = multipliers

    def forward(self, source_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        batch_size = source_feats.size(0)
        combined = torch.cat([source_feats, target_feats], dim=0)

        # Pairwise squared L2 distances via native matrix product
        r = torch.sum(combined ** 2, dim=1, keepdim=True)
        pairwise_sq_dists = torch.clamp(
            r + r.t() - 2.0 * torch.matmul(combined, combined.t()), min=0.0
        )

        # Median pairwise squared distance (detached heuristic)
        median_dist = torch.median(pairwise_sq_dists.detach())
        if median_dist.item() == 0:
            median_dist = torch.tensor(1.0, device=combined.device)

        # Sum of 3 RBF kernels
        total_kernel = torch.zeros_like(pairwise_sq_dists)
        for mult in self.multipliers:
            gamma = mult * median_dist + 1e-8
            total_kernel += torch.exp(-pairwise_sq_dists / gamma)

        k_ss = total_kernel[:batch_size, :batch_size]
        k_tt = total_kernel[batch_size:, batch_size:]
        k_st = total_kernel[:batch_size, batch_size:]

        mmd_loss = torch.mean(k_ss) + torch.mean(k_tt) - 2.0 * torch.mean(k_st)
        return mmd_loss


class DANModel(nn.Module):
    """ResNet-18 Backbone + 7-Class Classifier Head returning (features, logits)."""

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
    """
    Yields balanced source-only batches per training step:
    - 8 Photo, 8 Art, 8 Cartoon -> 24 Source samples total
    - Zero target / Sketch access
    """

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
                imgs, lbls = batch[0], batch[1]
                source_imgs.append(imgs)
                source_labels.append(lbls)
            yield torch.cat(source_imgs, dim=0), torch.cat(source_labels, dim=0)


def get_source_dataloaders(
    dataset_root: str = "data/PACS",
    split_file_path: str = "shared/splits/pacs_sketch_seed6304.json",
    seed: int = DEFAULT_SEED,
    num_workers: int = 2,
):
    splits = build_or_load_splits(dataset_root, split_file_path, seed)

    train_transform = get_pacs_transform("train")
    eval_transform = get_pacs_transform("eval")

    # Source Training Loaders (8 samples per source domain per step)
    source_train_loaders = {}
    total_source_samples = 0
    for d in SOURCE_DOMAINS:
        ds = PACSDataset(splits["sources"][d]["train"], transform=train_transform)
        total_source_samples += len(ds)
        source_train_loaders[d] = DataLoader(
            ds,
            batch_size=8,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    steps_per_epoch = total_source_samples // 24
    train_iterator = SourceOnlyBalancedBatchIterator(
        source_train_loaders, steps_per_epoch
    )

    # Source Validation Loaders
    val_loaders = {}
    for d in SOURCE_DOMAINS:
        ds = PACSDataset(splits["sources"][d]["val"], transform=eval_transform)
        val_loaders[d] = DataLoader(
            ds,
            batch_size=32,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_iterator, val_loaders


@torch.no_grad()
def evaluate_sources(
    model: nn.Module, val_loaders: Dict[str, DataLoader], device: torch.device
):
    model.eval()
    val_accs, val_f1s = {}, {}
    for dom, loader in val_loaders.items():
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

        val_accs[dom] = accuracy_score(all_targets, all_preds) * 100.0
        val_f1s[dom] = f1_score(all_targets, all_preds, average="macro") * 100.0

    mean_acc = float(np.mean(list(val_accs.values())))
    mean_f1 = float(np.mean(list(val_f1s.values())))
    return val_accs, val_f1s, mean_acc, mean_f1


def train_dan_dg(
    dataset_root: str = "data/PACS",
    max_epochs: int = 30,
    patience: int = 5,
    lambda_dg: float = 1.0,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    save_dir: str = "task3/checkpoints",
    history_filename: str = "dan_dg_history.json",
    ckpt_filename: str = "dan_dg_best.pth",
):
    set_seed(DEFAULT_SEED)
    device = get_device()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_model_path = os.path.join(save_dir, ckpt_filename)
    history_file = os.path.join(save_dir, history_filename)

    print(f"=== Starting DAN-DG Training (lambda_dg={lambda_dg}) on {device} ===")

    # 1. Source Data Only (Zero Target / Sketch Data)
    train_iterator, val_loaders = get_source_dataloaders(
        dataset_root=dataset_root, num_workers=2
    )

    # 2. Instantiate Model and Losses
    model = DANModel(pretrained=True, num_classes=7).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_mmd = NativeMKMMD(multipliers=(0.5, 1.0, 2.0))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_source_val_f1 = 0.0
    patience_counter = 0

    history = {
        "epochs": [],
        "cls_loss": [],
        "mmd_loss": [],
        "total_loss": [],
        "source_val_acc": [],
        "source_val_f1": [],
    }

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        set_train_mode_with_frozen_bn(model)
        running_cls_loss = 0.0
        running_mmd_loss = 0.0
        running_total_loss = 0.0
        total_samples = 0

        for x_s, y_s in train_iterator:
            x_s, y_s = x_s.to(device), y_s.to(device)

            optimizer.zero_grad()

            # Forward pass on 24-sample source batch
            f_s, logits_s = model(x_s)

            # 1. Cross-Entropy loss over all 24 source samples
            loss_cls = criterion_cls(logits_s, y_s)

            # 2. Pairwise MMD across the 3 source domains
            # Ordering in batch: Photo [0:8], Art Painting [8:16], Cartoon [16:24]
            f_photo = f_s[0:8]
            f_art = f_s[8:16]
            f_cartoon = f_s[16:24]

            mmd_pa = criterion_mmd(f_photo, f_art)
            mmd_pc = criterion_mmd(f_photo, f_cartoon)
            mmd_ac = criterion_mmd(f_art, f_cartoon)

            # (lambda_dg / 3) * sum_{e < e'} MMD^2
            loss_mmd = (mmd_pa + mmd_pc + mmd_ac) / 3.0
            total_loss = loss_cls + lambda_dg * loss_mmd

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

        # Evaluate on Source Validation sets
        val_accs, val_f1s, mean_val_acc, mean_val_f1 = evaluate_sources(
            model, val_loaders, device
        )

        history["epochs"].append(epoch)
        history["cls_loss"].append(epoch_cls_loss)
        history["mmd_loss"].append(epoch_mmd_loss)
        history["total_loss"].append(epoch_total_loss)
        history["source_val_acc"].append(mean_val_acc)
        history["source_val_f1"].append(mean_val_f1)

        improved = mean_val_f1 > best_source_val_f1
        if improved:
            best_source_val_f1 = mean_val_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "source_val_acc": mean_val_acc,
                    "source_val_f1": mean_val_f1,
                    "per_domain_acc": val_accs,
                    "per_domain_f1": val_f1s,
                    "lambda_dg": lambda_dg,
                },
                best_model_path,
            )
        else:
            patience_counter += 1

        print(
            f"Epoch [{epoch:02d}/{max_epochs:02d}] | "
            f"Cls Loss: {epoch_cls_loss:.4f} | MMD Loss: {epoch_mmd_loss:.4f} | "
            f"Src Val Acc: {mean_val_acc:.2f}% | Src Val F1: {mean_val_f1:.2f}%"
            f"{' -> [Best Saved]' if improved else f' (Patience: {patience_counter}/{patience})'}"
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed / 60:.2f} mins. Best Model: {best_model_path}")

    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history to {history_file}")


if __name__ == "__main__":
    train_dan_dg(lambda_dg=1.0)