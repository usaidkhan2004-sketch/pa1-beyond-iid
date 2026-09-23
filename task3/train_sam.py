"""
task3/train_sam.py
Trains ResNet-18 using Sharpness-Aware Minimization (SAM) on PACS source domains.
Objective:
    min_theta max_{||epsilon||_2 <= rho} L_ERM(theta + epsilon)
Protocol:
    - Fixed perturbation radius rho = 0.05
    - Two forward/backward passes per training step
    - Frozen BatchNorm running stats during both passes
    - Zero Sketch data access during training and validation
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
    SourceOnlyClassifier,
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


class SourceOnlyBalancedBatchIterator:
    """
    Yields balanced source-only batches:
    8 Photo + 8 Art + 8 Cartoon = 24 samples per step.
    Zero Sketch access.
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
            logits = model(imgs)
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


def train_sam(
    dataset_root: str = "data/PACS",
    max_epochs: int = 30,
    patience: int = 5,
    rho: float = 0.05,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    save_dir: str = "task3/checkpoints",
    history_filename: str = "sam_history.json",
    ckpt_filename: str = "sam_best.pth",
):
    set_seed(DEFAULT_SEED)
    device = get_device()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_model_path = os.path.join(save_dir, ckpt_filename)
    history_file = os.path.join(save_dir, history_filename)

    print(f"=== Starting SAM Training (rho={rho}) on Device: {device} ===")

    # 1. Source Data Only (Zero Sketch Access)
    train_iterator, val_loaders = get_source_dataloaders(
        dataset_root=dataset_root, num_workers=2
    )

    # 2. Instantiate Model and Optimizer
    model = SourceOnlyClassifier(pretrained=True, num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    best_source_val_f1 = 0.0
    patience_counter = 0

    history = {
        "epochs": [],
        "train_loss": [],
        "source_val_acc": [],
        "source_val_f1": [],
    }

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # Freeze BatchNorm running stats per manual
        set_train_mode_with_frozen_bn(model)
        running_loss = 0.0
        total_samples = 0

        for x_s, y_s in train_iterator:
            x_s, y_s = x_s.to(device), y_s.to(device)

            # -----------------------------------------------------------
            # Pass 1: Standard ERM loss to compute ascent direction
            # -----------------------------------------------------------
            set_train_mode_with_frozen_bn(model)
            optimizer.zero_grad()
            logits1 = model(x_s)
            loss1 = criterion(logits1, y_s)
            loss1.backward()

            # Compute L2 norm of gradients across all trainable parameters
            grad_norms = [p.grad.detach().norm(2) for p in trainable_params if p.grad is not None]
            total_grad_norm = torch.norm(torch.stack(grad_norms), 2)

            # Compute epsilon perturbation and climb to local maximum
            perturbations = {}
            scale = rho / (total_grad_norm + 1e-12)
            with torch.no_grad():
                for p in trainable_params:
                    if p.grad is not None:
                        e_w = p.grad * scale
                        p.add_(e_w)
                        perturbations[p] = e_w

            # -----------------------------------------------------------
            # Pass 2: Loss at perturbed point theta + epsilon
            # -----------------------------------------------------------
            set_train_mode_with_frozen_bn(model)
            optimizer.zero_grad()
            logits2 = model(x_s)
            loss2 = criterion(logits2, y_s)
            loss2.backward()

            # Revert parameters back to original theta
            with torch.no_grad():
                for p in trainable_params:
                    if p in perturbations:
                        p.sub_(perturbations[p])

            # Apply optimizer step on original parameters
            optimizer.step()

            batch_size = x_s.size(0)
            running_loss += loss2.item() * batch_size
            total_samples += batch_size

        epoch_loss = running_loss / total_samples

        # Evaluate on Source Validation Domains
        val_accs, val_f1s, mean_val_acc, mean_val_f1 = evaluate_sources(
            model, val_loaders, device
        )

        history["epochs"].append(epoch)
        history["train_loss"].append(epoch_loss)
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
                    "rho": rho,
                },
                best_model_path,
            )
        else:
            patience_counter += 1

        print(
            f"Epoch [{epoch:02d}/{max_epochs:02d}] | "
            f"SAM Loss: {epoch_loss:.4f} | "
            f"Src Val Acc: {mean_val_acc:.2f}% | Src Val F1: {mean_val_f1:.2f}%"
            f"{' -> [Best Saved]' if improved else f' (Patience: {patience_counter}/{patience})'}"
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    elapsed = time.time() - start_time
    print(f"SAM Training completed in {elapsed / 60:.2f} mins. Best Model: {best_model_path}")

    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Saved SAM training history to {history_file}")


if __name__ == "__main__":
    train_sam(rho=0.05)