"""
task4/train_gcsc.py
Trains the Good Closed-Set Classifier (GCSC) baseline on CIFAR-10:
- Same 100-epoch SGD schedule as Vanilla
- Augmentation incorporates RandAugment(num_ops=2, magnitude=9)
- Saves best checkpoint to task4/checkpoints/gcsc_best.pth
"""

import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from task4.data.dataset import (
    TransformedSubset,
    get_cifar10_datasets,
    get_transforms,
)
from task4.models.resnet_cifar import ResNet18CIFAR


def set_seed(seed: int = 6304):
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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        _, logits = model(imgs)
        loss = criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        total_loss += loss.item() * targets.size(0)

    acc = (correct / total) * 100.0
    avg_loss = total_loss / total
    return acc, avg_loss


def main():
    set_seed(6304)
    device = get_device()
    print(f"=== Training GCSC ResNet-18 (RandAugment) on {device} ===")

    checkpoints_dir = Path("task4/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoints_dir / "gcsc_best.pth"
    latest_ckpt_path = checkpoints_dir / "gcsc_latest.pth"
    history_path = checkpoints_dir / "gcsc_history.json"

    train_raw, test_raw, train_idx, val_idx = get_cifar10_datasets(seed=6304)
    train_transform, eval_transform = get_transforms("randaugment")

    train_ds = TransformedSubset(train_raw, train_idx, transform=train_transform)
    val_ds = TransformedSubset(train_raw, val_idx, transform=eval_transform)
    test_ds = TransformedSubset(test_raw, np.arange(len(test_raw)), transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    print(f"GCSC Training samples: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = ResNet18CIFAR(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    start_epoch = 1
    best_val_acc = 0.0
    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    if latest_ckpt_path.is_file():
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        saved_epoch = ckpt.get("epoch", 0)
        best_val_acc = ckpt.get("val_acc", 0.0)
        start_epoch = saved_epoch + 1
        for _ in range(saved_epoch):
            scheduler.step()
        if history_path.is_file():
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)
            except Exception:
                pass
        print(f"Resuming GCSC from Epoch {start_epoch} (Best Val: {best_val_acc:.2f}%)")

    start_time = time.time()

    for epoch in range(start_epoch, 101):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            _, logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0
        val_acc, val_loss = evaluate(model, val_loader, device)

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "optimizer_state_dict": optimizer.state_dict(),
            },
            latest_ckpt_path,
        )

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                },
                best_ckpt_path,
            )

        save_str = "-> [Best Saved]" if improved else ""
        if epoch % 5 == 0 or epoch == 100 or improved:
            print(
                f"GCSC Epoch [{epoch:03d}/100] | LR: {current_lr:.5f} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%) {save_str}"
            )

    elapsed = time.time() - start_time
    print(f"\nGCSC training completed in {elapsed/60.0:.2f} minutes.")
    print(f"Best GCSC Validation Accuracy: {best_val_acc:.2f}%")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    best_checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_acc, _ = evaluate(model, test_loader, device)
    print(f"GCSC CIFAR-10 Test Closed-Set Accuracy (CSA): {test_acc:.2f}%\n")


if __name__ == "__main__":
    main()