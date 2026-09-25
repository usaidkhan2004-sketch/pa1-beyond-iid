"""
task4/train_proser.py
Trains PROSER on CIFAR-10:
- Initialized from task4/checkpoints/vanilla_best.pth
- 50 epochs, SGD (lr=1e-3, momentum=0.9, weight_decay=5e-4), Cosine Annealing
- Mini-batch split 50/50: half for classifier placeholder, half for layer2 manifold mixup
- Checkpoint selected strictly by CIFAR-10 validation accuracy on known logits
- Saves to task4/checkpoints/proser_best.pth
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
from torch.distributions.beta import Beta
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from task4.data.dataset import (
    TransformedSubset,
    get_cifar10_datasets,
    get_transforms,
)
from task4.methods.proser_loss import PROSERLoss
from task4.models.proser_model import PROSERNet
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
def evaluate_known(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        _, logits = model(imgs)
        known_logits = logits[:, :10]  # CSA evaluated strictly on 10 known classes
        loss = criterion(known_logits, targets)

        preds = torch.argmax(known_logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        total_loss += loss.item() * targets.size(0)

    acc = (correct / total) * 100.0
    avg_loss = total_loss / total
    return acc, avg_loss


def main():
    set_seed(6304)
    device = get_device()
    print(f"=== Training PROSER (Placeholders + Manifold Mixup) on {device} ===")

    vanilla_ckpt_path = Path("task4/checkpoints/vanilla_best.pth")
    if not vanilla_ckpt_path.is_file():
        print(f"Error: {vanilla_ckpt_path} not found. Complete Vanilla training first.")
        return

    best_ckpt_path = Path("task4/checkpoints/proser_best.pth")
    history_path = Path("task4/checkpoints/proser_history.json")

    # Load Vanilla Backbone
    vanilla_model = ResNet18CIFAR(num_classes=10)
    vanilla_ckpt = torch.load(vanilla_ckpt_path, map_location="cpu")
    vanilla_model.load_state_dict(vanilla_ckpt["model_state_dict"])

    # Instantiate PROSER model with 5 dummy classes
    model = PROSERNet(vanilla_model, num_known=10, num_dummy=5).to(device)

    train_raw, test_raw, train_idx, val_idx = get_cifar10_datasets(seed=6304)
    train_transform, eval_transform = get_transforms("vanilla")

    train_ds = TransformedSubset(train_raw, train_idx, transform=train_transform)
    val_ds = TransformedSubset(train_raw, val_idx, transform=eval_transform)
    test_ds = TransformedSubset(test_raw, np.arange(len(test_raw)), transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    criterion = PROSERLoss(num_known=10, num_dummy=5, beta=1.0, gamma=0.1)
    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    beta_dist = Beta(2.0, 2.0)

    best_val_acc = 0.0
    history = {"epochs": [], "total_loss": [], "cls_loss": [], "cp_loss": [], "dp_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(1, 51):
        model.train()
        running_tot = 0.0
        running_cls = 0.0
        running_cp = 0.0
        running_dp = 0.0
        total_samples = 0

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            bsz = imgs.size(0)
            if bsz < 4:
                continue

            half = bsz // 2
            x1, y1 = imgs[:half], targets[:half]
            x2, y2 = imgs[half:], targets[half:]

            # 1. Clean forward pass on first half (for classification + classifier placeholders)
            _, clean_logits = model(x1)

            # 2. Manifold mixup on second half (for data placeholders)
            h2 = model.forward_layer2(x2)
            perm = torch.randperm(h2.size(0))
            h2_shuffled = h2[perm]
            y2_shuffled = y2[perm]

            # Enforce y_i != y_j where feasible
            diff_mask = (y2 != y2_shuffled)
            if diff_mask.sum() == 0:
                diff_mask = torch.ones_like(y2, dtype=torch.bool)

            lam = beta_dist.sample((h2.size(0), 1, 1, 1)).to(device)
            h_mixed = lam * h2 + (1.0 - lam) * h2_shuffled
            h_mixed = h_mixed[diff_mask]

            _, mixed_logits = model.forward_from_layer2(h_mixed)

            loss, l_cls, l_cp, l_dp = criterion(clean_logits, y1, mixed_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_tot += loss.item() * half
            running_cls += l_cls * half
            running_cp += l_cp * half
            running_dp += l_dp * half
            total_samples += half

        scheduler.step()
        val_acc, _ = evaluate_known(model, val_loader, device)

        history["epochs"].append(epoch)
        history["total_loss"].append(running_tot / total_samples)
        history["cls_loss"].append(running_cls / total_samples)
        history["cp_loss"].append(running_cp / total_samples)
        history["dp_loss"].append(running_dp / total_samples)
        history["val_acc"].append(val_acc)

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
        if epoch % 5 == 0 or epoch == 50 or improved:
            print(
                f"PROSER Epoch [{epoch:02d}/50] | Total: {running_tot/total_samples:.4f} "
                f"(Cls: {running_cls/total_samples:.4f}, CP: {running_cp/total_samples:.4f}, DP: {running_dp/total_samples:.4f}) | "
                f"Val Known Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%) {save_str}"
            )

    elapsed = time.time() - start_time
    print(f"\nPROSER training completed in {elapsed/60.0:.2f} minutes.")
    print(f"Best PROSER Validation Accuracy: {best_val_acc:.2f}%")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    best_checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_acc, _ = evaluate_known(model, test_loader, device)
    print(f"PROSER Known CIFAR-10 Test Closed-Set Accuracy (CSA): {test_acc:.2f}%\n")


if __name__ == "__main__":
    main()