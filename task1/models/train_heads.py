import sys
from pathlib import Path

# Add project root to sys.path so 'task1' resolves regardless of working directory
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
from sklearn.model_selection import train_test_split

from task1.models.backbones import (
    get_device,
    load_resnet50,
    load_vit_b16,
    load_clip_vit_b32,
)

SEED = 6304
DATA_DIR = REPO_ROOT / "data"
CHECKPOINT_DIR = REPO_ROOT / "task1" / "results" / "checkpoints"
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4


def set_seed(seed: int = SEED):
    """Locks random seeds for exact reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def extract_features(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Passes images through a frozen backbone to extract pooled embeddings."""
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            feats = model(images)
            features_list.append(feats.detach().cpu())
            labels_list.append(targets)

    return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)


def train_linear_head(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    embed_dim: int,
    device: torch.device,
) -> Tuple[nn.Linear, float]:
    """Trains a linear classifier on frozen embeddings and tracks validation accuracy."""
    head = nn.Linear(embed_dim, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_feats, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(EPOCHS):
        head.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = head(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validation evaluation
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = head(batch_x).argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        val_acc = (correct / total) * 100.0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    # Load best checkpoint weights
    head.load_state_dict(best_weights)
    return head, best_val_acc


def run_training_pipeline():
    set_seed(SEED)
    device = get_device()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Executing Linear Probe Training on device: {device}")

    model_factories = {
        "resnet50": load_resnet50,
        "vit_b16": load_vit_b16,
        "clip_vit_b32": load_clip_vit_b32,
    }

    summary_metrics = {}

    for name, loader_fn in model_factories.items():
        print(f"\n--- Processing Backbone: {name} ---")
        backbone, transform, embed_dim = loader_fn()
        backbone = backbone.to(device)

        # Load STL-10 train split with native backbone transforms
        full_train_set = STL10(
            root=str(DATA_DIR), split="train", transform=transform, download=False
        )
        targets = np.array(full_train_set.labels)

        # Stratified 80/20 train/validation split with seed 6304
        train_idx, val_idx = train_test_split(
            np.arange(len(targets)),
            test_size=0.20,
            random_state=SEED,
            stratify=targets,
        )

        train_sub = Subset(full_train_set, train_idx)
        val_sub = Subset(full_train_set, val_idx)

        # Set num_workers=0 to avoid Windows multiprocessing child process bugs
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print("Extracting training features...")
        train_feats, train_y = extract_features(backbone, train_loader, device)
        print("Extracting validation features...")
        val_feats, val_y = extract_features(backbone, val_loader, device)

        print("Training linear classification head...")
        head, best_acc = train_linear_head(
            train_feats, train_y, val_feats, val_y, embed_dim, device
        )

        checkpoint_path = CHECKPOINT_DIR / f"{name}_linear_head.pt"
        torch.save(head.state_dict(), checkpoint_path)
        print(f"Saved {name} head to {checkpoint_path} (Best Val Acc: {best_acc:.2f}%)")

        summary_metrics[name] = {
            "val_accuracy": round(best_acc, 2),
            "checkpoint": str(checkpoint_path),
        }

    with open(CHECKPOINT_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print("\nAll linear probe classifiers trained and saved successfully.")


if __name__ == "__main__":
    run_training_pipeline()