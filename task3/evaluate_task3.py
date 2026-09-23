"""task3/evaluate_task3.py

Comprehensive evaluation and diagnostic suite for Task 3 Domain
Generalization.
Evaluates:
  1. ERM (Source-only baseline from Task 2)
  2. DAN-DG (lambda_dg = 1.0)
  3. SAM (rho = 0.05)
Metrics Computed:
  - Source Validation: Per-domain, Mean-Source, Worst-Source Acc & Macro-F1
  - Source-Domain Separability: 3-way multinomial logistic regression (C=1,
  70/30 split)
  - Common Sharpness Proxy: Delta_sharp with rho = 0.05 on fixed 32-sample/domain
  batch
  - Target Generalization (Sketch): Accuracy, Macro-F1, Delta vs. ERM, Per-class
  breakdown
Outputs:
  - task3/results/task3_comparison_table.json
  - task3/results/task3_per_class_sketch.json
"""

import json
import os
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from shared.pacs import PACSDataset, get_pacs_transform
from shared.pacs_protocol import (
    DEFAULT_SEED,
    SOURCE_DOMAINS,
    TARGET_DOMAIN,
    build_or_load_splits,
)
from task2.models.backbone import (
    ResNet18Backbone,
    SourceOnlyClassifier,
    TaskClassifier,
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


class DANModel(nn.Module):
  """Backbone + Classifier Head for DAN-DG checkpoint loading."""

  def __init__(self, pretrained: bool = False, num_classes: int = 7):
    super().__init__()
    self.backbone = ResNet18Backbone(pretrained=pretrained)
    self.classifier = TaskClassifier(
        in_features=self.backbone.feature_dim, num_classes=num_classes
    )

  def forward(self, x: torch.Tensor):
    feats = self.backbone(x)
    logits = self.classifier(feats)
    return feats, logits


def load_model(
    model_type: str, ckpt_path: str, device: torch.device
) -> nn.Module:
  """Loads weights into the corresponding model architecture."""
  if not os.path.isfile(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

  checkpoint = torch.load(ckpt_path, map_location=device)
  state_dict = checkpoint.get("model_state_dict", checkpoint)

  if model_type in ["erm", "sam"]:
    model = SourceOnlyClassifier(pretrained=False, num_classes=7)
    model.load_state_dict(state_dict)
  elif model_type == "dan_dg":
    model = DANModel(pretrained=False, num_classes=7)
    model.load_state_dict(state_dict)
  else:
    raise ValueError(f"Unknown model_type: {model_type}")

  model.to(device)
  model.eval()
  return model


def get_eval_loaders(
    dataset_root: str = "data/PACS",
    split_file_path: str = "shared/splits/pacs_sketch_seed6304.json",
    seed: int = DEFAULT_SEED,
):
  """Builds validation loaders for sources and target loader for Sketch."""
  splits = build_or_load_splits(dataset_root, split_file_path, seed)
  eval_transform = get_pacs_transform("eval")

  val_loaders = {}
  for d in SOURCE_DOMAINS:
    ds = PACSDataset(splits["sources"][d]["val"], transform=eval_transform)
    val_loaders[d] = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

  target_ds = PACSDataset(
      splits["target"][TARGET_DOMAIN]["all"], transform=eval_transform
  )
  target_loader = DataLoader(
      target_ds, batch_size=32, shuffle=False, num_workers=2
  )

  return val_loaders, target_loader, splits


@torch.no_grad()
def evaluate_dataloader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
  """Computes Top-1 Accuracy, Macro-F1, all predictions, and all targets."""
  model.eval()
  all_preds, all_targets = [], []

  for batch in loader:
    imgs, targets = batch[0].to(device), batch[1]
    out = model(imgs)
    logits = out[1] if isinstance(out, tuple) else out
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
  return acc, f1, all_preds, all_targets


@torch.no_grad()
def extract_source_features(
    model: nn.Module, val_loaders: Dict[str, DataLoader], device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
  """Extracts 512-dim features and domain labels (0=Photo, 1=Art, 2=Cartoon)."""
  model.eval()
  all_features, all_domain_labels = [], []

  for dom_idx, dom_name in enumerate(SOURCE_DOMAINS):
    loader = val_loaders[dom_name]
    for batch in loader:
      imgs = batch[0].to(device)
      if hasattr(model, "backbone"):
        feats = model.backbone(imgs)
      else:
        feats = model.feature_extractor(imgs)

      all_features.append(feats.cpu().numpy())
      all_domain_labels.append(np.full(imgs.size(0), dom_idx))

  return np.concatenate(all_features, axis=0), np.concatenate(
      all_domain_labels, axis=0
  )


def compute_source_domain_separability(
    features: np.ndarray, domain_labels: np.ndarray, seed: int = DEFAULT_SEED
) -> float:
  """Trains a 3-way multinomial logistic regression (C=1, 70/30 split) on source validation features."""
  x_train, x_test, y_train, y_test = train_test_split(
      features,
      domain_labels,
      test_size=0.30,
      random_state=seed,
      stratify=domain_labels,
  )
  clf = LogisticRegression(
      C=1.0,
      max_iter=1000,
      random_state=seed,
      solver="lbfgs",
  )
  clf.fit(x_train, y_train)
  y_pred = clf.predict(x_test)
  return float(accuracy_score(y_test, y_pred) * 100.0)


def compute_sharpness_proxy(
    model: nn.Module,
    splits: Dict,
    dataset_root: str,
    device: torch.device,
    rho: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> float:
  """Fixed validation batch of 32 examples per source domain (96 total).

  Delta_sharp = L_val(theta + epsilon) - L_val(theta)
  """
  model.eval()
  eval_transform = get_pacs_transform("eval")
  rng = random.Random(seed)

  fixed_batches = []
  fixed_targets = []
  for d in SOURCE_DOMAINS:
    val_samples = splits["sources"][d]["val"]
    chosen = rng.sample(val_samples, 32)
    ds = PACSDataset(chosen, transform=eval_transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    for batch in loader:
      fixed_batches.append(batch[0])
      fixed_targets.append(batch[1])

  x_fix = torch.cat(fixed_batches, dim=0).to(device)
  y_fix = torch.cat(fixed_targets, dim=0).to(device)

  criterion = nn.CrossEntropyLoss()

  # Step 1: Base loss L_val(theta)
  out = model(x_fix)
  logits = out[1] if isinstance(out, tuple) else out
  loss_base = criterion(logits, y_fix)

  # Step 2: Gradient ascent direction
  model.zero_grad()
  loss_base.backward()

  trainable_params = [p for p in model.parameters() if p.requires_grad]
  grad_norms = [
      p.grad.detach().norm(2) for p in trainable_params if p.grad is not None
  ]
  total_grad_norm = torch.norm(torch.stack(grad_norms), 2)

  perturbations = {}
  scale = rho / (total_grad_norm + 1e-12)
  with torch.no_grad():
    for p in trainable_params:
      if p.grad is not None:
        e_w = p.grad * scale
        p.add_(e_w)
        perturbations[p] = e_w

  # Step 3: Perturbed loss L_val(theta + epsilon)
  with torch.no_grad():
    out_pert = model(x_fix)
    logits_pert = out_pert[1] if isinstance(out_pert, tuple) else out_pert
    loss_perturbed = criterion(logits_pert, y_fix)

  # Step 4: Revert weights
  with torch.no_grad():
    for p in trainable_params:
      if p in perturbations:
        p.sub_(perturbations[p])

  delta_sharp = (loss_perturbed - loss_base).item()
  return float(delta_sharp)


def main():
  set_seed(DEFAULT_SEED)
  device = get_device()
  print(f"=== Running Task 3 Common Evaluation on {device} ===")

  out_dir = Path("task3/results")
  out_dir.mkdir(parents=True, exist_ok=True)

  val_loaders, target_loader, splits = get_eval_loaders()

  models_config = {
      "ERM": ("erm", "task2/checkpoints/source_only_best.pth"),
      "DAN-DG": ("dan_dg", "task3/checkpoints/dan_dg_best.pth"),
      "SAM": ("sam", "task3/checkpoints/sam_best.pth"),
  }

  results = {}
  per_class_results = {}
  erm_sketch_acc = 0.0

  for name, (m_type, ckpt_path) in models_config.items():
    print(f"\n--- Evaluating {name} ({ckpt_path}) ---")
    model = load_model(m_type, ckpt_path, device)

    # 1. Source Validation Domains
    src_domain_accs = {}
    src_domain_f1s = {}
    for d in SOURCE_DOMAINS:
      acc, f1, _, _ = evaluate_dataloader(model, val_loaders[d], device)
      src_domain_accs[d] = acc
      src_domain_f1s[d] = f1

    mean_src_acc = float(np.mean(list(src_domain_accs.values())))
    mean_src_f1 = float(np.mean(list(src_domain_f1s.values())))
    worst_src_acc = float(np.min(list(src_domain_accs.values())))
    worst_src_f1 = float(np.min(list(src_domain_f1s.values())))

    # 2. Source-Domain Separability Probe
    print("  Extracting source features for domain separability probe...")
    features, dom_labels = extract_source_features(model, val_loaders, device)
    sep_score = compute_source_domain_separability(
        features, dom_labels, seed=DEFAULT_SEED
    )

    # 3. Local Sharpness Proxy
    print("  Measuring local sharpness proxy (rho=0.05)...")
    sharpness_val = compute_sharpness_proxy(
        model, splits, dataset_root="data/PACS", device=device, rho=0.05
    )

    # 4. Target Generalization (Sketch)
    print("  Evaluating on held-out Sketch target domain...")
    sketch_acc, sketch_f1, sketch_preds, sketch_targets = evaluate_dataloader(
        model, target_loader, device
    )

    if name == "ERM":
      erm_sketch_acc = sketch_acc
      delta_sketch = 0.0
    else:
      delta_sketch = sketch_acc - erm_sketch_acc

    # Per-class breakdown on Sketch (7 PACS classes)
    p_class_accs = {}
    for c in range(7):
      idx = sketch_targets == c
      if np.sum(idx) > 0:
        p_class_accs[f"class_{c}"] = float(
            accuracy_score(sketch_targets[idx], sketch_preds[idx]) * 100.0
        )
      else:
        p_class_accs[f"class_{c}"] = 0.0

    per_class_results[name] = p_class_accs

    results[name] = {
        "source_val_photo_acc": src_domain_accs["photo"],
        "source_val_art_acc": src_domain_accs["art_painting"],
        "source_val_cartoon_acc": src_domain_accs["cartoon"],
        "mean_source_acc": mean_src_acc,
        "mean_source_f1": mean_src_f1,
        "worst_source_acc": worst_src_acc,
        "worst_source_f1": worst_src_f1,
        "source_separability": sep_score,
        "sharpness_proxy": sharpness_val,
        "sketch_acc": sketch_acc,
        "sketch_f1": sketch_f1,
        "delta_sketch_vs_erm": delta_sketch,
    }

    print(
        f"  Mean Src Acc: {mean_src_acc:.2f}% | Worst Src Acc:"
        f" {worst_src_acc:.2f}%"
    )
    print(
        f"  Domain Separability: {sep_score:.2f}% (Chance=33.3%) | Sharpness:"
        f" {sharpness_val:.4f}"
    )
    print(
        f"  Sketch Acc: {sketch_acc:.2f}% | Sketch F1: {sketch_f1:.2f}% | Delta:"
        f" {delta_sketch:+.2f}%"
    )

  # Save summary deliverables
  table_file = out_dir / "task3_comparison_table.json"
  with open(table_file, "w") as f:
    json.dump(results, f, indent=4)

  per_class_file = out_dir / "task3_per_class_sketch.json"
  with open(per_class_file, "w") as f:
    json.dump(per_class_results, f, indent=4)

  print(f"\nSaved summary comparison table to: {table_file}")
  print(f"Saved per-class Sketch analysis to: {per_class_file}")


if __name__ == "__main__":
  main()