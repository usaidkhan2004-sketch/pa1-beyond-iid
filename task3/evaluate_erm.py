"""
task3/evaluate_erm.py
Loads the saved Task 2 Source-Only baseline checkpoint and evaluates:
1. Per-domain Source Val Acc & Macro-F1 (Photo, Art Painting, Cartoon)
2. Mean-source and Worst-source Acc & Macro-F1
3. 3-way Source Domain Separability (Chance: 33.33%)
4. Standardized Local Sharpness Proxy (rho=0.05, 96 fixed samples)
Saves results to task3/checkpoints/erm_metrics.json.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from shared.pacs_protocol import get_pacs_dataloaders
from task2.models.backbone import SourceOnlyClassifier
from task3.diagnostics.sharpness_proxy import (
    compute_sharpness_proxy,
    get_fixed_sharpness_batch,
)
from task3.diagnostics.source_separability import (
    compute_source_domain_separability,
    extract_source_features,
)

CHECKPOINTS_DIR = Path("task3/checkpoints")
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
SRC_CHECKPOINT = Path("task2/checkpoints/source_only_best.pth")


@torch.no_grad()
def evaluate_source_split(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        images, targets = batch[0], batch[1]
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        if isinstance(targets, torch.Tensor):
            all_targets.extend(targets.cpu().numpy())
        else:
            all_targets.extend(targets)

    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro") * 100.0
    return acc, macro_f1


def main():
    if not SRC_CHECKPOINT.exists():
        print(f"Error: Could not find Task 2 checkpoint at {SRC_CHECKPOINT}")
        return

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"--- Task 3: Evaluating ERM Baseline ---")
    print(f"Device: {device} | Loading: {SRC_CHECKPOINT}")

    # 1. Load data following the shared PACS protocol
    data_root = "data/PACS"
    _, val_loaders_raw, _ = get_pacs_dataloaders(data_root)
    val_loaders = {
        dom: DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=0)
        for dom, loader in val_loaders_raw.items()
    }

    # 2. Load Model using SourceOnlyClassifier
    model = SourceOnlyClassifier(pretrained=False, num_classes=7).to(device)
    state = torch.load(SRC_CHECKPOINT, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

    # 3. Source Validation Evaluation
    domains = ["photo", "art_painting", "cartoon"]
    domain_metrics = {}
    for domain in domains:
        acc, f1 = evaluate_source_split(model, val_loaders[domain], device)
        domain_metrics[domain] = {"acc": acc, "macro_f1": f1}
        print(f"  Source [{domain:<12}]: Acc = {acc:>6.2f}% | Macro-F1 = {f1:>6.2f}%")

    mean_src_acc = float(np.mean([domain_metrics[d]["acc"] for d in domains]))
    mean_src_f1 = float(np.mean([domain_metrics[d]["macro_f1"] for d in domains]))

    # Worst-source evaluation (lowest-performing source domain)
    worst_domain = min(domains, key=lambda d: domain_metrics[d]["macro_f1"])
    worst_src_acc = domain_metrics[worst_domain]["acc"]
    worst_src_f1 = domain_metrics[worst_domain]["macro_f1"]

    print("-" * 55)
    print(f"Mean Source Val Acc:   {mean_src_acc:.2f}% | Mean Source Val F1:  {mean_src_f1:.2f}%")
    print(f"Worst Source Domain:   {worst_domain} (Acc: {worst_src_acc:.2f}%, F1: {worst_src_f1:.2f}%)")

    # 4. Source-Domain Separability Diagnostic
    print("\nRunning 3-Way Source-Domain Separability Diagnostic...")
    X_src, y_src = extract_source_features(model, val_loaders, device)
    src_sep = compute_source_domain_separability(X_src, y_src, seed=6304)
    print(f"Source-Domain Separability: {src_sep:.2f}% (Chance: 33.33%)")

    # 5. Standardized Local Sharpness Proxy Diagnostic
    print("Running Standardized Local Sharpness Proxy (rho=0.05)...")
    fixed_x, fixed_y = get_fixed_sharpness_batch(val_loaders, device, seed=6304)
    delta_sharp = compute_sharpness_proxy(model, fixed_x, fixed_y, rho=0.05)
    print(f"Delta_sharp (Sharpness Proxy): {delta_sharp:.4f}")

    # Save to JSON
    output_data = {
        "method": "ERM",
        "source_domains": domain_metrics,
        "mean_source_acc": mean_src_acc,
        "mean_source_f1": mean_src_f1,
        "worst_source_domain": worst_domain,
        "worst_source_acc": worst_src_acc,
        "worst_source_f1": worst_src_f1,
        "source_domain_separability": src_sep,
        "sharpness_proxy": delta_sharp,
    }

    out_file = CHECKPOINTS_DIR / "erm_metrics.json"
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"\nSaved ERM metrics to: {out_file}")


if __name__ == "__main__":
    main()