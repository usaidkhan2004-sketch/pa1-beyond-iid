"""
task4/evaluate_all.py
Master evaluation script for Task 4 Open-Set Recognition:
- Extracts features/logits across CIFAR-10 and CIFAR-100 Near/Far Unknowns
- Computes MSP, MLS, Energy, Mahalanobis (diagonal + 1e-6), and PROSER Dummy score
- Calibrates rejection threshold tau at the 95th percentile of validation unknownness
- Computes AUROC and FPR@95TPR
- Caches score arrays for plotting and extracts false acceptance failure cases
"""

import json
import resource
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from task4.data.dataset import (
    TransformedSubset,
    get_cifar10_datasets,
    get_cifar100_unknowns,
    get_transforms,
)
from task4.models.proser_model import PROSERNet
from task4.models.resnet_cifar import ResNet18CIFAR
from task4.scores.score_functions import (
    MahalanobisScorer,
    compute_energy,
    compute_mls,
    compute_msp,
    compute_proser_dummy_score,
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def raise_fd_limit():
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
    except Exception:
        pass


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def extract_features_and_logits(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_feats, all_logits, all_targets = [], [], []

    for imgs, targets in loader:
        imgs = imgs.to(device)
        feats, logits = model(imgs)
        all_feats.append(feats.cpu())
        all_logits.append(logits.cpu())
        all_targets.append(targets)

    feats = torch.cat(all_feats, dim=0)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return feats, logits, targets


def evaluate_detection(known_scores: np.ndarray, unknown_scores: np.ndarray, val_threshold: float):
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    y_scores = np.concatenate([known_scores, unknown_scores])

    auroc = roc_auc_score(y_true, y_scores) * 100.0
    fpr95 = np.mean(unknown_scores <= val_threshold) * 100.0
    rejection_rate = 100.0 - fpr95
    return auroc, fpr95, rejection_rate


def main():
    raise_fd_limit()
    device = get_device()
    print(f"=== Running Task 4 Comprehensive Evaluation on {device} ===")

    results_dir = Path("task4/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Datasets & Loaders (num_workers=0 to prevent Errno 24)
    train_raw, test_raw, train_idx, val_idx = get_cifar10_datasets(seed=6304)
    _, eval_transform = get_transforms("eval")

    cifar10_train_ds = TransformedSubset(train_raw, train_idx, transform=eval_transform)
    cifar10_val_ds = TransformedSubset(train_raw, val_idx, transform=eval_transform)
    cifar10_test_ds = TransformedSubset(test_raw, np.arange(len(test_raw)), transform=eval_transform)

    cifar100_near_ds, cifar100_far_ds, cifar100_all_ds, near_map, far_map = get_cifar100_unknowns()

    train_loader = DataLoader(cifar10_train_ds, batch_size=128, shuffle=False, num_workers=0)
    val_loader = DataLoader(cifar10_val_ds, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(cifar10_test_ds, batch_size=128, shuffle=False, num_workers=0)

    near_loader = DataLoader(cifar100_near_ds, batch_size=128, shuffle=False, num_workers=0)
    far_loader = DataLoader(cifar100_far_ds, batch_size=128, shuffle=False, num_workers=0)
    all_u_loader = DataLoader(cifar100_all_ds, batch_size=128, shuffle=False, num_workers=0)

    # =========================================================================
    # Part 1: Vanilla Model & 4 Post-Hoc Scores
    # =========================================================================
    print("\n--- Part 1: Evaluating Frozen Vanilla Baseline (MSP, MLS, Energy, Mahalanobis) ---")
    vanilla_model = ResNet18CIFAR(num_classes=10).to(device)
    vanilla_ckpt = torch.load("task4/checkpoints/vanilla_best.pth", map_location=device)
    vanilla_model.load_state_dict(vanilla_ckpt["model_state_dict"])

    v_tr_feats, _, v_tr_targets = extract_features_and_logits(vanilla_model, train_loader, device)
    v_val_feats, v_val_logits, _ = extract_features_and_logits(vanilla_model, val_loader, device)
    v_te_feats, v_te_logits, v_te_targets = extract_features_and_logits(vanilla_model, test_loader, device)
    v_near_feats, v_near_logits, _ = extract_features_and_logits(vanilla_model, near_loader, device)
    v_far_feats, v_far_logits, _ = extract_features_and_logits(vanilla_model, far_loader, device)
    v_all_feats, v_all_logits, _ = extract_features_and_logits(vanilla_model, all_u_loader, device)

    csa_preds = torch.argmax(v_te_logits, dim=1)
    vanilla_csa = (csa_preds == v_te_targets).float().mean().item() * 100.0
    print(f"Vanilla CIFAR-10 Test CSA: {vanilla_csa:.2f}%")

    mah_scorer = MahalanobisScorer(v_tr_feats, v_tr_targets, num_classes=10)

    scores_dict = {
        "MSP": {
            "val": compute_msp(v_val_logits),
            "test": compute_msp(v_te_logits),
            "near": compute_msp(v_near_logits),
            "far": compute_msp(v_far_logits),
            "all": compute_msp(v_all_logits),
        },
        "MLS": {
            "val": compute_mls(v_val_logits),
            "test": compute_mls(v_te_logits),
            "near": compute_mls(v_near_logits),
            "far": compute_mls(v_far_logits),
            "all": compute_mls(v_all_logits),
        },
        "Energy": {
            "val": compute_energy(v_val_logits),
            "test": compute_energy(v_te_logits),
            "near": compute_energy(v_near_logits),
            "far": compute_energy(v_far_logits),
            "all": compute_energy(v_all_logits),
        },
        "Mahalanobis": {
            "val": mah_scorer.score(v_val_feats),
            "test": mah_scorer.score(v_te_feats),
            "near": mah_scorer.score(v_near_feats),
            "far": mah_scorer.score(v_far_feats),
            "all": mah_scorer.score(v_all_feats),
        },
    }

    table1_results = {}
    for score_name, d in scores_dict.items():
        tau = np.percentile(d["val"], 95)
        test_acc_rate = np.mean(d["test"] <= tau) * 100.0

        near_auc, near_fpr, near_rej = evaluate_detection(d["test"], d["near"], tau)
        far_auc, far_fpr, far_rej = evaluate_detection(d["test"], d["far"], tau)
        all_auc, all_fpr, all_rej = evaluate_detection(d["test"], d["all"], tau)

        table1_results[score_name] = {
            "threshold_tau": float(tau),
            "cifar10_test_accept_rate": float(test_acc_rate),
            "near_auroc": float(near_auc),
            "near_fpr95": float(near_fpr),
            "far_auroc": float(far_auc),
            "far_fpr95": float(far_fpr),
            "all_auroc": float(all_auc),
            "all_fpr95": float(all_fpr),
        }
        print(f"[{score_name:11s}] AUROC -> Near: {near_auc:.2f}% | Far: {far_auc:.2f}% | All: {all_auc:.2f}% | FPR@95 (All): {all_fpr:.2f}%")

    with open(results_dir / "osr_table1_scores.json", "w") as f:
        json.dump(table1_results, f, indent=4)

    # =========================================================================
    # Part 2: Model Comparison (Vanilla, GCSC, PROSER)
    # =========================================================================
    print("\n--- Part 2: Model Comparison (Vanilla, GCSC, PROSER) with MLS ---")
    table2_results = {
        "Vanilla_MLS": {
            "csa": vanilla_csa,
            **table1_results["MLS"]
        }
    }

    cache_payload = {
        "scores_dict": scores_dict,
        "models_mls": {
            "Vanilla": {
                "test": scores_dict["MLS"]["test"],
                "near": scores_dict["MLS"]["near"],
                "far": scores_dict["MLS"]["far"],
            }
        }
    }

    # Evaluate GCSC
    gcsc_ckpt_path = Path("task4/checkpoints/gcsc_best.pth")
    if gcsc_ckpt_path.is_file():
        gcsc_model = ResNet18CIFAR(num_classes=10).to(device)
        gcsc_ckpt = torch.load(gcsc_ckpt_path, map_location=device)
        gcsc_model.load_state_dict(gcsc_ckpt["model_state_dict"])

        _, g_val_logits, _ = extract_features_and_logits(gcsc_model, val_loader, device)
        _, g_te_logits, g_te_targets = extract_features_and_logits(gcsc_model, test_loader, device)
        _, g_near_logits, _ = extract_features_and_logits(gcsc_model, near_loader, device)
        _, g_far_logits, _ = extract_features_and_logits(gcsc_model, far_loader, device)
        _, g_all_logits, _ = extract_features_and_logits(gcsc_model, all_u_loader, device)

        gcsc_csa = (torch.argmax(g_te_logits, dim=1) == g_te_targets).float().mean().item() * 100.0
        g_val_mls = compute_mls(g_val_logits)
        g_tau = np.percentile(g_val_mls, 95)
        g_te_mls = compute_mls(g_te_logits)
        g_near_mls = compute_mls(g_near_logits)
        g_far_mls = compute_mls(g_far_logits)
        g_all_mls = compute_mls(g_all_logits)

        cache_payload["models_mls"]["GCSC"] = {
            "test": g_te_mls, "near": g_near_mls, "far": g_far_mls
        }

        near_auc, near_fpr, _ = evaluate_detection(g_te_mls, g_near_mls, g_tau)
        far_auc, far_fpr, _ = evaluate_detection(g_te_mls, g_far_mls, g_tau)
        all_auc, all_fpr, _ = evaluate_detection(g_te_mls, g_all_mls, g_tau)

        table2_results["GCSC_MLS"] = {
            "csa": float(gcsc_csa),
            "threshold_tau": float(g_tau),
            "cifar10_test_accept_rate": float(np.mean(g_te_mls <= g_tau) * 100.0),
            "near_auroc": float(near_auc),
            "near_fpr95": float(near_fpr),
            "far_auroc": float(far_auc),
            "far_fpr95": float(far_fpr),
            "all_auroc": float(all_auc),
            "all_fpr95": float(all_fpr),
        }
        print(f"[GCSC MLS   ] CSA: {gcsc_csa:.2f}% | Near AUROC: {near_auc:.2f}% | Far AUROC: {far_auc:.2f}% | All AUROC: {all_auc:.2f}%")

    # Evaluate PROSER
    proser_ckpt_path = Path("task4/checkpoints/proser_best.pth")
    if proser_ckpt_path.is_file():
        p_base = ResNet18CIFAR(num_classes=10)
        proser_model = PROSERNet(p_base, num_known=10, num_dummy=5).to(device)
        proser_ckpt = torch.load(proser_ckpt_path, map_location=device)
        proser_model.load_state_dict(proser_ckpt["model_state_dict"])

        _, p_val_logits, _ = extract_features_and_logits(proser_model, val_loader, device)
        _, p_te_logits, p_te_targets = extract_features_and_logits(proser_model, test_loader, device)
        _, p_near_logits, _ = extract_features_and_logits(proser_model, near_loader, device)
        _, p_far_logits, _ = extract_features_and_logits(proser_model, far_loader, device)
        _, p_all_logits, _ = extract_features_and_logits(proser_model, all_u_loader, device)

        p_te_known = p_te_logits[:, :10]
        proser_csa = (torch.argmax(p_te_known, dim=1) == p_te_targets).float().mean().item() * 100.0

        p_val_mls = compute_mls(p_val_logits[:, :10])
        p_tau_mls = np.percentile(p_val_mls, 95)
        p_te_mls = compute_mls(p_te_known)
        p_near_mls = compute_mls(p_near_logits[:, :10])
        p_far_mls = compute_mls(p_far_logits[:, :10])
        p_all_mls = compute_mls(p_all_logits[:, :10])

        cache_payload["models_mls"]["PROSER"] = {
            "test": p_te_mls, "near": p_near_mls, "far": p_far_mls
        }

        near_auc, near_fpr, _ = evaluate_detection(p_te_mls, p_near_mls, p_tau_mls)
        far_auc, far_fpr, _ = evaluate_detection(p_te_mls, p_far_mls, p_tau_mls)
        all_auc, all_fpr, _ = evaluate_detection(p_te_mls, p_all_mls, p_tau_mls)

        table2_results["PROSER_MLS"] = {
            "csa": float(proser_csa),
            "threshold_tau": float(p_tau_mls),
            "cifar10_test_accept_rate": float(np.mean(p_te_mls <= p_tau_mls) * 100.0),
            "near_auroc": float(near_auc),
            "near_fpr95": float(near_fpr),
            "far_auroc": float(far_auc),
            "far_fpr95": float(far_fpr),
            "all_auroc": float(all_auc),
            "all_fpr95": float(all_fpr),
        }
        print(f"[PROSER MLS ] CSA: {proser_csa:.2f}% | Near AUROC: {near_auc:.2f}% | Far AUROC: {far_auc:.2f}% | All AUROC: {all_auc:.2f}%")

        p_val_dummy = compute_proser_dummy_score(p_val_logits)
        p_tau_dummy = np.percentile(p_val_dummy, 95)
        p_te_dummy = compute_proser_dummy_score(p_te_logits)
        p_near_dummy = compute_proser_dummy_score(p_near_logits)
        p_far_dummy = compute_proser_dummy_score(p_far_logits)
        p_all_dummy = compute_proser_dummy_score(p_all_logits)

        near_auc_d, near_fpr_d, _ = evaluate_detection(p_te_dummy, p_near_dummy, p_tau_dummy)
        far_auc_d, far_fpr_d, _ = evaluate_detection(p_te_dummy, p_far_dummy, p_tau_dummy)
        all_auc_d, all_fpr_d, _ = evaluate_detection(p_te_dummy, p_all_dummy, p_tau_dummy)

        table2_results["PROSER_Placeholder"] = {
            "csa": float(proser_csa),
            "threshold_tau": float(p_tau_dummy),
            "cifar10_test_accept_rate": float(np.mean(p_te_dummy <= p_tau_dummy) * 100.0),
            "near_auroc": float(near_auc_d),
            "near_fpr95": float(near_fpr_d),
            "far_auroc": float(far_auc_d),
            "far_fpr95": float(far_fpr_d),
            "all_auroc": float(all_auc_d),
            "all_fpr95": float(all_fpr_d),
        }
        print(f"[PROSER DUMMY] CSA: {proser_csa:.2f}% | Near AUROC: {near_auc_d:.2f}% | Far AUROC: {far_auc_d:.2f}% | All AUROC: {all_auc_d:.2f}%")

    with open(results_dir / "osr_table2_models.json", "w") as f:
        json.dump(table2_results, f, indent=4)

    # Save evaluation cache for instant plotting
    torch.save(cache_payload, results_dir / "osr_cache.pt")

    # =========================================================================
    # Part 3: Failure Cases (Under Vanilla MLS Threshold)
    # =========================================================================
    print("\n--- Part 3: Extracting Failure Cases (Vanilla MLS) ---")
    mls_tau = table1_results["MLS"]["threshold_tau"]
    inv_near_map = {v: k for k, v in near_map.items()}
    inv_far_map = {v: k for k, v in far_map.items()}

    def find_failures(loader, inv_map, group_name):
        failures = []
        vanilla_model.eval()

        for imgs, targets in loader:
            imgs = imgs.to(device)
            _, logits = vanilla_model(imgs)
            scores = compute_mls(logits)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for i in range(len(scores)):
                if scores[i] <= mls_tau:
                    unknown_class = inv_map.get(targets[i].item(), f"ID_{targets[i].item()}")
                    pred_class = CIFAR10_CLASSES[preds[i]]
                    failures.append({
                        "group": group_name,
                        "unknown_class": unknown_class,
                        "predicted_known_class": pred_class,
                        "score_mls": float(scores[i]),
                        "threshold_tau": float(mls_tau),
                        "max_logit": float(-scores[i]),
                    })
        return failures

    near_failures = sorted(find_failures(near_loader, inv_near_map, "Near Unknown"), key=lambda x: x["score_mls"])[:5]
    far_failures = sorted(find_failures(far_loader, inv_far_map, "Far Unknown"), key=lambda x: x["score_mls"])[:5]

    with open(results_dir / "osr_failures.json", "w") as f:
        json.dump({
            "vanilla_mls_threshold": float(mls_tau),
            "near_unknown_failures": near_failures,
            "far_unknown_failures": far_failures,
        }, f, indent=4)

    print("Master evaluation completed successfully.")


if __name__ == "__main__":
    main()