"""
task4/scores/score_functions.py
Novelty / Unknownness Scoring Functions:
- Higher score indicates greater novelty / higher likelihood of being unknown.
- Includes MSP, MLS, Energy, Mahalanobis (diagonal), and PROSER Placeholder.
"""

from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F


def compute_msp(logits: torch.Tensor) -> np.ndarray:
    """u_MSP(x) = 1 - max_k p_k(x)"""
    probs = F.softmax(logits, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    return (1.0 - max_probs).detach().cpu().numpy()


def compute_mls(logits: torch.Tensor) -> np.ndarray:
    """u_MLS(x) = -max_k z_k(x)"""
    max_logits, _ = torch.max(logits, dim=1)
    return (-max_logits).detach().cpu().numpy()


def compute_energy(logits: torch.Tensor) -> np.ndarray:
    """u_Energy(x) = -log sum_k exp(z_k(x))"""
    energy = -torch.logsumexp(logits, dim=1)
    return energy.detach().cpu().numpy()


class MahalanobisScorer:
    """
    Computes class-conditional Mahalanobis distance using unaugmented training features.
    Uses shared diagonal covariance + 1e-6 as required by Page 13.
    """
    def __init__(self, train_feats: torch.Tensor, train_targets: torch.Tensor, num_classes: int = 10):
        self.num_classes = num_classes
        feat_dim = train_feats.size(1)

        # Estimate class centroids mu_c
        self.means = torch.zeros(num_classes, feat_dim)
        for c in range(num_classes):
            c_mask = (train_targets == c)
            self.means[c] = train_feats[c_mask].mean(dim=0)

        # Estimate shared diagonal covariance
        diffs = []
        for c in range(num_classes):
            c_mask = (train_targets == c)
            diff = train_feats[c_mask] - self.means[c].unsqueeze(0)
            diffs.append(diff)
        diffs = torch.cat(diffs, dim=0)

        # Var(d) + 1e-6
        diag_var = torch.mean(diffs ** 2, dim=0) + 1e-6
        self.inv_diag = (1.0 / diag_var).unsqueeze(0)  # [1, 512]

    def score(self, feats: torch.Tensor) -> np.ndarray:
        """
        u_Mah(x) = min_c (f(x) - mu_c)^T Sigma^{-1} (f(x) - mu_c)
        """
        batch_size = feats.size(0)
        min_dists = torch.full((batch_size,), float("inf"))

        for c in range(self.num_classes):
            diff = feats - self.means[c].unsqueeze(0)  # [B, 512]
            weighted_sq_diff = (diff ** 2) * self.inv_diag  # [B, 512]
            dist_c = torch.sum(weighted_sq_diff, dim=1)  # [B]
            min_dists = torch.minimum(min_dists, dist_c)

        return min_dists.detach().cpu().numpy()


def compute_proser_dummy_score(logits_15: torch.Tensor, num_known: int = 10) -> np.ndarray:
    """
    Placeholder-based unknownness score:
    u_dummy(x) = sum(p_dummy) / sum(p_all)
    """
    probs = F.softmax(logits_15, dim=1)
    dummy_probs = torch.sum(probs[:, num_known:], dim=1)
    return dummy_probs.detach().cpu().numpy()