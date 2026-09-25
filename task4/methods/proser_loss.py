"""
task4/methods/proser_loss.py
Implements PROSER Loss Functions (Zhou et al., 2021):
- Standard cross-entropy on known classes
- Classifier placeholder loss (beta=1.0)
- Manifold mixup data placeholder loss (gamma=0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PROSERLoss(nn.Module):
    def __init__(self, num_known: int = 10, num_dummy: int = 5, beta: float = 1.0, gamma: float = 0.1):
        super().__init__()
        self.num_known = num_known
        self.num_dummy = num_dummy
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        clean_logits: torch.Tensor,
        clean_targets: torch.Tensor,
        mixed_logits: torch.Tensor,
    ):
        """
        clean_logits: [B1, 15] logits for standard samples
        clean_targets: [B1] ground truth known class IDs (0..9)
        mixed_logits: [B2, 15] logits for manifold-mixup samples
        """
        # 1. Standard classification loss on known classes
        known_logits = clean_logits[:, :self.num_known]
        loss_cls = F.cross_entropy(known_logits, clean_targets)

        # 2. Classifier Placeholder Loss:
        # Mask out true class y. The dummy classes should dominate remaining known classes.
        batch_size = clean_logits.size(0)
        mask = torch.ones_like(clean_logits, dtype=torch.bool)
        mask[torch.arange(batch_size, device=clean_logits.device), clean_targets] = False

        masked_logits = torch.where(mask, clean_logits, torch.tensor(-1e9, device=clean_logits.device))
        dummy_logsumexp = torch.logsumexp(masked_logits[:, self.num_known:], dim=1)
        total_remaining_logsumexp = torch.logsumexp(masked_logits, dim=1)
        loss_cp = torch.mean(total_remaining_logsumexp - dummy_logsumexp)

        # 3. Data Placeholder Loss on mixed representations:
        # Mixed representations must be driven into the dummy classes
        mixed_dummy_logsumexp = torch.logsumexp(mixed_logits[:, self.num_known:], dim=1)
        mixed_total_logsumexp = torch.logsumexp(mixed_logits, dim=1)
        loss_dp = torch.mean(mixed_total_logsumexp - mixed_dummy_logsumexp)

        total_loss = loss_cls + self.beta * loss_cp + self.gamma * loss_dp

        return total_loss, loss_cls.item(), loss_cp.item(), loss_dp.item()