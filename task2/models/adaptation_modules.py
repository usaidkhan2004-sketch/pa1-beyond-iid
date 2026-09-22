"""
task2/models/adaptation_modules.py
Adaptation components matching manual specifications:
1. Multi-Kernel Maximum Mean Discrepancy (MK-MMD) for DAN with median heuristic.
2. Gradient Reversal Layer (GRL) with dynamic alpha(p) schedule for DANN/CDAN.
3. Domain Discriminator (256 hidden units, ReLU, Dropout 0.5, 2-class output).
4. CDAN multilinear feature tensor conditioning: g(x) = vec(f (x) p).
"""

import torch
import torch.nn as nn
from torch.autograd import Function


# ==========================================
# 1. Multi-Kernel MMD (DAN)
# ==========================================
class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    """
    Sum of three RBF kernels whose bandwidths are 0.5, 1.0, and 2.0 times
    the median pairwise squared feature distance of the combined batch.
    """
    def __init__(self, multipliers=(0.5, 1.0, 2.0)):
        super().__init__()
        self.multipliers = multipliers

    def forward(self, source_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        batch_size = source_feats.size(0)
        combined = torch.cat([source_feats, target_feats], dim=0)

        # Pairwise squared L2 distances: ||z_i - z_j||^2
        pairwise_sq_dists = torch.cdist(combined, combined, p=2) ** 2

        # Median pairwise squared distance (detached so gradients do not flow through bandwidth)
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

        # Empirical MMD squared discrepancy
        mmd_loss = torch.mean(k_ss) + torch.mean(k_tt) - 2.0 * torch.mean(k_st)
        return mmd_loss


# ==========================================
# 2. Gradient Reversal Layer (GRL)
# ==========================================
class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


class GradientReverseLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def set_alpha(self, alpha: float):
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReverseFunction.apply(x, self.alpha)


# ==========================================
# 3. Domain Discriminator (Manual Specification)
# 256-unit hidden layer, ReLU, Dropout(0.5), 2-class output
# ==========================================
class DomainDiscriminator(nn.Module):
    def __init__(self, in_features: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2),  # 2-class output (Source=0, Target=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==========================================
# 4. CDAN Multilinear Conditioner: vec(f (x) p)
# ==========================================
def multilinear_conditioning(features: torch.Tensor, softmax_probs: torch.Tensor) -> torch.Tensor:
    """
    Given features [B, 512] and softmax predictions [B, 7],
    computes outer product and flattens: g(x) = vec(f (x) p) in R^{B, 3584}.
    No detaching is applied as mandated by manual.
    """
    batch_size = features.size(0)
    op = torch.bmm(features.unsqueeze(2), softmax_probs.unsqueeze(1))
    return op.view(batch_size, -1)