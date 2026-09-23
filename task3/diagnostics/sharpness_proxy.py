"""
task3/diagnostics/sharpness_proxy.py
Computes the standardized local sharpness proxy:
    Delta_sharp = L_val(theta + epsilon) - L_val(theta)
with epsilon = 0.05 * (grad / ||grad||_2) on a fixed validation batch
of 32 examples from each source domain (96 total), using seed 6304.
"""

import numpy as np
import torch
import torch.nn as nn


def get_fixed_sharpness_batch(val_loaders, device, seed=6304):
    """Samples exactly 32 examples from each source domain deterministically."""
    domains = ["photo", "art_painting", "cartoon"]
    batch_images, batch_labels = [], []
    rng = np.random.RandomState(seed)

    for domain in domains:
        all_imgs, all_lbls = [], []
        loader = val_loaders[domain]
        for batch in loader:
            imgs, lbls = batch[0], batch[1]
            all_imgs.append(imgs)
            all_lbls.append(lbls)
        all_imgs = torch.cat(all_imgs, dim=0)
        all_lbls = torch.cat(all_lbls, dim=0)

        # Draw fixed 32 samples
        idx = rng.choice(len(all_imgs), size=32, replace=False)
        batch_images.append(all_imgs[idx])
        batch_labels.append(all_lbls[idx])

    fixed_x = torch.cat(batch_images, dim=0).to(device)
    fixed_y = torch.cat(batch_labels, dim=0).to(device)
    return fixed_x, fixed_y


def compute_sharpness_proxy(model, fixed_x, fixed_y, rho=0.05):
    """
    Computes Delta_sharp on the fixed validation batch.
    Preserves model weights by restoring original parameters afterwards.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Step 1: Base validation loss at theta
    model.zero_grad()
    logits = model(fixed_x)
    base_loss = criterion(logits, fixed_y)
    base_loss.backward()

    # Step 2: Compute L2 norm across all parameter gradients
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)

    if total_norm == 0.0:
        model.zero_grad()
        return 0.0

    # Step 3: Perturb theta -> theta + epsilon
    scale = rho / (total_norm + 1e-12)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.add_(p.grad * scale)

    # Step 4: Compute loss at perturbed point
    with torch.no_grad():
        perturbed_logits = model(fixed_x)
        perturbed_loss = criterion(perturbed_logits, fixed_y)

    delta_sharp = (perturbed_loss - base_loss).item()

    # Step 5: Restore original parameters: theta + epsilon -> theta
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.sub_(p.grad * scale)

    model.zero_grad()
    return delta_sharp 