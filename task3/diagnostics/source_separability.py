"""
task3/diagnostics/source_separability.py
Evaluates 3-way source domain separability (Photo vs Art Painting vs Cartoon)
using a multinomial Logistic Regression classifier (C=1.0, 70/30 split, seed 6304).
Chance performance is 33.33%.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@torch.no_grad()
def extract_source_features(model, val_loaders, device):
    """
    Extracts features for Photo (0), Art Painting (1), and Cartoon (2).
    Balances classes across domains to the minimum domain size.
    """
    model.eval()
    domain_features = {}
    domains = ["photo", "art_painting", "cartoon"]

    for d_idx, domain in enumerate(domains):
        feats_list = []
        loader = val_loaders[domain]
        for batch in loader:
            images = batch[0].to(device)
            if hasattr(model, "backbone"):
                feats = model.backbone(images)
            else:
                feats = model.extract_features(images)
            feats_list.append(feats.cpu().numpy())
        domain_features[domain] = np.concatenate(feats_list, axis=0)

    # Balance feature counts across domains
    min_count = min(len(feats) for feats in domain_features.values())
    rng = np.random.RandomState(6304)

    balanced_X = []
    balanced_y = []
    for d_idx, domain in enumerate(domains):
        feats = domain_features[domain]
        idx = rng.choice(len(feats), size=min_count, replace=False)
        balanced_X.append(feats[idx])
        balanced_y.append(np.full(min_count, d_idx, dtype=np.int64))

    X = np.concatenate(balanced_X, axis=0)
    y = np.concatenate(balanced_y, axis=0)
    return X, y


def compute_source_domain_separability(X, y, seed=6304):
    """
    Trains multinomial Logistic Regression on 70% train split and
    evaluates classification accuracy on the 30% held-out test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )

    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test) * 100.0
    return acc