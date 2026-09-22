"""
shared/pacs_protocol.py
Protocol splitting and balanced multi-domain data loading for Tasks 2 and 3.
"""

import itertools
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader

from shared.pacs import (
    PACSDataset,
    find_domain_directory,
    get_pacs_transform,
    scan_domain_samples,
)

SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"
DEFAULT_SEED = 6304


def build_or_load_splits(
    dataset_root: str,
    split_file_path: str = "shared/splits/pacs_sketch_seed6304.json",
    seed: int = DEFAULT_SEED,
) -> Dict:
    """
    Ensures a fixed stratified 80/20 train/val split for source domains
    and stores image paths in a persistent JSON file.
    """
    split_file = Path(split_file_path)
    root_path = Path(dataset_root)

    if split_file.is_file():
        with open(split_file, "r") as f:
            return json.load(f)

    split_file.parent.mkdir(parents=True, exist_ok=True)
    splits_dict = {"seed": seed, "sources": {}, "target": {}}

    for s_dom in SOURCE_DOMAINS:
        s_dir = find_domain_directory(root_path, s_dom)
        samples = scan_domain_samples(s_dir)
        labels = [s[1] for s in samples]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_idx, val_idx = next(sss.split(samples, labels))

        splits_dict["sources"][s_dom] = {
            "train": [samples[i] for i in train_idx],
            "val": [samples[i] for i in val_idx],
        }

    t_dir = find_domain_directory(root_path, TARGET_DOMAIN)
    target_samples = scan_domain_samples(t_dir)
    splits_dict["target"][TARGET_DOMAIN] = {
        "all": target_samples
    }

    with open(split_file, "w") as f:
        json.dump(splits_dict, f, indent=2)

    return splits_dict


class BalancedDomainBatchIterator:
    """
    Yields balanced batches per training step:
    - 8 Photo, 8 Art, 8 Cartoon -> 24 Source samples total
    - 24 Sketch (Target) samples
    """

    def __init__(
        self,
        source_loaders: Dict[str, DataLoader],
        target_loader: DataLoader,
        steps_per_epoch: int,
    ):
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.steps_per_epoch = steps_per_epoch

        self.source_iters = {
            d: itertools.cycle(loader) for d, loader in source_loaders.items()
        }
        self.target_iter = itertools.cycle(target_loader)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        for _ in range(self.steps_per_epoch):
            source_imgs = []
            source_labels = []

            for d in SOURCE_DOMAINS:
                imgs, lbls, _ = next(self.source_iters[d])
                source_imgs.append(imgs)
                source_labels.append(lbls)

            x_s = torch.cat(source_imgs, dim=0)
            y_s = torch.cat(source_labels, dim=0)
            x_t, _, _ = next(self.target_iter)

            yield x_s, y_s, x_t


def get_pacs_dataloaders(
    dataset_root: str,
    split_file_path: str = "shared/splits/pacs_sketch_seed6304.json",
    seed: int = DEFAULT_SEED,
    num_workers: int = 2,
):
    """
    Instantiates training iterators and validation/test DataLoaders.
    """
    splits = build_or_load_splits(dataset_root, split_file_path, seed)

    train_transform = get_pacs_transform("train")
    eval_transform = get_pacs_transform("eval")

    source_train_loaders = {}
    total_source_train_samples = 0
    for d in SOURCE_DOMAINS:
        ds = PACSDataset(splits["sources"][d]["train"], transform=train_transform)
        total_source_train_samples += len(ds)
        source_train_loaders[d] = DataLoader(
            ds,
            batch_size=8,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    target_all_ds = PACSDataset(
        splits["target"][TARGET_DOMAIN]["all"], transform=train_transform
    )
    target_train_loader = DataLoader(
        target_all_ds,
        batch_size=24,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    steps_per_epoch = total_source_train_samples // 24

    train_iterator = BalancedDomainBatchIterator(
        source_train_loaders, target_train_loader, steps_per_epoch
    )

    val_loaders = {}
    for d in SOURCE_DOMAINS:
        ds = PACSDataset(splits["sources"][d]["val"], transform=eval_transform)
        val_loaders[d] = DataLoader(
            ds, batch_size=32, shuffle=False, num_workers=num_workers
        )

    target_test_ds = PACSDataset(
        splits["target"][TARGET_DOMAIN]["all"], transform=eval_transform
    )
    target_test_loader = DataLoader(
        target_test_ds, batch_size=32, shuffle=False, num_workers=num_workers
    )

    return train_iterator, val_loaders, target_test_loader