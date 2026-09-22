"""
shared/pacs.py
PyTorch Dataset definition and transforms for the PACS dataset.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Fixed class ordering for PACS
PACS_CLASSES = sorted([
    "dog",
    "elephant",
    "giraffe",
    "guitar",
    "horse",
    "house",
    "person",
])
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(PACS_CLASSES)}

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_pacs_transform(split: str = "train") -> Callable:
    """
    Returns transforms per assignment specifications:
    - Train: Resize to 256x256, RandomCrop 224x224, RandomHorizontalFlip, Normalize.
    - Val/Eval: Resize to 256x256, CenterCrop 224x224, Normalize.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class PACSDataset(Dataset):
    """
    Standard dataset loading image files for a single domain in PACS.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[Callable] = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[object, int, str]:
        path_str, label = self.samples[idx]
        image = Image.open(path_str).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, path_str


def find_domain_directory(root_dir: Path, domain_name: str) -> Path:
    """
    Resolves variations in domain folder naming (e.g. art_painting vs ArtPainting).
    """
    candidates = [
        domain_name.lower(),
        domain_name.lower().replace("_", ""),
        domain_name.lower().replace("_", " "),
    ]
    for child in root_dir.iterdir():
        if child.is_dir():
            norm_name = child.name.lower().replace("_", "").replace(" ", "")
            for c in candidates:
                if norm_name == c.replace("_", "").replace(" ", ""):
                    return child
    raise FileNotFoundError(
        f"Domain '{domain_name}' not found under dataset root {root_dir}"
    )


def scan_domain_samples(domain_dir: Path) -> List[Tuple[str, int]]:
    """
    Scans a domain folder and pairs valid image paths with class indices.
    """
    samples = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_folder in sorted(domain_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        cls_name = class_folder.name.lower()
        if cls_name not in CLASS_TO_IDX:
            continue
        label = CLASS_TO_IDX[cls_name]

        for img_path in sorted(class_folder.iterdir()):
            if img_path.suffix.lower() in valid_exts:
                samples.append((str(img_path.resolve()), label))

    return samples