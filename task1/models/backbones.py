from typing import Tuple, Callable
import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)
import open_clip


def get_device() -> torch.device:
    """Select CUDA if available (NVIDIA GPU), else MPS (Mac M-series), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_resnet50() -> Tuple[nn.Module, Callable, int]:
    """Loads frozen ResNet-50 (IMAGENET1K_V2) returning (backbone, transform, embedding_dim)."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    transform = weights.transforms()
    model = resnet50(weights=weights)

    # Strip the final classification layer to extract 2048-dim pool5 features
    model.fc = nn.Identity()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, transform, 2048


def load_vit_b16() -> Tuple[nn.Module, Callable, int]:
    """Loads frozen ViT-B/16 (IMAGENET1K_V1) returning (backbone, transform, embedding_dim)."""
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    model = vit_b_16(weights=weights)

    # Strip the classification head to extract 768-dim class token features
    model.heads = nn.Identity()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, transform, 768


class OpenCLIPWrapper(nn.Module):
    """Wraps OpenCLIP ViT-B-32 for frozen image feature extraction and zero-shot scoring."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or get_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.embed_dim = 512

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts normalized visual embeddings."""
        images = images.to(self.device)
        image_features = self.model.encode_image(images)
        return image_features / image_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_zero_shot_weights(self, class_names: list) -> torch.Tensor:
        """Constructs normalized zero-shot text classifier weights."""
        prompts = [f"a photo of a {name}" for name in class_names]
        tokens = self.tokenizer(prompts).to(self.device)
        text_features = self.model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.T  # (embed_dim, num_classes)


def load_clip_vit_b32() -> Tuple[OpenCLIPWrapper, Callable, int]:
    """Loads frozen OpenCLIP ViT-B-32 returning (wrapper, transform, embedding_dim)."""
    device = get_device()
    wrapper = OpenCLIPWrapper(device=device)
    return wrapper, wrapper.preprocess, wrapper.embed_dim