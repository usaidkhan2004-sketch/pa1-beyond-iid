"""
task4/models/resnet_cifar.py
CIFAR-appropriate ResNet-18 implementation:
- 3x3 conv1 with stride 1, padding 1
- Initial max pooling removed
- Supports feature extraction and PROSER manifold mixup split at layer2
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet


class ResNet18CIFAR(ResNet):
    def __init__(self, num_classes: int = 10):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

        # Adapt for CIFAR-10 32x32 resolution
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()  # Remove max pooling

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate 512-dim feature embedding."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        return feats

    def forward_layer2(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts features after layer2 for PROSER manifold mixup."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward_from_layer2(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Continues forward pass from mixed layer2 representations."""
        x = self.layer3(h)
        x = self.layer4(x)
        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.fc(feats)
        return feats, logits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns both penultimate 512-dim features and classifier logits."""
        feats = self.forward_features(x)
        logits = self.fc(feats)
        return feats, logits