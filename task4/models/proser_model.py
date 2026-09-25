"""
task4/models/proser_model.py
PROSER Model wrapper:
- Takes pretrained 10-class Vanilla ResNet-18
- Appends 5 dummy classifiers (total 15 outputs)
- Supports layer2 manifold mixup for synthetic data placeholders
"""

from typing import Tuple
import torch
import torch.nn as nn
from task4.models.resnet_cifar import ResNet18CIFAR


class PROSERNet(nn.Module):
    def __init__(self, vanilla_backbone: ResNet18CIFAR, num_known: int = 10, num_dummy: int = 5):
        super().__init__()
        self.num_known = num_known
        self.num_dummy = num_dummy
        self.num_classes = num_known + num_dummy

        # Retain backbone layers
        self.conv1 = vanilla_backbone.conv1
        self.bn1 = vanilla_backbone.bn1
        self.relu = vanilla_backbone.relu
        self.maxpool = vanilla_backbone.maxpool
        self.layer1 = vanilla_backbone.layer1
        self.layer2 = vanilla_backbone.layer2
        self.layer3 = vanilla_backbone.layer3
        self.layer4 = vanilla_backbone.layer4
        self.avgpool = vanilla_backbone.avgpool

        # Expanded linear head: 512 -> 15
        self.fc = nn.Linear(512, self.num_classes)

        # Copy over pretrained 10-class weights; initialize dummy classifiers
        with torch.no_grad():
            self.fc.weight[:num_known].copy_(vanilla_backbone.fc.weight)
            self.fc.bias[:num_known].copy_(vanilla_backbone.fc.bias)
            nn.init.kaiming_normal_(self.fc.weight[num_known:], mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(self.fc.bias[num_known:])

    def forward_layer2(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward_from_layer2(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.layer3(h)
        x = self.layer4(x)
        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.fc(feats)
        return feats, logits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.forward_layer2(x)
        feats, logits = self.forward_from_layer2(h)
        return feats, logits