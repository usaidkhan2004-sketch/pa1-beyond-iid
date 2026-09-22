"""
task2/models/backbone.py
Modular ResNet-18 feature extractor and classification head for PACS.
Follows manual specifications:
- ResNet18_Weights.IMAGENET1K_V1
- 7-class linear classifier head
- Helper function to freeze BatchNorm running stats while keeping scale/bias trainable
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def set_train_mode_with_frozen_bn(model: nn.Module):
    """
    Manual specification (Page 6):
    Place the complete model in training mode, but keep ALL BatchNorm
    modules in evaluation mode so running mean and variance are NOT updated.
    The scale and bias parameters (gamma and beta) remain trainable.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 backbone pretrained on ImageNet-1K V1.
    Outputs 512-dim feature embeddings immediately before the classification head.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = resnet18(weights=weights)

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = base_model.avgpool
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        return features


class TaskClassifier(nn.Module):
    """
    Linear classification head mapping 512-dim features to 7 PACS classes.
    """
    def __init__(self, in_features: int = 512, num_classes: int = 7):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class SourceOnlyClassifier(nn.Module):
    """
    End-to-end baseline model combining backbone and classifier.
    """
    def __init__(self, pretrained: bool = True, num_classes: int = 7):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained)
        self.classifier = TaskClassifier(in_features=self.backbone.feature_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits