"""
models/cnn.py
-------------
CNN architectures used throughout the pipeline.

Provides:
  - SmallCNN   : lightweight 3-layer conv net (MNIST / quick tests)
  - RobustCNN  : deeper ResNet-style net (CIFAR-10 / GTSRB)
  - get_model  : factory function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# SmallCNN — good for MNIST, fast iteration
# ──────────────────────────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    """3-layer convolutional network for 1-channel 28×28 images (MNIST)."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28×28 → 28×28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 28×28 → 28×28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 28×28 → 14×14
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 14×14 → 14×14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 14×14 → 7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ──────────────────────────────────────────────────────────────────────────────
# Residual block
# ──────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """Basic residual block with optional downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# ──────────────────────────────────────────────────────────────────────────────
# RobustCNN — ResNet-18-style for CIFAR-10 / GTSRB (3-channel 32×32)
# ──────────────────────────────────────────────────────────────────────────────
class RobustCNN(nn.Module):
    """
    ResNet-18-inspired architecture for 3-channel inputs.
    Input:  (B, 3, 32, 32)  — CIFAR-10 / GTSRB resized
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(ResBlock(64, 64),  ResBlock(64, 64))
        self.layer2 = nn.Sequential(ResBlock(64, 128, stride=2),  ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.layer4 = nn.Sequential(ResBlock(256, 512, stride=2), ResBlock(512, 512))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────
def get_model(name: str, num_classes: int = 10, dropout: float = 0.1) -> nn.Module:
    """
    Factory function.

    Args:
        name        : 'small' | 'robust'
        num_classes : number of output classes
        dropout     : dropout rate

    Returns:
        nn.Module
    """
    name = name.lower()
    if name == "small":
        return SmallCNN(num_classes=num_classes, dropout=dropout)
    elif name == "robust":
        return RobustCNN(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose 'small' or 'robust'.")
