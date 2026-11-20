"""Custom Convolutional Neural Network designed from scratch for EuroSAT.

Architecture Walkthrough (for 64x64x3 Sentinel-2 patches):
    - Block 1 (64x64 -> 32x32, 32 channels): Two 3x3 convs learn primitive
      local edge, color-contrast, and texture filters (receptive field 5x5)
      before 2x2 max pooling halves spatial dimensions.
    - Block 2 (32x32 -> 16x16, 64 channels): Stacking another pair of 3x3
      convs widens the receptive field to capture roof corners, crop rows,
      and tree-canopy textures while doubling channel capacity.
    - Block 3 (16x16 -> 8x8, 128 channels): Captures mid-level structural
      motifs such as road intersections, river bends, and orchard grids.
    - Block 4 (8x8 -> 4x4, 256 channels): Encodes high-level semantic scene
      patterns across the 640m x 640m satellite patch.
    - Head: AdaptiveAvgPool2d((1, 1)) collapses each 4x4 feature map into a
      single spatially invariant summary scalar (Global Average Pooling),
      drastically reducing parameters compared to flattening (256 vs 4096) and
      allowing the network to accept arbitrary input resolutions.

Design Notes:
    - ``bias=False`` precedes every ``BatchNorm2d`` because BatchNorm subtracts
      the channel mean and adds its own learnable shift ``beta``, rendering a
      preceding convolutional bias redundant.
    - ``Dropout`` in the classification head forces redundant feature pathways
      and mitigates overfitting on medium-sized datasets.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class GeoVisionCNN(nn.Module):
    """4-stage custom CNN with BatchNorm and Global Average Pooling."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            _conv_bn_relu(in_channels, 32),
            _conv_bn_relu(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 32x32 -> 16x16
            _conv_bn_relu(32, 64),
            _conv_bn_relu(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 16x16 -> 8x8
            _conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4: 8x8 -> 4x4
            _conv_bn_relu(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count total or trainable model parameters."""
        params = (
            p for p in self.parameters() if (not trainable_only or p.requires_grad)
        )
        return sum(p.numel() for p in params)
