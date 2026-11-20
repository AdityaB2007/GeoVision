"""Transfer learning models leveraging ImageNet-pretrained CNN backbones.

Why Transfer Learning Works for Satellite Imagery:
    Even though ImageNet consists of ground-level natural photos rather than
    overhead Sentinel-2 patches, the early and intermediate layers of ResNet
    and EfficientNet learn universal visual primitives — oriented Gabor-like
    edges, color gradients, textures, and geometric corners. Reusing these
    pre-trained representations dramatically speeds up convergence and raises
    accuracy compared to random initialization.

Two-Stage Fine-Tuning Strategy:
    1. Linear Probing / Head Warmup (``freeze_backbone()``):
       Freeze all convolutional weights and train ONLY the newly initialized
       classification head. If we unfreeze the entire backbone immediately,
       large random gradients from the untrained head can overwrite delicate
       pre-trained filters.
    2. Selective Fine-Tuning (``unfreeze_top_layers()``):
       Unfreeze the deepest semantic stage (e.g., ResNet ``layer4`` or
       EfficientNet's final two feature blocks) together with the head using
       a 10x smaller learning rate on the backbone (differential LR), adapting
       high-level features specifically to overhead Earth-observation semantics.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

BackboneName = Literal["resnet18", "resnet50", "efficientnet_b0"]
SUPPORTED_BACKBONES: tuple[str, ...] = ("resnet18", "resnet50", "efficientnet_b0")


class GeoVisionTransferNet(nn.Module):
    """Transfer learning classifier supporting ResNet-18, ResNet-50, and EfficientNet-B0."""

    def __init__(
        self,
        backbone: BackboneName = "resnet18",
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. Choose from {SUPPORTED_BACKBONES}."
            )

        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            self.net = net
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            self.net = net
        else:  # efficientnet_b0
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            in_features = net.classifier[1].in_features
            net.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _head_module(self) -> nn.Module:
        if self.backbone_name.startswith("resnet"):
            return self.net.fc
        return self.net.classifier

    def freeze_backbone(self) -> None:
        """Stage 1: Freeze all pre-trained feature layers; train only the head."""
        for param in self.net.parameters():
            param.requires_grad = False
        for param in self._head_module().parameters():
            param.requires_grad = True

    def unfreeze_top_layers(self) -> None:
        """Stage 2: Unfreeze the top semantic block plus the classifier head."""
        self.freeze_backbone()
        if self.backbone_name.startswith("resnet"):
            for param in self.net.layer4.parameters():
                param.requires_grad = True
        else:
            for block in list(self.net.features.children())[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Full end-to-end fine-tuning across every layer."""
        for param in self.net.parameters():
            param.requires_grad = True

    def get_param_groups(
        self, backbone_lr: float, head_lr: float
    ) -> list[dict[str, object]]:
        """Return differential learning-rate parameter groups for AdamW.

        Keeps updates to pre-trained convolutional layers gentle (``backbone_lr``)
        while allowing the task-specific classification head to adapt faster
        (``head_lr``).
        """
        head = self._head_module()
        head_ids = {id(p) for p in head.parameters()}

        backbone_params = [
            p
            for p in self.net.parameters()
            if p.requires_grad and id(p) not in head_ids
        ]
        head_params = [p for p in head.parameters() if p.requires_grad]

        groups: list[dict[str, object]] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        return groups

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count total or currently trainable model parameters."""
        params = (
            p for p in self.parameters() if (not trainable_only or p.requires_grad)
        )
        return sum(p.numel() for p in params)
