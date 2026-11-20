"""Unit tests for preprocessing, multi-spectral helpers, splits, and models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import EUROSAT_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from src.data.dataset import assert_disjoint_splits, load_splits
from src.data.preprocessing import (
    RandomRotate90,
    build_transforms,
    compute_channel_stats,
    compute_ndvi,
    ms_to_rgb,
)
from src.models.custom_cnn import GeoVisionCNN
from src.models.transfer import GeoVisionTransferNet


def test_random_rotate_90_preserves_square_size() -> None:
    img = Image.new("RGB", (64, 64), color=(10, 120, 40))
    rot = RandomRotate90()
    for _ in range(10):
        out = rot(img)
        assert out.size == (64, 64)


def test_build_transforms_shape_and_dtype() -> None:
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    tf_eval = build_transforms(
        input_size=128, mean=IMAGENET_MEAN, std=IMAGENET_STD, is_train=False
    )
    tf_train = build_transforms(
        input_size=64, mean=IMAGENET_MEAN, std=IMAGENET_STD, is_train=True
    )

    out_eval = tf_eval(img)
    out_train = tf_train(img)
    assert out_eval.shape == (3, 128, 128)
    assert out_eval.dtype == torch.float32
    assert out_train.shape == (3, 64, 64)
    assert out_train.dtype == torch.float32


def test_compute_channel_stats_on_synthetic_images(tmp_path: Path) -> None:
    paths: list[Path] = []
    for i in range(4):
        p = tmp_path / f"img_{i}.png"
        arr = np.full((16, 16, 3), fill_value=127 if i % 2 == 0 else 255, dtype=np.uint8)
        Image.fromarray(arr).save(p)
        paths.append(p)

    mean, std = compute_channel_stats(paths, max_samples=10, seed=42)
    assert len(mean) == 3 and len(std) == 3
    assert all(0.0 <= m <= 1.0 for m in mean)
    assert all(s > 0.0 for s in std)


def test_multispectral_ndvi_and_rgb_helpers() -> None:
    # Synthetic 13-band patch shaped (H, W, 13)
    ms = np.zeros((64, 64, 13), dtype=np.float32)
    ms[:, :, 1] = 550.0   # B02 Blue
    ms[:, :, 2] = 1100.0  # B03 Green
    ms[:, :, 3] = 400.0   # B04 Red
    ms[:, :, 7] = 2800.0  # B08 NIR

    ndvi = compute_ndvi(ms)
    assert ndvi.shape == (64, 64)
    assert ndvi.dtype == np.float32
    expected_ndvi = (2800.0 - 400.0) / (2800.0 + 400.0)
    assert np.allclose(ndvi, expected_ndvi, atol=1e-4)

    rgb = ms_to_rgb(ms)
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0, 0] < rgb[0, 0, 1]  # Green > Red


def test_stratified_split_zero_overlap(tmp_path: Path) -> None:
    # Create tiny synthetic dataset with all 10 EuroSAT class folders
    for cls_name in EUROSAT_CLASSES:
        cls_dir = tmp_path / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            Image.new("RGB", (64, 64), color=(i * 20, 80, 40)).save(
                cls_dir / f"{cls_name}_{i}.jpg"
            )

    splits = load_splits(
        rgb_root=tmp_path,
        strategy="stratified",
        max_per_class=4,
        seed=42,
    )
    assert_disjoint_splits(splits)
    assert len(splits["train"]) == 40
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10


def test_custom_cnn_forward_and_param_count() -> None:
    model = GeoVisionCNN(num_classes=10)
    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 10)
    assert model.count_parameters() > 0


def test_transfer_net_freezing_and_unfreezing() -> None:
    # pretrained=False keeps unit tests offline and deterministic
    model = GeoVisionTransferNet(backbone="resnet18", num_classes=10, pretrained=False)
    x = torch.randn(2, 3, 128, 128)
    logits = model(x)
    assert logits.shape == (2, 10)

    model.freeze_backbone()
    frozen_trainable = model.count_parameters(trainable_only=True)
    assert frozen_trainable == sum(p.numel() for p in model.net.fc.parameters())

    model.unfreeze_top_layers()
    tuned_trainable = model.count_parameters(trainable_only=True)
    assert tuned_trainable > frozen_trainable
    groups = model.get_param_groups(backbone_lr=1e-4, head_lr=1e-3)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4
    assert groups[1]["lr"] == 1e-3
