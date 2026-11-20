"""Unit tests for Grad-CAM attribution and heatmap visualization utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.explainability.gradcam import GradCAM
from src.explainability.visualization import (
    heatmap_to_rgb,
    image_to_base64_png,
    overlay_heatmap,
)
from src.models.custom_cnn import GeoVisionCNN
from src.models.transfer import GeoVisionTransferNet


@pytest.mark.parametrize(
    "model_factory,input_size,expected_layer",
    [
        (lambda: GeoVisionCNN(num_classes=10), 64, "features[-2]"),
        (
            lambda: GeoVisionTransferNet(
                backbone="resnet18", num_classes=10, pretrained=False
            ),
            128,
            "net.layer4[-1]",
        ),
        (
            lambda: GeoVisionTransferNet(
                backbone="efficientnet_b0", num_classes=10, pretrained=False
            ),
            128,
            "net.features[-1]",
        ),
    ],
)
def test_gradcam_across_architectures(
    model_factory, input_size: int, expected_layer: str
) -> None:
    model = model_factory()
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    cam = GradCAM(model)
    assert cam.target_layer_name == expected_layer

    x = torch.randn(1, 3, input_size, input_size)
    result = cam.generate(x, target_class=3)

    assert result.target_idx == 3
    assert result.target_layer_name == expected_layer
    assert result.heatmap.ndim == 2
    assert result.heatmap.dtype == np.float32
    assert np.isfinite(result.heatmap).all()
    assert float(result.heatmap.min()) >= 0.0
    assert float(result.heatmap.max()) <= 1.0
    assert result.probabilities.shape == (10,)
    assert abs(float(result.probabilities.sum()) - 1.0) < 1e-4
    # Ensure forward hook was cleanly detached after generation
    assert len(cam.target_layer._forward_hooks) == 0


def test_gradcam_visualization_helpers() -> None:
    img = Image.new("RGB", (64, 64), color=(40, 120, 60))
    heatmap = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)

    heat_rgb = heatmap_to_rgb(heatmap, size=(256, 256), colormap="turbo")
    overlay_rgb = overlay_heatmap(
        img, heatmap, alpha=0.45, size=(256, 256), colormap="inferno"
    )
    b64 = image_to_base64_png(overlay_rgb)

    assert heat_rgb.size == (256, 256)
    assert heat_rgb.mode == "RGB"
    assert overlay_rgb.size == (256, 256)
    assert overlay_rgb.mode == "RGB"
    assert b64.startswith("data:image/png;base64,")
