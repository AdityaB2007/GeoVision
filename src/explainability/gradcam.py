"""Gradient-weighted Class Activation Mapping (Grad-CAM) for GeoVision CNNs.

Mathematical Intuition:
    Given a target spatial convolutional layer producing feature maps
    ``A^k in R^{H' x W'}`` and a target class logit ``y^c`` (pre-softmax):

    1. **Importance Weights (alpha_k^c):**
       Global-average-pool the backward gradient with respect to feature map
       ``A^k``:
           ``alpha_k^c = (1 / Z) * sum_i sum_j (dy^c / dA^k_{i,j})``
       This scalar measures how strongly channel ``k`` pushes the model toward
       class ``c``.
    2. **Weighted Positive Evidence (ReLU Combination):**
           ``L^c_{Grad-CAM} = ReLU( sum_k alpha_k^c * A^k )``
       ``ReLU`` discards negative activations (pixels belonging to competing
       classes) so the heatmap isolates ONLY image regions that actively
       support class ``c``.

Implementation Safety Note:
    PyTorch's ``register_full_backward_hook`` can raise a ``RuntimeError`` when
    a module contains in-place ``ReLU(inplace=True)`` ops (used inside both
    ResNet and EfficientNet). Attaching a tensor-level hook via
    ``output.register_hook(...)`` inside the forward hook captures exact
    gradients cleanly without colliding with in-place autograd buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.models.custom_cnn import GeoVisionCNN
from src.models.transfer import GeoVisionTransferNet


@dataclass
class GradCAMResult:
    """Container for a single Grad-CAM attribution pass."""

    heatmap: np.ndarray
    predicted_idx: int
    target_idx: int
    probabilities: np.ndarray
    target_layer_name: str


def resolve_target_layer(model: nn.Module) -> tuple[str, nn.Module]:
    """Select the final spatial convolutional block for Grad-CAM attribution.

    Why the final convolutional layer?
        Early conv layers have crisp spatial resolution but only encode generic
        edges. The final conv block before global pooling strikes the optimal
        compromise: high-level land-cover semantics (e.g., road stripes, roofs,
        tree canopies) while retaining spatial localization (e.g., 4x4 or 8x8).
    """
    if isinstance(model, GeoVisionTransferNet):
        if model.backbone_name in {"resnet18", "resnet50"}:
            return "net.layer4[-1]", model.net.layer4[-1]
        if model.backbone_name == "efficientnet_b0":
            return "net.features[-1]", model.net.features[-1]

    if isinstance(model, GeoVisionCNN):
        # features[-1] is MaxPool2d(2) (4x4); features[-2] is the final Conv-BN-ReLU
        # block at 8x8 resolution for 64x64 patches, giving crisper spatial heatmaps.
        return "features[-2]", model.features[-2]

    last_name: str | None = None
    last_conv: nn.Module | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_conv = module

    if last_conv is None or last_name is None:
        raise ValueError("Could not find a Conv2d target layer for Grad-CAM.")
    return last_name, last_conv


class GradCAM:
    """Hook-based Grad-CAM attribution engine supporting CNN & Transfer backbones."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        target_layer_name: str | None = None,
    ) -> None:
        self.model = model
        if target_layer is None:
            resolved_name, resolved_layer = resolve_target_layer(model)
            self.target_layer_name = target_layer_name or resolved_name
            self.target_layer = resolved_layer
        else:
            self.target_layer = target_layer
            self.target_layer_name = target_layer_name or target_layer.__class__.__name__

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> GradCAMResult:
        """Compute a [0, 1]-normalized Grad-CAM heatmap for ``(1, C, H, W)`` input.

        Args:
            input_tensor: Preprocessed input batch shaped ``(1, C, H, W)``.
            target_class: Optional class index to explain (enables counterfactual
                "where does the model see class X?" inspection). Defaults to
                the top predicted class when ``None``.
        """
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"GradCAM expects input shaped (1, C, H, W), got {tuple(input_tensor.shape)}"
            )

        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def _save_grad(grad: torch.Tensor) -> None:
            gradients.append(grad.detach())

        def _forward_hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            activations.append(output.detach())
            if output.requires_grad:
                output.register_hook(_save_grad)

        was_training = self.model.training
        self.model.eval()
        handle = self.target_layer.register_forward_hook(_forward_hook)

        try:
            with torch.enable_grad():
                # Marking the input tensor requires_grad=True guarantees autograd
                # builds a graph through the target conv activations even if all
                # backbone weights happen to be frozen (requires_grad=False).
                x = input_tensor.detach().clone().requires_grad_(True)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                predicted_idx = int(np.argmax(probs))
                chosen_idx = predicted_idx if target_class is None else int(target_class)

                if not (0 <= chosen_idx < logits.shape[1]):
                    raise ValueError(
                        f"target_class index {chosen_idx} out of range [0, {logits.shape[1] - 1}]"
                    )

                self.model.zero_grad(set_to_none=True)
                score = logits[0, chosen_idx]
                score.backward()

            if not activations or not gradients:
                raise RuntimeError(
                    f"Grad-CAM failed to capture activations/gradients at {self.target_layer_name}."
                )

            act = activations[0]   # (1, K, H', W')
            grad = gradients[0]    # (1, K, H', W')
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act).sum(dim=1)).squeeze(0)

            cam_min = cam.min()
            cam_max = cam.max()
            denom = (cam_max - cam_min).clamp(min=1e-8)
            cam_norm = ((cam - cam_min) / denom).cpu().numpy().astype(np.float32)

            return GradCAMResult(
                heatmap=cam_norm,
                predicted_idx=predicted_idx,
                target_idx=chosen_idx,
                probabilities=probs.astype(np.float32),
                target_layer_name=self.target_layer_name,
            )
        finally:
            handle.remove()
            self.model.zero_grad(set_to_none=True)
            self.model.train(mode=was_training)
