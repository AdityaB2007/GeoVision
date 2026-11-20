"""PyTorch inference engine that loads self-contained GeoVision checkpoints.

Production ML Principle — Preventing Train/Serve Skew:
    A common production bug is training a model with one resize/normalization
    pipeline and deploying an API that normalizes images differently.
    GeoVision checkpoints embed ``input_size``, ``mean``, ``std``, and
    ``class_names`` directly alongside ``state_dict``, and this engine
    reconstructs the exact deterministic transform via ``build_transforms``.
"""

from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from src.config import (
    BEST_MODEL_PATH,
    CLASS_DESCRIPTIONS,
    CLASS_DISPLAY_NAMES,
    EUROSAT_CLASSES,
)
from src.data.preprocessing import build_transforms
from src.explainability.gradcam import GradCAM
from src.explainability.visualization import (
    SUPPORTED_COLORMAPS,
    heatmap_to_rgb,
    image_to_base64_png,
    overlay_heatmap,
)
from src.models.custom_cnn import GeoVisionCNN
from src.models.transfer import GeoVisionTransferNet


def resolve_checkpoint_path(explicit_path: Path | None = None) -> Path:
    """Resolve checkpoint path from argument, env var, or default location."""
    if explicit_path is not None:
        return explicit_path
    env_path = os.getenv("GEOVISION_MODEL_PATH")
    if env_path:
        return Path(env_path)
    return BEST_MODEL_PATH


class GeoVisionInferenceEngine:
    """Loads a trained GeoVision checkpoint and serves calibrated predictions."""

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded: bool = False
        self.model: torch.nn.Module | None = None
        self.transform = None

        self.run_name: str | None = None
        self.model_type: str | None = None
        self.backbone: str | None = None
        self.class_names: list[str] = list(EUROSAT_CLASSES)
        self.input_size: int | None = None
        self.mean: list[float] = []
        self.std: list[float] = []
        self.metrics: dict = {}
        self.training_config: dict = {}
        self.saved_at: str | None = None
        # Serializes hook registration/backward passes across concurrent HTTP workers
        self._gradcam_lock = threading.Lock()

        if self.checkpoint_path.exists():
            self._load_checkpoint(self.checkpoint_path)

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.run_name = str(ckpt.get("run_name", path.stem))
        self.model_type = str(ckpt["model_type"])
        self.backbone = str(ckpt.get("backbone", self.model_type))
        self.class_names = list(ckpt.get("class_names", EUROSAT_CLASSES))
        self.input_size = int(ckpt["input_size"])
        self.mean = [float(x) for x in ckpt["mean"]]
        self.std = [float(x) for x in ckpt["std"]]
        self.metrics = dict(ckpt.get("metrics", {}))
        self.training_config = dict(ckpt.get("training_config", {}))
        self.saved_at = ckpt.get("saved_at")

        if self.model_type == "custom_cnn":
            model: torch.nn.Module = GeoVisionCNN(num_classes=len(self.class_names))
        elif self.model_type == "transfer":
            # pretrained=False avoids external weight downloads at inference time
            model = GeoVisionTransferNet(
                backbone=self.backbone,  # type: ignore[arg-type]
                num_classes=len(self.class_names),
                pretrained=False,
            )
        else:
            raise ValueError(f"Unknown model_type in checkpoint: {self.model_type}")

        model.load_state_dict(ckpt["state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model
        self.transform = build_transforms(
            input_size=self.input_size,
            mean=self.mean,
            std=self.std,
            is_train=False,
        )
        self.model_loaded = True

    def _decode_rgb_bytes(self, image_bytes: bytes) -> Image.Image:
        """Validate model readiness and decode raw bytes into an RGB PIL image."""
        if not self.model_loaded or self.model is None or self.transform is None:
            raise RuntimeError("No trained model checkpoint is currently loaded.")
        if not image_bytes:
            raise ValueError("Uploaded file is empty.")

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Invalid or corrupted image file: {exc}") from exc

    def _format_prediction_payload(
        self, probs: list[float], elapsed_ms: float
    ) -> dict:
        ranked = sorted(
            zip(self.class_names, probs),
            key=lambda item: item[1],
            reverse=True,
        )
        top_class, top_prob = ranked[0]
        return {
            "predicted_class": top_class,
            "display_name": CLASS_DISPLAY_NAMES.get(top_class, top_class),
            "description": CLASS_DESCRIPTIONS.get(top_class, ""),
            "confidence": round(float(top_prob), 6),
            "confidence_percentage": round(float(top_prob) * 100.0, 2),
            "probabilities": [
                {
                    "class_name": cls_name,
                    "display_name": CLASS_DISPLAY_NAMES.get(cls_name, cls_name),
                    "probability": round(float(prob), 6),
                    "percentage": round(float(prob) * 100.0, 2),
                }
                for cls_name, prob in ranked
            ],
            "model_name": self.run_name or "GeoVision",
            "backbone": self.backbone or "unknown",
            "input_size": int(self.input_size or 64),
            "inference_time_ms": round(elapsed_ms, 2),
        }

    def predict_bytes(self, image_bytes: bytes) -> dict:
        """Decode raw uploaded bytes, preprocess, and return sorted probabilities."""
        rgb_img = self._decode_rgb_bytes(image_bytes)
        assert self.model is not None and self.transform is not None

        t0 = time.perf_counter()
        tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)
        # torch.inference_mode() disables autograd tracking and view overhead,
        # cutting memory usage and latency compared to normal training mode.
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._format_prediction_payload(probs, elapsed_ms)

    def explain_bytes(
        self,
        image_bytes: bytes,
        target_class: str | None = None,
        alpha: float = 0.45,
        colormap: str = "turbo",
    ) -> dict:
        """Compute prediction plus thread-safe Grad-CAM heatmap and overlay PNGs."""
        if colormap not in SUPPORTED_COLORMAPS:
            raise ValueError(
                f"Unsupported colormap '{colormap}'. Choose from {SUPPORTED_COLORMAPS}."
            )
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")

        rgb_img = self._decode_rgb_bytes(image_bytes)
        assert self.model is not None and self.transform is not None

        target_idx: int | None = None
        if target_class is not None:
            if target_class not in self.class_names:
                raise ValueError(
                    f"Unknown target_class '{target_class}'. Choose from {self.class_names}."
                )
            target_idx = self.class_names.index(target_class)

        t0 = time.perf_counter()
        tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)
        with self._gradcam_lock:
            cam_engine = GradCAM(self.model)
            cam_res = cam_engine.generate(tensor, target_class=target_idx)
        gradcam_ms = (time.perf_counter() - t0) * 1000.0

        chosen_class = self.class_names[cam_res.target_idx]
        heat_img = heatmap_to_rgb(cam_res.heatmap, size=(256, 256), colormap=colormap)
        overlay_img = overlay_heatmap(
            rgb_img, cam_res.heatmap, alpha=alpha, size=(256, 256), colormap=colormap
        )

        payload = self._format_prediction_payload(
            cam_res.probabilities.tolist(), gradcam_ms
        )
        payload.update(
            {
                "target_class": chosen_class,
                "target_display_name": CLASS_DISPLAY_NAMES.get(chosen_class, chosen_class),
                "target_layer": cam_res.target_layer_name,
                "colormap": colormap,
                "alpha": float(alpha),
                "heatmap_base64": image_to_base64_png(heat_img),
                "overlay_base64": image_to_base64_png(overlay_img),
                "gradcam_time_ms": round(gradcam_ms, 2),
            }
        )
        return payload
