"""Perceptual colormaps, RGB blending, and gallery rendering for Grad-CAM.

Why use perceptual colormaps (``turbo``, ``inferno``, ``magma``, ``viridis``)?
    Unlike naive grayscale or classic rainbow/jet maps, perceptual colormaps
    preserve monotonic luminance gradients so hot attribution peaks stand out
    cleanly when alpha-blended over Sentinel-2 satellite imagery.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SUPPORTED_COLORMAPS: tuple[str, ...] = ("turbo", "inferno", "magma", "viridis")


def _validate_colormap(colormap: str) -> str:
    if colormap not in SUPPORTED_COLORMAPS:
        raise ValueError(
            f"Unsupported colormap '{colormap}'. Choose from {SUPPORTED_COLORMAPS}."
        )
    return colormap


def heatmap_to_rgb(
    heatmap: np.ndarray,
    size: tuple[int, int] = (256, 256),
    colormap: str = "turbo",
) -> Image.Image:
    """Upsample a [0, 1] Grad-CAM grid smoothly and apply a perceptual colormap."""
    cmap_name = _validate_colormap(colormap)
    arr = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D heatmap array, got shape {arr.shape}")

    hm_uint8 = np.rint(arr * 255.0).astype(np.uint8)
    hm_resized = Image.fromarray(hm_uint8, mode="L").resize(
        size, resample=Image.Resampling.BICUBIC
    )
    norm_arr = np.asarray(hm_resized, dtype=np.float32) / 255.0

    rgba = matplotlib.colormaps[cmap_name](norm_arr)
    rgb_uint8 = np.rint(rgba[:, :, :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb_uint8, mode="RGB")


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    size: tuple[int, int] = (256, 256),
    colormap: str = "turbo",
) -> Image.Image:
    """Blend a Grad-CAM heatmap over the original satellite RGB image."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be within [0.0, 1.0], got {alpha}")

    base_rgb = image.convert("RGB").resize(size, resample=Image.Resampling.BICUBIC)
    heat_rgb = heatmap_to_rgb(heatmap, size=size, colormap=colormap)

    base_arr = np.asarray(base_rgb, dtype=np.float32)
    heat_arr = np.asarray(heat_rgb, dtype=np.float32)
    blended = (1.0 - alpha) * base_arr + alpha * heat_arr
    return Image.fromarray(np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8), mode="RGB")


def image_to_base64_png(image: Image.Image) -> str:
    """Encode a PIL Image into a base64 PNG data URI ready for JSON/HTML <img>."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def save_gradcam_gallery(items: list[dict], save_path: Path) -> Path:
    """Render a 2x5 class gallery showing Original RGB beside Grad-CAM Overlay.

    Each item expects keys: ``display_name``, ``predicted_display``, ``confidence``,
    ``correct``, ``original_image``, and ``overlay_image``.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_items = len(items)
    ncols = 2
    nrows = (n_items + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols * 2,
        figsize=(13.5, 2.75 * nrows),
        dpi=160,
        facecolor="#0b0f14",
    )
    axes_2d = np.atleast_2d(axes)

    for idx, item in enumerate(items):
        r = idx // ncols
        c_base = (idx % ncols) * 2
        ax_orig = axes_2d[r, c_base]
        ax_cam = axes_2d[r, c_base + 1]

        ax_orig.imshow(item["original_image"])
        ax_orig.set_title(
            f"{item['display_name']} (RGB)",
            color="#f2f5f7",
            fontsize=9.5,
            fontweight="bold",
        )
        ax_orig.axis("off")

        status_color = "#55d6a4" if item["correct"] else "#e17a7a"
        ax_cam.imshow(item["overlay_image"])
        ax_cam.set_title(
            f"Pred: {item['predicted_display']} ({item['confidence'] * 100:.1f}%)",
            color=status_color,
            fontsize=9.5,
            fontweight="bold",
        )
        ax_cam.axis("off")

    for empty_idx in range(n_items, nrows * ncols):
        r = empty_idx // ncols
        c_base = (empty_idx % ncols) * 2
        axes_2d[r, c_base].axis("off")
        axes_2d[r, c_base + 1].axis("off")

    fig.suptitle(
        "GeoVision ResNet-18 — 10-Class EuroSAT Grad-CAM Attention Gallery",
        color="#f2f5f7",
        fontsize=13.5,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=160, facecolor="#0b0f14")
    plt.close(fig)
    return save_path
