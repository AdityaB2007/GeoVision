"""Image preprocessing, satellite-specific augmentation, and spectral utilities.

Core ML Concepts Taught in This Module:
1. Pixel Scaling & Z-Score Standardization:
   Raw 8-bit RGB images store integers in [0, 255]. Neural networks optimize
   much more smoothly when inputs are scaled to [0.0, 1.0] and standardized
   channel-wise via ``(x - mean) / std`` so every channel has ~0 mean and unit
   variance. This prevents large pixel magnitudes from saturating activations
   or creating ill-conditioned loss landscapes.
2. Preventing Data Leakage:
   When training a custom CNN from scratch, channel mean and std MUST be
   computed exclusively from the training split — never from validation or test
   images, which would leak test-distribution information into preprocessing.
3. Transfer Learning Statistics:
   Pre-trained backbones (e.g., ResNet-18 on ImageNet) learned their early edge
   and color-blob filters under ImageNet channel statistics. Reusing ImageNet
   mean/std aligns EuroSAT pixel ranges to what those filters expect.
4. Dihedral Symmetry (D4) in Remote Sensing:
   Unlike street-level photos (an upside-down car is unnatural), Sentinel-2
   imagery is nadir (top-down). Rotating a forest or crop field by 90/180/270
   degrees or flipping horizontally/vertically preserves the true land-cover
   class exactly — giving an 8x symmetry group for data augmentation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.config import EUROSAT_NATIVE_SIZE


class RandomRotate90:
    """Randomly rotate an image by 0, 90, 180, or 270 degrees.

    Unlike arbitrary-angle rotation (which introduces black border triangles
    or interpolation blur on 64x64 patches), 90-degree multiples are lossless
    and respect the grid alignment of satellite pixels.
    """

    ANGLES: tuple[int, ...] = (0, 90, 180, 270)

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.choice(self.ANGLES)
        if angle == 0:
            return img
        return TF.rotate(img, angle)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angles={self.ANGLES})"


def compute_channel_stats(
    image_paths: Sequence[Path],
    max_samples: int = 5000,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Compute exact per-channel RGB mean and std in [0, 1] on training images.

    Uses exact two-moment accumulation (E[X] and E[X^2]) so memory stays O(1)
    regardless of dataset size.
    """
    if not image_paths:
        raise ValueError("image_paths must not be empty when computing channel stats.")

    paths = list(image_paths)
    if len(paths) > max_samples:
        rng = random.Random(seed)
        paths = rng.sample(paths, max_samples)

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for path in paths:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        pixels = arr.reshape(-1, 3)
        channel_sum += pixels.sum(axis=0)
        channel_sq_sum += np.square(pixels).sum(axis=0)
        total_pixels += pixels.shape[0]

    mean = channel_sum / total_pixels
    variance = np.maximum(channel_sq_sum / total_pixels - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    return [round(float(x), 6) for x in mean], [round(float(x), 6) for x in std]


def build_transforms(
    input_size: int = EUROSAT_NATIVE_SIZE,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    is_train: bool = False,
) -> transforms.Compose:
    """Build deterministic evaluation/inference or augmented training transforms.

    Args:
        input_size: Target spatial height/width (e.g., 64 for custom CNN,
            128 or 224 for ImageNet transfer backbones).
        mean: Per-channel RGB normalization mean in [0, 1].
        std: Per-channel RGB normalization standard deviation in [0, 1].
        is_train: When True, enables D4 geometric symmetries and mild
            photometric jitter; when False, yields strictly deterministic
            preprocessing suitable for validation, testing, and API serving.
    """
    ops: list[object] = []
    if input_size != EUROSAT_NATIVE_SIZE:
        ops.append(
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )

    if is_train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                RandomRotate90(),
                # Mild brightness/contrast/saturation perturbation mimics
                # seasonal sun-angle and atmospheric haze shifts without
                # distorting class semantics (e.g., turning green forest blue).
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02
                ),
            ]
        )

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )
    return transforms.Compose(ops)


def compute_ndvi(ms_array: np.ndarray, red_idx: int = 3, nir_idx: int = 7) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index from 13-band Sentinel-2 array.

    NDVI = (NIR - Red) / (NIR + Red + eps)
    Healthy chlorophyll absorbs Red light (B04, 665 nm) for photosynthesis and
    strongly reflects Near-Infrared light (B08, 842 nm), yielding NDVI in [+0.3, +0.9]
    for dense forest/crops and negative values for open water.

    Args:
        ms_array: Array shaped ``(C, H, W)`` or ``(H, W, C)`` with >= 8 bands.
        red_idx: Band index for B04 Red (default 3).
        nir_idx: Band index for B08 NIR (default 7).
    """
    arr = np.asarray(ms_array, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D multi-spectral patch, got shape {arr.shape}")

    if arr.shape[0] <= 16 and arr.shape[2] > 16:
        # (C, H, W) layout
        red = arr[red_idx]
        nir = arr[nir_idx]
    else:
        # (H, W, C) layout (standard tifffile output)
        red = arr[:, :, red_idx]
        nir = arr[:, :, nir_idx]

    denom = nir + red
    eps = np.float32(1e-6)
    ndvi = np.where(np.abs(denom) > eps, (nir - red) / (denom + eps), 0.0)
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


def ms_to_rgb(
    ms_array: np.ndarray,
    clip_max: float = 2750.0,
    band_indices: tuple[int, int, int] = (3, 2, 1),
) -> np.ndarray:
    """Convert a 13-band Sentinel-2 reflectance patch to uint8 RGB.

    Matches Helber et al.'s official GDAL conversion recipe:
        ``-scale 0 2750 1 255 -b 4 -b 3 -b 2``
    where band indices (0-based) 3=B04 (Red), 2=B03 (Green), 1=B02 (Blue).
    Clipping at 2750 top-of-atmosphere reflectance prevents bright clouds or
    metallic roofs from compressing vegetative contrast into dark shadows.
    """
    arr = np.asarray(ms_array, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D multi-spectral patch, got shape {arr.shape}")

    if arr.shape[0] <= 16 and arr.shape[2] > 16:
        rgb = np.stack([arr[i] for i in band_indices], axis=-1)
    else:
        rgb = arr[:, :, list(band_indices)]

    scaled = np.clip(rgb, 0.0, clip_max) / clip_max * 255.0
    return np.rint(scaled).astype(np.uint8)
