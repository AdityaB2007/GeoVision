"""Dataset split resolution and PyTorch Dataset / DataLoader builders.

Supports three splitting strategies so learners can explore data leakage:
1. ``official``: Uses TorchGeo's deterministic ``eurosat-{train,val,test}.txt``
   splits (60% / 20% / 20% = 16,200 / 5,400 / 5,400) for reproducible benchmarks.
2. ``spatial``: Uses TorchGeo's ``eurosat-spatial-{train,val,test}.txt`` splits,
   holding out entire geographic regions so neighboring patches cannot leak
   across train and test sets (preventing spatial autocorrelation bias).
3. ``stratified``: Generates a reproducible 60/20/20 stratified split directly
   from the extracted folders using scikit-learn.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import tifffile
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

from src.config import CLASS_TO_IDX, EUROSAT_CLASSES, SPLITS_DIR
from src.data.download import download_official_splits

SplitStrategy = Literal["official", "spatial", "stratified"]
SampleRecord = tuple[Path, int]


def _parse_split_file(split_file: Path, rgb_root: Path) -> list[SampleRecord]:
    """Read lines like ``Forest_1399.jpg`` into ``(path, class_idx)`` records."""
    if not split_file.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_file}")

    records: list[SampleRecord] = []
    with split_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            filename = raw_line.strip()
            if not filename:
                continue
            cls_name = filename.split("_", 1)[0]
            if cls_name not in CLASS_TO_IDX:
                raise ValueError(
                    f"Unexpected class prefix '{cls_name}' in {split_file}: {filename}"
                )
            img_path = rgb_root / cls_name / filename
            if not img_path.exists():
                raise FileNotFoundError(f"Listed image missing on disk: {img_path}")
            records.append((img_path, CLASS_TO_IDX[cls_name]))
    return records


def _build_stratified_splits(
    rgb_root: Path, seed: int = 42
) -> dict[str, list[SampleRecord]]:
    """Build deterministic 60/20/20 class-stratified splits from disk."""
    all_paths: list[Path] = []
    all_labels: list[int] = []
    for cls_name in EUROSAT_CLASSES:
        cls_dir = rgb_root / cls_name
        cls_idx = CLASS_TO_IDX[cls_name]
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                all_paths.append(img_path)
                all_labels.append(cls_idx)

    if not all_paths:
        raise FileNotFoundError(f"No EuroSAT RGB images found under {rgb_root}")

    labels_arr = np.asarray(all_labels)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
    train_idx, temp_idx = next(sss1.split(all_paths, labels_arr))

    temp_labels = labels_arr[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel_idx, test_rel_idx = next(sss2.split(temp_idx, temp_labels))
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    return {
        "train": [(all_paths[i], all_labels[i]) for i in train_idx],
        "val": [(all_paths[i], all_labels[i]) for i in val_idx],
        "test": [(all_paths[i], all_labels[i]) for i in test_idx],
    }


def _subsample_stratified(
    records: list[SampleRecord], per_class_limit: int, seed: int
) -> list[SampleRecord]:
    """Deterministically cap each class while preserving class balance."""
    by_class: dict[int, list[SampleRecord]] = defaultdict(list)
    for rec in records:
        by_class[rec[1]].append(rec)

    rng = random.Random(seed)
    subset: list[SampleRecord] = []
    for cls_idx in sorted(by_class):
        cls_items = sorted(by_class[cls_idx], key=lambda r: r[0].name)
        if len(cls_items) > per_class_limit:
            cls_items = rng.sample(cls_items, per_class_limit)
        subset.extend(cls_items)
    rng.shuffle(subset)
    return subset


def assert_disjoint_splits(splits: dict[str, list[SampleRecord]]) -> None:
    """Assert zero image-path overlap across train, val, and test splits."""
    train_set = {p.resolve() for p, _ in splits["train"]}
    val_set = {p.resolve() for p, _ in splits["val"]}
    test_set = {p.resolve() for p, _ in splits["test"]}

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set
    if tv or tt or vt:
        raise AssertionError(
            f"Data leakage detected across splits! Overlaps: "
            f"train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}"
        )


def load_splits(
    rgb_root: Path,
    splits_dir: Path = SPLITS_DIR,
    strategy: SplitStrategy = "official",
    max_per_class: int | None = None,
    seed: int = 42,
) -> dict[str, list[SampleRecord]]:
    """Load train/val/test splits and optionally subsample proportionally for CPU speed.

    When ``max_per_class`` is given, ``train`` is capped at ``max_per_class``
    images per class, while ``val`` and ``test`` are capped at
    ``max(1, max_per_class // 3)`` per class to mirror the 60/20/20 split ratio.
    """
    if strategy in {"official", "spatial"}:
        prefix = "eurosat" if strategy == "official" else "eurosat-spatial"
        needed = [splits_dir / f"{prefix}-{s}.txt" for s in ("train", "val", "test")]
        if not all(p.exists() for p in needed):
            download_official_splits(splits_dir)
        splits = {
            "train": _parse_split_file(splits_dir / f"{prefix}-train.txt", rgb_root),
            "val": _parse_split_file(splits_dir / f"{prefix}-val.txt", rgb_root),
            "test": _parse_split_file(splits_dir / f"{prefix}-test.txt", rgb_root),
        }
    elif strategy == "stratified":
        splits = _build_stratified_splits(rgb_root, seed=seed)
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    if max_per_class is not None:
        if max_per_class <= 0:
            raise ValueError("max_per_class must be a positive integer.")
        eval_cap = max(1, max_per_class // 3)
        splits = {
            "train": _subsample_stratified(splits["train"], max_per_class, seed=seed),
            "val": _subsample_stratified(splits["val"], eval_cap, seed=seed + 1),
            "test": _subsample_stratified(splits["test"], eval_cap, seed=seed + 2),
        }

    assert_disjoint_splits(splits)
    return splits


class EuroSATRGBDataset(Dataset):
    """PyTorch Dataset for EuroSAT RGB Sentinel-2 image patches."""

    def __init__(
        self,
        samples: list[SampleRecord],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, int]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
        if self.transform is not None:
            tensor = self.transform(rgb_img)
            return tensor, label
        return rgb_img, label


class EuroSATMultispectralDataset(Dataset):
    """PyTorch Dataset for 13-band Sentinel-2 GeoTIFF patches (EuroSAT_MS).

    Supports extending GeoVision beyond optical RGB into NIR/SWIR bands by
    loading 13-channel float32 arrays via ``tifffile`` and normalizing by a
    configurable reflectance scale factor (default 10000.0 for Sentinel-2 L1C).
    """

    def __init__(
        self,
        samples: list[SampleRecord],
        reflectance_scale: float = 10000.0,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.samples = list(samples)
        self.reflectance_scale = reflectance_scale
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        arr = tifffile.imread(path).astype(np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D multi-spectral TIF at {path}, got {arr.shape}")
        # tifffile reads (H, W, C); PyTorch Conv2d expects (C, H, W)
        if arr.shape[2] <= 16 and arr.shape[0] > 16:
            arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr) / self.reflectance_scale
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label


def create_dataloaders(
    splits: dict[str, list[SampleRecord]],
    train_transform: Callable,
    eval_transform: Callable,
    batch_size: int = 64,
    num_workers: int | None = None,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Create train, val, and test PyTorch DataLoaders safely across OSes.

    Windows uses ``spawn`` rather than ``fork`` for worker processes, which
    adds high startup overhead on CPU-only laptops; defaulting ``num_workers=0``
    on Windows keeps training fast and deterministic.
    """
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 2

    datasets = {
        "train": EuroSATRGBDataset(splits["train"], transform=train_transform),
        "val": EuroSATRGBDataset(splits["val"], transform=eval_transform),
        "test": EuroSATRGBDataset(splits["test"], transform=eval_transform),
    }
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
