"""Download and verify the EuroSAT Sentinel-2 RGB dataset and official splits.

Why this module exists:
    Historically, ``torchvision.datasets.EuroSAT(download=True)`` pointed to
    ``madm.dfki.de``, which frequently returns SSL/timeout errors. The dataset
    authors officially archived EuroSAT on Zenodo (DOI 10.5281/zenodo.7711810)
    with a verified MD5 checksum, while TorchGeo hosts reproducible random and
    spatial train/val/test split manifests on HuggingFace.

Usage:
    python -m src.data.download
    python -m src.data.download --splits-only
    python -m src.data.download --verify
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from src.config import (
    DATA_RAW_DIR,
    EUROSAT_CLASSES,
    EUROSAT_RGB_HF_FALLBACK_URL,
    EUROSAT_RGB_MD5,
    EUROSAT_RGB_ZENODO_URL,
    OFFICIAL_SPLIT_FILES,
    SPLITS_DIR,
    TORCHGEO_SPLITS_BASE_URL,
)


def compute_md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex MD5 digest of a file without loading it entirely into RAM."""
    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _stream_download(url: str, dest_path: Path, desc: str) -> None:
    """Stream an HTTP download to a temporary file then atomically rename."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    req = urllib.request.Request(
        url, headers={"User-Agent": "GeoVision-DatasetDownloader/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))
        with (
            tmp_path.open("wb") as out_file,
            tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
            ) as pbar,
        ):
            while chunk := response.read(64 * 1024):
                out_file.write(chunk)
                pbar.update(len(chunk))
    tmp_path.replace(dest_path)


def locate_class_root(base_dir: Path) -> Path:
    """Locate the folder containing all 10 EuroSAT class subdirectories.

    Zip archives from different mirrors unpack to slightly different nesting
    depths (e.g. ``EuroSAT_RGB/<class>`` vs ``2750/<class>``). Checking the
    base directory and up to two directory levels deep makes loading immune to
    mirror packaging quirks.
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {base_dir}")

    expected = set(EUROSAT_CLASSES)
    candidates: list[Path] = [base_dir]
    for child in base_dir.iterdir():
        if child.is_dir():
            candidates.append(child)
            for grandchild in child.iterdir():
                if grandchild.is_dir():
                    candidates.append(grandchild)

    for candidate in candidates:
        subdirs = {p.name for p in candidate.iterdir() if p.is_dir()}
        if expected.issubset(subdirs):
            return candidate

    raise FileNotFoundError(
        f"Could not find all 10 EuroSAT class folders inside {base_dir}. "
        f"Expected: {sorted(expected)}"
    )


def count_images_per_class(class_root: Path) -> dict[str, int]:
    """Count image files per EuroSAT class folder."""
    counts: dict[str, int] = {}
    for cls_name in EUROSAT_CLASSES:
        cls_dir = class_root / cls_name
        files = [
            p
            for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        ]
        counts[cls_name] = len(files)
    return counts


def download_eurosat_rgb(
    raw_dir: Path = DATA_RAW_DIR, force: bool = False
) -> Path:
    """Download and extract EuroSAT_RGB.zip from Zenodo (with HF fallback).

    Returns:
        Path to the directory containing the 10 class subfolders.
    """
    extract_dir = raw_dir / "eurosat_rgb"
    if extract_dir.exists() and not force:
        try:
            class_root = locate_class_root(extract_dir)
            print(f"[download] EuroSAT RGB already present at {class_root}")
            return class_root
        except FileNotFoundError:
            pass

    zip_path = raw_dir / "EuroSAT_RGB.zip"
    need_download = force or not zip_path.exists()
    if not need_download:
        existing_md5 = compute_md5(zip_path)
        if existing_md5 != EUROSAT_RGB_MD5:
            print(
                f"[download] Existing zip MD5 {existing_md5} != {EUROSAT_RGB_MD5}; re-downloading."
            )
            need_download = True

    if need_download:
        try:
            print(f"[download] Fetching EuroSAT_RGB.zip from Zenodo...")
            _stream_download(EUROSAT_RGB_ZENODO_URL, zip_path, desc="EuroSAT_RGB.zip")
            actual_md5 = compute_md5(zip_path)
            if actual_md5 != EUROSAT_RGB_MD5:
                raise ValueError(
                    f"Checksum mismatch for Zenodo archive: got {actual_md5}, expected {EUROSAT_RGB_MD5}"
                )
            print(f"[download] Verified MD5 checksum: {actual_md5}")
        except Exception as exc:
            print(
                f"[download] Zenodo download failed ({exc}); falling back to HuggingFace mirror..."
            )
            _stream_download(
                EUROSAT_RGB_HF_FALLBACK_URL, zip_path, desc="EuroSAT.zip (HF)"
            )

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] Extracting {zip_path.name} -> {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    class_root = locate_class_root(extract_dir)
    print(f"[download] Extraction complete. Class root: {class_root}")
    return class_root


def download_official_splits(
    splits_dir: Path = SPLITS_DIR, force: bool = False
) -> Path:
    """Download TorchGeo's reproducible random and spatial EuroSAT splits."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    for filename in OFFICIAL_SPLIT_FILES:
        dest = splits_dir / filename
        if dest.exists() and not force and dest.stat().st_size > 0:
            continue
        url = f"{TORCHGEO_SPLITS_BASE_URL}/{filename}"
        _stream_download(url, dest, desc=filename)
    print(f"[download] Official split files ready in {splits_dir}")
    return splits_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download EuroSAT Sentinel-2 RGB imagery and official splits."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DATA_RAW_DIR,
        help="Directory to store raw downloaded archives and extracted images.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_DIR,
        help="Directory to store official train/val/test split text files.",
    )
    parser.add_argument(
        "--splits-only",
        action="store_true",
        help="Only download the split manifest files without downloading imagery.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist locally.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify extracted class counts without re-downloading.",
    )
    args = parser.parse_args()

    if args.verify:
        class_root = locate_class_root(args.raw_dir / "eurosat_rgb")
    elif args.splits_only:
        download_official_splits(args.splits_dir, force=args.force)
        return
    else:
        class_root = download_eurosat_rgb(args.raw_dir, force=args.force)
        download_official_splits(args.splits_dir, force=args.force)

    counts = count_images_per_class(class_root)
    total = sum(counts.values())
    print("\nEuroSAT Class Distribution:")
    for cls_name, count in counts.items():
        print(f"  {cls_name:<22} {count:>5}")
    print(f"  {'TOTAL':<22} {total:>5}")


if __name__ == "__main__":
    main()
