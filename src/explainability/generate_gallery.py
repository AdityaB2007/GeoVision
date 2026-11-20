"""CLI to generate a 10-class Grad-CAM attention gallery on held-out EuroSAT patches.

Usage:
    python -m src.explainability.generate_gallery --checkpoint models/best_model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.config import (
    BEST_MODEL_PATH,
    CLASS_DISPLAY_NAMES,
    DATA_RAW_DIR,
    EUROSAT_CLASSES,
    OUTPUTS_DIR,
    SPLITS_DIR,
)
from src.data.dataset import load_splits
from src.data.download import locate_class_root
from src.data.preprocessing import build_transforms
from src.explainability.gradcam import GradCAM
from src.explainability.visualization import (
    SUPPORTED_COLORMAPS,
    overlay_heatmap,
    save_gradcam_gallery,
)
from src.models.custom_cnn import GeoVisionCNN
from src.models.transfer import GeoVisionTransferNet


def generate_gradcam_gallery(
    checkpoint_path: Path = BEST_MODEL_PATH,
    split_strategy: str = "official",
    colormap: str = "turbo",
    alpha: float = 0.45,
    raw_dir: Path = DATA_RAW_DIR,
    splits_dir: Path = SPLITS_DIR,
    output_dir: Path = OUTPUTS_DIR / "explainability",
) -> tuple[Path, Path]:
    """Generate a 10-class Grad-CAM visual gallery and sample JSON metadata."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_names = list(ckpt.get("class_names", EUROSAT_CLASSES))
    model_type = str(ckpt["model_type"])
    backbone = str(ckpt.get("backbone", model_type))
    input_size = int(ckpt["input_size"])
    mean = [float(x) for x in ckpt["mean"]]
    std = [float(x) for x in ckpt["std"]]

    if model_type == "custom_cnn":
        model: torch.nn.Module = GeoVisionCNN(num_classes=len(class_names))
    elif model_type == "transfer":
        model = GeoVisionTransferNet(
            backbone=backbone,  # type: ignore[arg-type]
            num_classes=len(class_names),
            pretrained=False,
        )
    else:
        raise ValueError(f"Grad-CAM does not support model_type={model_type}")

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    transform = build_transforms(
        input_size=input_size, mean=mean, std=std, is_train=False
    )
    cam_engine = GradCAM(model)

    rgb_root = locate_class_root(raw_dir / "eurosat_rgb")
    splits = load_splits(
        rgb_root=rgb_root,
        splits_dir=splits_dir,
        strategy=split_strategy,  # type: ignore[arg-type]
    )
    test_records = splits["test"]

    # Pick the first sorted test image per EuroSAT class for deterministic galleries
    chosen_by_class: dict[int, Path] = {}
    for img_path, label_idx in sorted(test_records, key=lambda r: (r[1], r[0].name)):
        if label_idx not in chosen_by_class:
            chosen_by_class[label_idx] = img_path
        if len(chosen_by_class) == len(class_names):
            break

    gallery_items: list[dict] = []
    sample_records: list[dict] = []

    for cls_idx in range(len(class_names)):
        img_path = chosen_by_class[cls_idx]
        cls_name = class_names[cls_idx]
        with Image.open(img_path) as raw_img:
            rgb_img = raw_img.convert("RGB")

        tensor = transform(rgb_img).unsqueeze(0).to(device)
        cam_res = cam_engine.generate(tensor, target_class=None)
        pred_cls = class_names[cam_res.predicted_idx]
        conf = float(cam_res.probabilities[cam_res.predicted_idx])

        orig_resized = rgb_img.resize((256, 256), resample=Image.Resampling.BICUBIC)
        overlay_img = overlay_heatmap(
            rgb_img,
            cam_res.heatmap,
            alpha=alpha,
            size=(256, 256),
            colormap=colormap,
        )

        gallery_items.append(
            {
                "class_name": cls_name,
                "display_name": CLASS_DISPLAY_NAMES.get(cls_name, cls_name),
                "predicted_class": pred_cls,
                "predicted_display": CLASS_DISPLAY_NAMES.get(pred_cls, pred_cls),
                "confidence": conf,
                "correct": cam_res.predicted_idx == cls_idx,
                "original_image": orig_resized,
                "overlay_image": overlay_img,
            }
        )
        sample_records.append(
            {
                "file": img_path.name,
                "true_class": cls_name,
                "predicted_class": pred_cls,
                "confidence": round(conf, 5),
                "correct": bool(cam_res.predicted_idx == cls_idx),
                "target_layer": cam_res.target_layer_name,
                "heatmap_shape": list(cam_res.heatmap.shape),
                "colormap": colormap,
                "alpha": alpha,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = output_dir / "gradcam_gallery.png"
    save_gradcam_gallery(gallery_items, gallery_path)

    samples_path = output_dir / "gradcam_samples.json"
    samples_path.write_text(json.dumps(sample_records, indent=2), encoding="utf-8")

    print(
        f"[gradcam] Saved 10-class gallery -> {gallery_path} "
        f"({sum(1 for r in sample_records if r['correct'])}/{len(sample_records)} correct)"
    )
    return gallery_path, samples_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 10-class EuroSAT Grad-CAM attention gallery."
    )
    parser.add_argument("--checkpoint", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument(
        "--split-strategy",
        choices=("official", "spatial", "stratified"),
        default="official",
    )
    parser.add_argument(
        "--colormap",
        choices=SUPPORTED_COLORMAPS,
        default="turbo",
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUTS_DIR / "explainability"
    )
    args = parser.parse_args()

    generate_gradcam_gallery(
        checkpoint_path=args.checkpoint,
        split_strategy=args.split_strategy,
        colormap=args.colormap,
        alpha=args.alpha,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
