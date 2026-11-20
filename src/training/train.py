"""Unified training CLI for Baseline, Custom CNN, and Transfer Learning models.

ML Concepts Implemented & Annotated Here:
    1. Reproducible Seeding: Fixes Python, NumPy, and PyTorch RNG states so
       split subsampling, weight initialization, and batch shuffling repeat.
    2. Cross-Entropy with Label Smoothing: Prevents overconfident logits by
       softening one-hot targets (e.g., 0.955 on true class, 0.005 elsewhere),
       regularizing the final linear layer and improving calibration.
    3. AdamW + Cosine Annealing LR: AdamW decouples L2 weight decay from the
       adaptive gradient moments, while CosineAnnealingLR smoothly lowers the
       learning rate down toward zero so weights settle into wide minima.
    4. Gradient Clipping (max_norm=1.0): Prevents rare high-gradient batches
       from destabilizing optimization.
    5. Self-Contained Checkpoints: Every saved ``.pt`` bundle embeds the exact
       ``input_size``, ``mean``, ``std``, and ``class_names`` used during
       training, eliminating train/serve skew at inference time.

Usage Examples:
    python -m src.training.train --model baseline --split-strategy official
    python -m src.training.train --model custom_cnn --epochs 10 --max-per-class 400
    python -m src.training.train --model transfer --backbone resnet18 --set-best
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BEST_MODEL_PATH,
    DATA_RAW_DIR,
    EUROSAT_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODELS_DIR,
    SPLITS_DIR,
    TRAINING_OUTPUTS_DIR,
)
from src.data.dataset import create_dataloaders, load_splits
from src.data.download import locate_class_root
from src.data.preprocessing import build_transforms, compute_channel_stats
from src.evaluation.evaluate import evaluate_predictions
from src.models.baseline import BaselineClassifier, extract_feature_matrix
from src.models.custom_cnn import GeoVisionCNN
from src.models.transfer import SUPPORTED_BACKBONES, GeoVisionTransferNet


def set_seed(seed: int = 42) -> None:
    """Fix RNG seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "Epoch",
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Run one training or evaluation epoch and return loss, acc, y_true, y_pred, y_probs."""
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    probs_list: list[np.ndarray] = []
    preds_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    ctx = torch.enable_grad() if is_train else torch.inference_mode()
    with ctx:
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            targets = targets.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, targets)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            probs = torch.softmax(logits.detach(), dim=1)
            preds = probs.argmax(dim=1)
            total_correct += int((preds == targets).sum().item())
            total_samples += batch_size

            probs_list.append(probs.cpu().numpy())
            preds_list.append(preds.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return (
        avg_loss,
        avg_acc,
        np.concatenate(targets_list),
        np.concatenate(preds_list),
        np.vstack(probs_list),
    )


def _plot_learning_curves(
    history: list[dict[str, float | int | str]],
    run_name: str,
    save_path: Path,
) -> None:
    """Plot 2-panel training/validation loss and accuracy curves."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(h["epoch"]) for h in history]
    tr_loss = [float(h["train_loss"]) for h in history]
    va_loss = [float(h["val_loss"]) for h in history]
    tr_acc = [float(h["train_acc"]) * 100.0 for h in history]
    va_acc = [float(h["val_acc"]) * 100.0 for h in history]

    plt.rcParams.update(
        {
            "figure.facecolor": "#0b0f14",
            "axes.facecolor": "#111820",
            "axes.edgecolor": "#25313d",
            "axes.labelcolor": "#f2f5f7",
            "text.color": "#f2f5f7",
            "xtick.color": "#8b99a8",
            "ytick.color": "#8b99a8",
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160)

    ax1.plot(epochs, tr_loss, "-o", color="#5ca4ea", label="Train Loss", markersize=4)
    ax1.plot(epochs, va_loss, "-o", color="#55d6a4", label="Val Loss", markersize=4)
    ax1.set_title(f"{run_name} — Cross-Entropy Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.25)
    ax1.legend(facecolor="#17212b", edgecolor="#25313d")

    ax2.plot(epochs, tr_acc, "-o", color="#5ca4ea", label="Train Acc", markersize=4)
    ax2.plot(epochs, va_acc, "-o", color="#55d6a4", label="Val Acc", markersize=4)
    ax2.set_title(f"{run_name} — Classification Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 102)
    ax2.grid(alpha=0.25)
    ax2.legend(facecolor="#17212b", edgecolor="#25313d")

    plt.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def train_baseline(args: argparse.Namespace) -> dict:
    """Train and evaluate the classical color+edge LogisticRegression baseline."""
    rgb_root = locate_class_root(args.raw_dir / "eurosat_rgb")
    splits = load_splits(
        rgb_root=rgb_root,
        splits_dir=args.splits_dir,
        strategy=args.split_strategy,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    print(
        f"[baseline] Split sizes ({args.split_strategy}): "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )

    t0 = time.perf_counter()
    X_train = extract_feature_matrix(
        [p for p, _ in splits["train"]], desc="Baseline train features"
    )
    y_train = np.array([y for _, y in splits["train"]], dtype=np.int64)

    X_val = extract_feature_matrix(
        [p for p, _ in splits["val"]], desc="Baseline val features"
    )
    y_val = np.array([y for _, y in splits["val"]], dtype=np.int64)

    X_test = extract_feature_matrix(
        [p for p, _ in splits["test"]], desc="Baseline test features"
    )
    y_test = np.array([y for _, y in splits["test"]], dtype=np.int64)

    clf = BaselineClassifier(algorithm="logreg", random_state=args.seed)
    clf.fit(X_train, y_train)
    val_acc = float((clf.predict(X_val) == y_val).mean())
    elapsed = time.perf_counter() - t0
    print(f"[baseline] Trained in {elapsed:.1f}s | Val Acc = {val_acc * 100:.2f}%")

    test_probs = clf.predict_proba(X_test)
    test_preds = test_probs.argmax(axis=1)

    run_name = "baseline_logreg"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{run_name}.joblib"

    metrics = evaluate_predictions(
        y_true=y_test,
        y_pred=test_preds,
        y_probs=test_probs,
        class_names=EUROSAT_CLASSES,
        image_paths=[p for p, _ in splits["test"]],
        model_name=run_name,
    )
    clf.save(
        model_path,
        metadata={
            "run_name": run_name,
            "val_accuracy": val_acc,
            "test_metrics": {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_roc_auc_ovr": metrics["macro_roc_auc_ovr"],
            },
            "split_strategy": args.split_strategy,
            "train_time_s": round(elapsed, 2),
        },
    )
    print(f"[baseline] Saved model -> {model_path}")
    return metrics


def train_deep_model(args: argparse.Namespace) -> dict:
    """Train and evaluate Custom CNN or two-stage Transfer Learning model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_root = locate_class_root(args.raw_dir / "eurosat_rgb")
    splits = load_splits(
        rgb_root=rgb_root,
        splits_dir=args.splits_dir,
        strategy=args.split_strategy,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

    if args.model == "custom_cnn":
        input_size = args.input_size or 64
        run_name = "custom_cnn"
        backbone_label = "GeoVisionCNN"
        # Compute channel statistics strictly on the training split (no leakage)
        train_paths = [p for p, _ in splits["train"]]
        mean, std = compute_channel_stats(train_paths, seed=args.seed)
        model: nn.Module = GeoVisionCNN(num_classes=len(EUROSAT_CLASSES))
    else:
        input_size = args.input_size or 128
        backbone_label = args.backbone
        run_name = f"transfer_{args.backbone}"
        mean, std = list(IMAGENET_MEAN), list(IMAGENET_STD)
        model = GeoVisionTransferNet(
            backbone=args.backbone,
            num_classes=len(EUROSAT_CLASSES),
            pretrained=True,
        )

    model.to(device)
    print(
        f"[train] Model={run_name} | Device={device} | Input={input_size}x{input_size} | "
        f"Mean={mean} | Std={std} | Params={model.count_parameters():,}"
    )
    print(
        f"[train] Split sizes ({args.split_strategy}): "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )

    train_tf = build_transforms(input_size=input_size, mean=mean, std=std, is_train=True)
    eval_tf = build_transforms(input_size=input_size, mean=mean, std=std, is_train=False)
    loaders = create_dataloaders(
        splits=splits,
        train_transform=train_tf,
        eval_transform=eval_tf,
        batch_size=args.batch_size,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history: list[dict[str, float | int | str]] = []
    best_val_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    total_train_start = time.perf_counter()

    def run_stage(
        stage_name: str,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch_offset: int,
    ) -> None:
        nonlocal best_val_acc, best_state, best_epoch
        for ep in range(1, epochs + 1):
            global_ep = epoch_offset + ep
            ep_t0 = time.perf_counter()
            tr_loss, tr_acc, _, _, _ = _run_epoch(
                model,
                loaders["train"],
                criterion,
                device,
                optimizer=optimizer,
                desc=f"{stage_name} Ep {ep}/{epochs} [train]",
            )
            va_loss, va_acc, _, _, _ = _run_epoch(
                model,
                loaders["val"],
                criterion,
                device,
                optimizer=None,
                desc=f"{stage_name} Ep {ep}/{epochs} [val]",
            )
            scheduler.step()
            ep_sec = time.perf_counter() - ep_t0

            marker = ""
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_epoch = global_ep
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                marker = " *"

            print(
                f"  Epoch {global_ep:02d} ({stage_name:<12}) | "
                f"Train Loss={tr_loss:.4f} Acc={tr_acc * 100:.2f}% | "
                f"Val Loss={va_loss:.4f} Acc={va_acc * 100:.2f}% | "
                f"{ep_sec:.1f}s{marker}"
            )
            history.append(
                {
                    "epoch": global_ep,
                    "stage": stage_name,
                    "train_loss": round(tr_loss, 5),
                    "train_acc": round(tr_acc, 5),
                    "val_loss": round(va_loss, 5),
                    "val_acc": round(va_acc, 5),
                    "epoch_seconds": round(ep_sec, 2),
                }
            )

    if args.model == "custom_cnn":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )
        run_stage("scratch", args.epochs, optimizer, scheduler, epoch_offset=0)
    else:
        assert isinstance(model, GeoVisionTransferNet)
        # Stage 1: Linear probing (frozen backbone, train classifier head only)
        if args.stage1_epochs > 0:
            model.freeze_backbone()
            print(
                f"[stage 1] Frozen backbone — trainable params: "
                f"{model.count_parameters(trainable_only=True):,}"
            )
            opt1 = torch.optim.AdamW(
                model.get_param_groups(backbone_lr=0.0, head_lr=args.lr),
                weight_decay=args.weight_decay,
            )
            sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt1, T_max=max(1, args.stage1_epochs)
            )
            run_stage("stage1_head", args.stage1_epochs, opt1, sched1, epoch_offset=0)

        # Stage 2: Selective fine-tuning (unfreeze top semantic block + differential LR)
        if args.stage2_epochs > 0:
            model.unfreeze_top_layers()
            print(
                f"[stage 2] Unfroze top layers — trainable params: "
                f"{model.count_parameters(trainable_only=True):,}"
            )
            opt2 = torch.optim.AdamW(
                model.get_param_groups(
                    backbone_lr=args.backbone_lr, head_lr=args.lr * 0.5
                ),
                weight_decay=args.weight_decay,
            )
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt2, T_max=max(1, args.stage2_epochs)
            )
            run_stage(
                "stage2_tune",
                args.stage2_epochs,
                opt2,
                sched2,
                epoch_offset=args.stage1_epochs,
            )

    total_train_time = time.perf_counter() - total_train_start

    # Restore best validation checkpoint before held-out test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_acc, y_true, y_pred, y_probs = _run_epoch(
        model, loaders["test"], criterion, device, optimizer=None, desc="Test evaluation"
    )
    metrics = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        class_names=EUROSAT_CLASSES,
        image_paths=[p for p, _ in splits["test"]],
        model_name=run_name,
    )

    # Save training history & learning curves
    TRAINING_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = TRAINING_OUTPUTS_DIR / f"{run_name}_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    curves_path = TRAINING_OUTPUTS_DIR / f"{run_name}_curves.png"
    _plot_learning_curves(history, run_name, curves_path)

    # Save self-contained PyTorch checkpoint
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / f"{run_name}.pt"
    checkpoint_payload = {
        "run_name": run_name,
        "model_type": args.model,
        "backbone": backbone_label,
        "num_classes": len(EUROSAT_CLASSES),
        "class_names": list(EUROSAT_CLASSES),
        "input_size": input_size,
        "mean": list(mean),
        "std": list(std),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "metrics": {
            "best_val_accuracy": round(best_val_acc, 5),
            "best_epoch": best_epoch,
            "test_accuracy": metrics["accuracy"],
            "test_top3_accuracy": metrics["top3_accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "test_macro_roc_auc": metrics["macro_roc_auc_ovr"],
            "test_ece": metrics["expected_calibration_error"],
        },
        "training_config": {
            "split_strategy": args.split_strategy,
            "max_per_class": args.max_per_class,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "backbone_lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "seed": args.seed,
            "total_train_seconds": round(total_train_time, 2),
        },
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint_payload, ckpt_path)
    print(
        f"[train] Saved checkpoint -> {ckpt_path} (best val={best_val_acc * 100:.2f}% @ ep {best_epoch}, "
        f"test={test_acc * 100:.2f}%)"
    )

    if args.set_best:
        shutil.copy2(ckpt_path, BEST_MODEL_PATH)
        print(f"[train] Promoted checkpoint -> {BEST_MODEL_PATH}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GeoVision Baseline, Custom CNN, or Transfer Learning model."
    )
    parser.add_argument(
        "--model",
        choices=("baseline", "custom_cnn", "transfer"),
        required=True,
    )
    parser.add_argument(
        "--backbone",
        choices=SUPPORTED_BACKBONES,
        default="resnet18",
        help="Pre-trained backbone when --model transfer is selected.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("official", "spatial", "stratified"),
        default="official",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Input resolution (defaults to 64 for custom_cnn, 128 for transfer).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional per-class cap in train split (val/test capped at //3).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--stage1-epochs", type=int, default=3)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DATA_RAW_DIR,
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_DIR,
    )
    parser.add_argument(
        "--set-best",
        action="store_true",
        help="Also copy the trained model checkpoint to models/best_model.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.model == "baseline":
        train_baseline(args)
    else:
        train_deep_model(args)


if __name__ == "__main__":
    main()
