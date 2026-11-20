"""Quantitative evaluation, calibration analysis, and model comparison tools.

Metrics & Diagnostic Plots Produced:
    1. Accuracy & Top-3 Accuracy: How often the true land cover is #1 or in the
       top 3 ranked predictions.
    2. Macro vs Weighted Precision, Recall & F1: EuroSAT has mild class
       imbalance (Pasture has 2,000 samples vs 3,000 for Forest/SeaLake).
       Macro-F1 weights every class equally so minority classes cannot hide.
    3. Confusion Matrix (counts + row-normalized): Reveals semantic overlap
       such as AnnualCrop <-> PermanentCrop or Highway <-> River.
    4. Multi-class One-vs-Rest ROC-AUC: Evaluates probability ranking quality
       across all classification thresholds.
    5. Expected Calibration Error (ECE) & Reliability Diagram: Measures whether
       predicted softmax confidence matches empirical accuracy (e.g., among
       predictions made at ~85% confidence, are ~85% actually correct?).
    6. Top-15 Confident Misclassifications: Concrete error analysis list for
       inspecting the exact failure modes of the model.

Usage:
    python -m src.evaluation.evaluate --checkpoint models/best_model.pt
    python -m src.evaluation.evaluate --compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from src.config import (
    BEST_MODEL_PATH,
    CLASS_DISPLAY_NAMES,
    DATA_RAW_DIR,
    EUROSAT_CLASSES,
    EVALUATION_OUTPUTS_DIR,
    SPLITS_DIR,
)
from src.data.dataset import EuroSATRGBDataset, load_splits
from src.data.download import locate_class_root
from src.data.preprocessing import build_transforms
from src.models.custom_cnn import GeoVisionCNN
from src.models.transfer import GeoVisionTransferNet

plt.rcParams.update(
    {
        "figure.facecolor": "#0b0f14",
        "axes.facecolor": "#111820",
        "axes.edgecolor": "#25313d",
        "axes.labelcolor": "#f2f5f7",
        "text.color": "#f2f5f7",
        "xtick.color": "#8b99a8",
        "ytick.color": "#8b99a8",
        "grid.color": "#25313d",
    }
)


def compute_ece(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, float | int]]]:
    """Compute 10-bin Expected Calibration Error (ECE) and per-bin stats.

    ECE = sum_{b=1}^{B} (|B_b| / N) * |acc(B_b) - conf(B_b)|
    Lower is better: 0.0 means predicted probabilities match true accuracy.
    """
    confidences = y_probs.max(axis=1)
    predictions = y_probs.argmax(axis=1)
    correctness = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_summary: list[dict[str, float | int]] = []
    n_total = len(y_true)

    for idx in range(n_bins):
        low, high = bin_edges[idx], bin_edges[idx + 1]
        if idx == 0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)

        count = int(mask.sum())
        if count > 0:
            avg_conf = float(confidences[mask].mean())
            avg_acc = float(correctness[mask].mean())
            ece += (count / n_total) * abs(avg_acc - avg_conf)
        else:
            avg_conf = float((low + high) / 2.0)
            avg_acc = 0.0

        bins_summary.append(
            {
                "bin_lower": round(float(low), 2),
                "bin_upper": round(float(high), 2),
                "count": count,
                "avg_confidence": round(avg_conf, 4),
                "avg_accuracy": round(avg_acc, 4),
            }
        )

    return float(ece), bins_summary


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    model_name: str,
    save_path: Path,
) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        np.where(row_sums == 0, 1, row_sums),
    )
    labels = [CLASS_DISPLAY_NAMES.get(c, c) for c in class_names]

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_norm[i, j] * 100:.1f}%\n({cm[i, j]})"

    fig, ax = plt.subplots(figsize=(11.5, 9.5), dpi=160)
    sns.heatmap(
        cm_norm * 100.0,
        annot=annot,
        fmt="",
        cmap="YlGnBu",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Row-normalized recall (%)"},
        ax=ax,
        linewidths=0.4,
        linecolor="#1c2733",
    )
    ax.set_title(
        f"{model_name} — Test Confusion Matrix (Row % & Counts)",
        fontsize=14,
        pad=14,
        fontweight="bold",
    )
    ax.set_xlabel("Predicted Land Cover", fontsize=11, labelpad=10)
    ax.set_ylabel("True Land Cover", fontsize=11, labelpad=10)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: Sequence[str],
    model_name: str,
    save_path: Path,
) -> dict[str, float]:
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    per_class_auc: dict[str, float] = {}

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=160)
    palette = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, cls_name in enumerate(class_names):
        if y_bin[:, i].sum() == 0 or y_bin[:, i].sum() == len(y_true):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        cls_auc = float(auc(fpr, tpr))
        per_class_auc[cls_name] = round(cls_auc, 5)
        disp = CLASS_DISPLAY_NAMES.get(cls_name, cls_name)
        ax.plot(
            fpr,
            tpr,
            lw=1.8,
            color=palette[i],
            label=f"{disp} (AUC={cls_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#8b99a8", lw=1.2, label="Random (0.500)")
    ax.set_title(f"{model_name} — One-vs-Rest ROC Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5, facecolor="#17212b", edgecolor="#25313d")
    plt.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return per_class_auc


def _plot_calibration(
    bins_summary: list[dict[str, float | int]],
    ece: float,
    model_name: str,
    save_path: Path,
) -> None:
    centers = [0.5 * (b["bin_lower"] + b["bin_upper"]) for b in bins_summary]
    accs = [b["avg_accuracy"] if b["count"] > 0 else 0.0 for b in bins_summary]
    counts = [b["count"] for b in bins_summary]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.5, 7.0), dpi=160, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.bar(
        centers,
        accs,
        width=0.085,
        color="#55d6a4",
        edgecolor="#08110d",
        alpha=0.88,
        label="Empirical Accuracy",
    )
    ax1.plot([0, 1], [0, 1], "--", color="#f2f5f7", lw=1.3, label="Perfect Calibration")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("Empirical Accuracy")
    ax1.set_title(
        f"{model_name} — Reliability Diagram (ECE = {ece:.4f})",
        fontsize=12.5,
        fontweight="bold",
    )
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left", facecolor="#17212b", edgecolor="#25313d")

    ax2.bar(centers, counts, width=0.085, color="#5ca4ea", alpha=0.85)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Predicted Confidence")
    ax2.set_ylabel("Samples")
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_probs: np.ndarray,
    class_names: Sequence[str],
    image_paths: Sequence[Path | str],
    model_name: str,
    output_dir: Path = EVALUATION_OUTPUTS_DIR,
) -> dict:
    """Compute comprehensive classification, ROC, calibration, and error metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    y_probs_arr = np.asarray(y_probs, dtype=np.float64)
    n_classes = len(class_names)
    all_labels = list(range(n_classes))

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    top3_acc = float(
        top_k_accuracy_score(
            y_true_arr, y_probs_arr, k=min(3, n_classes), labels=all_labels
        )
    )

    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=all_labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=all_labels)
    cm_path = output_dir / f"{model_name}_confusion_matrix.png"
    _plot_confusion_matrix(cm, class_names, model_name, cm_path)

    macro_roc_auc = float(
        roc_auc_score(
            y_true_arr,
            y_probs_arr,
            multi_class="ovr",
            average="macro",
            labels=all_labels,
        )
    )
    roc_path = output_dir / f"{model_name}_roc_curves.png"
    per_class_auc = _plot_roc_curves(
        y_true_arr, y_probs_arr, class_names, model_name, roc_path
    )

    ece, bins_summary = compute_ece(y_true_arr, y_probs_arr, n_bins=10)
    cal_path = output_dir / f"{model_name}_calibration.png"
    _plot_calibration(bins_summary, ece, model_name, cal_path)

    # Error analysis: top-15 most confident misclassifications
    mis_indices = np.where(y_pred_arr != y_true_arr)[0]
    mis_records: list[dict[str, object]] = []
    for idx in mis_indices:
        pred_idx = int(y_pred_arr[idx])
        true_idx = int(y_true_arr[idx])
        mis_records.append(
            {
                "file": Path(image_paths[idx]).name,
                "true_class": class_names[true_idx],
                "predicted_class": class_names[pred_idx],
                "confidence": round(float(y_probs_arr[idx, pred_idx]), 5),
                "true_class_prob": round(float(y_probs_arr[idx, true_idx]), 5),
            }
        )
    mis_records.sort(key=lambda r: float(r["confidence"]), reverse=True)
    top_misclassifications = mis_records[:15]

    # Top confusion pairs (off-diagonal)
    confusion_pairs: list[dict[str, object]] = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append(
                    {
                        "true_class": class_names[i],
                        "predicted_class": class_names[j],
                        "count": int(cm[i, j]),
                    }
                )
    confusion_pairs.sort(key=lambda p: int(p["count"]), reverse=True)

    per_class_metrics = {
        cls_name: {
            "precision": round(float(report[cls_name]["precision"]), 5),
            "recall": round(float(report[cls_name]["recall"]), 5),
            "f1_score": round(float(report[cls_name]["f1-score"]), 5),
            "support": int(report[cls_name]["support"]),
            "roc_auc": per_class_auc.get(cls_name),
        }
        for cls_name in class_names
    }

    metrics = {
        "model_name": model_name,
        "num_samples": int(len(y_true_arr)),
        "accuracy": round(acc, 5),
        "top3_accuracy": round(top3_acc, 5),
        "macro_precision": round(float(report["macro avg"]["precision"]), 5),
        "macro_recall": round(float(report["macro avg"]["recall"]), 5),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 5),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 5),
        "macro_roc_auc_ovr": round(macro_roc_auc, 5),
        "expected_calibration_error": round(ece, 5),
        "calibration_bins": bins_summary,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "top_confusion_pairs": confusion_pairs[:10],
        "top_confident_misclassifications": top_misclassifications,
        "artifacts": {
            "confusion_matrix_plot": str(cm_path),
            "roc_curves_plot": str(roc_path),
            "calibration_plot": str(cal_path),
        },
    }

    metrics_path = output_dir / f"{model_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    errors_path = output_dir / f"{model_name}_top_errors.json"
    errors_path.write_text(
        json.dumps(top_misclassifications, indent=2), encoding="utf-8"
    )

    print(
        f"[eval] {model_name}: Acc={acc * 100:.2f}% | Top-3={top3_acc * 100:.2f}% | "
        f"Macro-F1={metrics['macro_f1'] * 100:.2f}% | ROC-AUC={macro_roc_auc:.4f} | "
        f"ECE={ece:.4f}"
    )
    return metrics


def generate_comparison_report(
    eval_dir: Path = EVALUATION_OUTPUTS_DIR,
) -> dict[str, dict]:
    """Aggregate all ``*_metrics.json`` reports into a comparison chart & JSON."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    metric_files = sorted(eval_dir.glob("*_metrics.json"))
    if not metric_files:
        raise FileNotFoundError(f"No *_metrics.json files found in {eval_dir}")

    summary: dict[str, dict] = {}
    for mf in metric_files:
        data = json.loads(mf.read_text(encoding="utf-8"))
        summary[data["model_name"]] = {
            "num_samples": data["num_samples"],
            "accuracy": data["accuracy"],
            "top3_accuracy": data["top3_accuracy"],
            "macro_f1": data["macro_f1"],
            "weighted_f1": data["weighted_f1"],
            "macro_roc_auc_ovr": data["macro_roc_auc_ovr"],
            "expected_calibration_error": data["expected_calibration_error"],
        }

    comp_json = eval_dir / "model_comparison.json"
    comp_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_names = list(summary.keys())
    metric_keys = [
        ("accuracy", "Top-1 Accuracy"),
        ("top3_accuracy", "Top-3 Accuracy"),
        ("macro_f1", "Macro F1-Score"),
        ("macro_roc_auc_ovr", "Macro ROC-AUC"),
    ]
    x = np.arange(len(model_names))
    width = 0.18
    colors = ["#55d6a4", "#5ca4ea", "#f3b760", "#c789f4"]

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=160)
    for idx, (key, label) in enumerate(metric_keys):
        vals = [summary[m][key] * 100.0 for m in model_names]
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=colors[idx])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10.5)
    ax.set_ylim(0, 109)
    ax.set_ylabel("Score (%)")
    ax.set_title(
        "GeoVision Model Comparison: Baseline vs Custom CNN vs Transfer Learning",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", facecolor="#17212b", edgecolor="#25313d")
    plt.tight_layout()

    comp_png = eval_dir / "model_comparison.png"
    fig.savefig(comp_png, dpi=160)
    plt.close(fig)
    print(f"[compare] Saved {comp_json} and {comp_png}")
    return summary


def evaluate_checkpoint(
    checkpoint_path: Path,
    split_strategy: str = "official",
    max_per_class: int | None = None,
    batch_size: int = 64,
    raw_dir: Path = DATA_RAW_DIR,
    splits_dir: Path = SPLITS_DIR,
    output_dir: Path = EVALUATION_OUTPUTS_DIR,
) -> dict:
    """Load a saved PyTorch checkpoint and evaluate it on the test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_type = ckpt["model_type"]
    backbone = ckpt["backbone"]
    class_names = ckpt["class_names"]
    input_size = int(ckpt["input_size"])
    mean = ckpt["mean"]
    std = ckpt["std"]

    if model_type == "custom_cnn":
        model: torch.nn.Module = GeoVisionCNN(num_classes=len(class_names))
    elif model_type == "transfer":
        model = GeoVisionTransferNet(
            backbone=backbone, num_classes=len(class_names), pretrained=False
        )
    else:
        raise ValueError(f"Unsupported checkpoint model_type: {model_type}")

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    rgb_root = locate_class_root(raw_dir / "eurosat_rgb")
    splits = load_splits(
        rgb_root=rgb_root,
        splits_dir=splits_dir,
        strategy=split_strategy,  # type: ignore[arg-type]
        max_per_class=max_per_class,
    )
    test_records = splits["test"]
    transform = build_transforms(
        input_size=input_size, mean=mean, std=std, is_train=False
    )
    dataset = EuroSATRGBDataset(test_records, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(probs.argmax(axis=1))
            all_targets.append(targets.numpy())

    y_probs = np.vstack(all_probs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    image_paths = [p for p, _ in test_records]
    run_name = ckpt.get("run_name", checkpoint_path.stem)

    return evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        class_names=class_names,
        image_paths=image_paths,
        model_name=run_name,
        output_dir=output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a GeoVision checkpoint or compare evaluated models."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BEST_MODEL_PATH,
        help="Path to PyTorch model checkpoint (.pt).",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("official", "spatial", "stratified"),
        default="official",
    )
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Aggregate all *_metrics.json files into model_comparison.{json,png}.",
    )
    args = parser.parse_args()

    if args.compare:
        generate_comparison_report()
    else:
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            split_strategy=args.split_strategy,
            max_per_class=args.max_per_class,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
