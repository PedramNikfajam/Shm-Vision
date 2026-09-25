#!/usr/bin/env python3
"""
Model Evaluation Script for SHM Classification
==============================================

Comprehensive evaluation: top-1/top-5 accuracy, full confusion matrix,
per-class precision/recall/F1, calibration diagnostics (ECE), and speed
benchmark. All predictions are collected explicitly (no stubs), so every
reported metric is computed from the actual per-image predictions.

Usage:
    python scripts/evaluate.py --weights best.pt --data data/processed/test
    python scripts/evaluate.py --weights best.pt --data data/processed/test --benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")


def collect_predictions(
    model: YOLO,
    data_path: str,
    device: str = "cpu",
) -> tuple:
    """
    Run inference over a folder-structured (class-subfolder) dataset and
    collect ground-truth vs predicted labels.

    Args:
        model: Loaded YOLO classification model.
        data_path: Root directory containing one subfolder per class.
        device: Torch device specifier.

    Returns:
        (y_true, y_pred, y_prob, class_names):
            y_true/y_pred: integer label arrays aligned by index.
            y_prob: (N, C) softmax probability matrix.
            class_names: class-id -> name mapping from the model.

    Raises:
        FileNotFoundError: If no images are found under ``data_path``.
    """
    root = Path(data_path)
    image_paths: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(root.rglob(ext))
    image_paths = sorted(image_paths)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found under {data_path} "
            f"(expected <data_path>/<class_name>/<image>)"
        )

    # Ground truth from folder name; fall back to id if unmapped.
    name_to_id = {name: int(i) for i, name in model.names.items()}
    y_true: List[int] = []
    valid_paths: List[Path] = []
    for p in image_paths:
        cls_name = p.parent.name.lower()
        if cls_name in name_to_id:
            y_true.append(name_to_id[cls_name])
            valid_paths.append(p)
        else:
            logger.warning("Skipping %s: folder '%s' is not a known class", p, cls_name)

    logger.info("Evaluating %d images across %d classes...",
                len(valid_paths), len(model.names))

    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    for i, p in enumerate(valid_paths, 1):
        res = model.predict(str(p), device=device, verbose=False)[0]
        y_prob.append(np.asarray(res.probs.data.cpu(), dtype=np.float64))
        y_pred.append(int(res.probs.top1))
        if i % 100 == 0:
            logger.info("  %d/%d inferred", i, len(valid_paths))

    return (
        np.array(y_true),
        np.array(y_pred),
        np.stack(y_prob),
        dict(model.names),
    )


def expected_calibration_error(
    y_prob: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> float:
    """
    Expected Calibration Error: |confidence - accuracy| weighted by bin mass.

    A well-calibrated model's predicted confidence should match its empirical
    accuracy -- essential when confidence drives UNCERTAIN flagging downstream.

    Args:
        y_prob: (N, C) probability matrix.
        y_true: (N,) integer labels.
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE in [0, 1] (lower is better; 0 = perfectly calibrated).
    """
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    ece = 0.0
    for lo in np.linspace(0.0, 1.0, n_bins + 1)[:-1]:
        hi = lo + 1.0 / n_bins
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def evaluate_model(
    weights_path: str,
    data_path: str,
    output_dir: str = "runs/evaluation",
    device: str = "cpu",
    benchmark_runs: int = 0,
) -> Dict:
    """
    Full evaluation pipeline: metrics, confusion matrix, per-class report,
    calibration, and optional speed benchmark.

    Args:
        weights_path: Path to trained weights.
        data_path: Dataset root (class-subfolder layout).
        output_dir: Where artifacts (JSON/CSV/PNG) are written.
        device: 'cpu' or GPU id.
        benchmark_runs: If > 0, run a speed benchmark with this many runs.

    Returns:
        Metrics dict (also written to ``evaluation_results.json``).
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", weights_path)
    model = YOLO(weights_path)

    y_true, y_pred, y_prob, class_names = collect_predictions(model, data_path, device)

    top1 = float(accuracy_score(y_true, y_pred))
    top5 = float(np.mean([
        gt in np.argsort(-(row))[:5] for gt, row in zip(y_true, y_prob)
    ])) if y_prob.shape[1] >= 5 else None

    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info("Top-1 Accuracy: %.4f", top1)
    if top5 is not None:
        logger.info("Top-5 Accuracy: %.4f", top5)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Confusion matrix (raw counts + row-normalized)
    # ------------------------------------------------------------------
    labels = sorted(class_names.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    # ------------------------------------------------------------------
    # Per-class metrics
    # ------------------------------------------------------------------
    target_names = [class_names[i] for i in labels]
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    logger.info("\n%s", classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
    ))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    ece = expected_calibration_error(y_prob, y_true)
    logger.info("Expected Calibration Error: %.4f", ece)

    results = {
        "model": weights_path,
        "dataset": data_path,
        "n_images": int(len(y_true)),
        "n_classes": int(len(labels)),
        "top1_accuracy": round(top1, 4),
        "top5_accuracy": round(top5, 4) if top5 is not None else None,
        "expected_calibration_error": round(ece, 4),
        "macro_avg": {
            "precision": report_dict["macro avg"]["precision"],
            "recall": report_dict["macro avg"]["recall"],
            "f1-score": report_dict["macro avg"]["f1-score"],
        },
        "weighted_avg": {
            "precision": report_dict["weighted avg"]["precision"],
            "recall": report_dict["weighted avg"]["recall"],
            "f1-score": report_dict["weighted avg"]["f1-score"],
        },
        "per_class": {
            name: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1-score": round(float(f), 4),
                "support": int(s),
            }
            for name, p, r, f, s in zip(target_names, precision, recall, f1, support)
        },
        "confusion_matrix": {
            "labels": target_names,
            "counts": cm.tolist(),
            "row_normalized": np.round(cm_norm, 4).tolist(),
        },
    }

    # ------------------------------------------------------------------
    # Optional speed benchmark
    # ------------------------------------------------------------------
    if benchmark_runs > 0:
        sample = str(next(
            p for ext in IMAGE_EXTENSIONS for p in Path(data_path).rglob(ext)
        ))
        for _ in range(10):  # warmup
            model.predict(sample, device=device, verbose=False)
        times = []
        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            model.predict(sample, device=device, verbose=False)
            times.append(time.perf_counter() - t0)
        results["speed_benchmark"] = {
            "average_ms": round(float(np.mean(times)) * 1000, 2),
            "std_ms": round(float(np.std(times)) * 1000, 2),
            "throughput_fps": round(1.0 / float(np.mean(times)), 2),
            "runs": benchmark_runs,
        }
        logger.info(
            "Speed: %.2f ± %.2f ms (%.2f img/s)",
            results["speed_benchmark"]["average_ms"],
            results["speed_benchmark"]["std_ms"],
            results["speed_benchmark"]["throughput_fps"],
        )

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    with open(out_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Per-class CSV for spreadsheets.
    with open(out_dir / "per_class_metrics.csv", "w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1,support\n")
        for name in target_names:
            pc = results["per_class"][name]
            f.write(
                f"{name},{pc['precision']},{pc['recall']},{pc['f1-score']},{pc['support']}\n"
            )

    # Confusion matrix plot (matplotlib available with ultralytics).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(target_names)))
        ax.set_yticks(range(len(target_names)))
        ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(target_names, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}",
                        ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (top-1 = {top1:.1%})")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        logger.info("Saved confusion_matrix.png")
    except Exception as exc:
        logger.warning("Matplotlib plot skipped: %s", exc)

    logger.info("Results saved to %s", out_dir / "evaluation_results.json")
    return results


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate SHM Classification Model")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default="data/processed/test",
                        help="Dataset root with class subfolders")
    parser.add_argument("--output", type=str, default="runs/evaluation",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device ('cpu' or GPU id)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference speed benchmark")
    parser.add_argument("--runs", type=int, default=100,
                        help="Benchmark runs (with --benchmark)")
    args = parser.parse_args()

    evaluate_model(
        weights_path=args.weights,
        data_path=args.data,
        output_dir=args.output,
        device=args.device,
        benchmark_runs=args.runs if args.benchmark else 0,
    )


if __name__ == "__main__":
    main()
