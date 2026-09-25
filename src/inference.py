#!/usr/bin/env python3
"""
Structural Health Monitoring - Classification Inference & Analysis Pipeline
===========================================================================

Classifies concrete structure images (YOLOv8) and enriches every prediction
with a structural engineering assessment (severity, safety factor,
recommendation, optional physical crack measurement).

Usage:
    # Single image
    python src/inference.py --image data/raw/deck/cracked/sample.jpg \
        --weights runs/classify/shm_classification/weights/best.pt

    # Batch directory (recursive) + consolidated reports
    python src/inference.py --source data/raw/ --weights best.pt --save-report

    # Video analysis (every 5th frame, no temp files: frames passed in-memory)
    python src/inference.py --video inspection.mp4 --weights best.pt

    # Real-time webcam / RTSP (press 'q' to quit)
    python src/inference.py --stream 0 --weights best.pt

    # Enable physical crack measurement (photogrammetry)
    python src/inference.py --image sample.jpg --weights best.pt \
        --distance-mm 2000 --focal-px 800
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

# Make sibling modules importable whether run as script or package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = Path(__file__).resolve().parent
for _p in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    CLASS_NAMES,
    CLASS_COLORS,
    CONDITION_MAP,
    OTHER_CLASS,
    DamageSeverity,
    STRUCTURE_MAP,
    assess_severity_from_confidence,
    assess_severity_from_width,
    can_escalate_severity,
    get_safety_factor,
    get_structure_thresholds,
    get_recommendation,
)
from crack_measurement import estimate_crack_width

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")


@dataclass
class ClassificationResult:
    """One classification outcome enriched with engineering assessment."""

    image_path: str
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    structure_type: str
    condition: str
    severity: Optional[DamageSeverity]
    recommendation: str
    is_damaged: bool
    safety_factor: float
    measurement: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """JSON-safe dict (enum -> string)."""
        d = asdict(self)
        d["severity"] = self.severity.value if self.severity else None
        return d


class StructuralClassifier:
    """
    YOLOv8 classification + structural assessment pipeline.

    Wraps an Ultralytics classification model and attaches engineering
    context (structure type, condition, severity, safety factor, optional
    physical crack measurement) to every prediction.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.75,
        device: str = "auto",
    ):
        """
        Args:
            model_path: Path to trained YOLO classification weights (.pt).
            confidence_threshold: Below this the verdict is UNCERTAIN.
            device: ``'auto'`` picks CUDA when available and falls back to CPU
                otherwise; ``'0'`` forces the first GPU, ``'cpu'`` forces CPU.
                The default used to be ``'0'``, which made the CLI crash with
                ``RuntimeError: Invalid device string: '0'`` on any CPU-only
                install.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Weights not found: {model_path}")

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = self._resolve_device(device)

        from ultralytics import YOLO  # imported lazily: heavy dependency

        logger.info("Loading classification model: %s", model_path)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.results: List[ClassificationResult] = []
        logger.info("Model loaded successfully (task=%s, device=%s)",
                    self.model.task, self.device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """
        Turn a requested device into one torch can actually use.

        Args:
            device: ``'auto'``, a GPU index such as ``'0'``, or ``'cpu'``.

        Returns:
            A device string that is valid on this machine. ``'auto'`` becomes
            ``'0'`` when CUDA is available and ``'cpu'`` otherwise; an
            explicitly requested GPU index also degrades to CPU with a warning
            rather than aborting the run.
        """
        import torch

        has_cuda = torch.cuda.is_available()
        if device == "auto":
            resolved = "0" if has_cuda else "cpu"
            logger.info("Device auto-selected: %s", resolved)
            return resolved
        if device != "cpu" and not has_cuda:
            logger.warning(
                "Device '%s' requested but CUDA is unavailable; using CPU.",
                device,
            )
            return "cpu"
        return device

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def classify_image(
        self,
        image_path: str,
        save_output: bool = True,
        output_dir: str = "runs/inference",
        distance_mm: float = 2000,
        focal_length_px: float = 800,
        enable_measurement: bool = False,
    ) -> ClassificationResult:
        """
        Classify one image file and assess its structural condition.

        Args:
            image_path: Path to the input image.
            save_output: Save annotated image + JSON report next to results.
            output_dir: Directory for outputs.
            distance_mm: Camera distance for measurement (if enabled).
            focal_length_px: Focal length for measurement (if enabled).
            enable_measurement: Run the photogrammetric crack measurement.

        Returns:
            :class:`ClassificationResult`.

        Raises:
            FileNotFoundError: If the image cannot be read.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        result = self._predict(img, source_label=image_path)
        if enable_measurement and result.is_damaged:
            result.measurement = estimate_crack_width(
                img,
                distance_mm=distance_mm,
                focal_length_px=focal_length_px,
                structure_type=result.structure_type,
            )
            measurement = result.measurement
            if measurement.get("measured"):
                # Trust gate: geometry alone never escalates severity. When
                # classification confidence is low or the calibration is too
                # coarse to resolve serviceability widths for this element,
                # the measured width is reported but severity stays at the
                # conservative confidence-based fallback.
                mm_per_px = float(measurement.get("mm_per_px", distance_mm / focal_length_px))
                if can_escalate_severity(
                    result.confidence, mm_per_px, result.structure_type
                ):
                    result.severity = measurement["severity"]
                    result.safety_factor = measurement["safety_factor"]
                else:
                    fallback = assess_severity_from_confidence(
                        result.confidence, True
                    )
                    result.severity = fallback
                    result.safety_factor = get_safety_factor(fallback)
                    result.recommendation += (
                        f" [Measured width {measurement['avg_width_mm']:.2f}mm shown "
                        f"for reference; severity not escalated: insufficient "
                        f"confidence/calibration resolution.]"
                    )
            elif result.severity is not None and measurement.get("severity"):
                # Measurement ran but rejected the blob -> keep conservative
                # confidence-based severity, but surface the reason.
                result.recommendation += f" [{measurement.get('message', '')}]"

        self.results.append(result)

        if save_output:
            self._save_single_output(img, result, output_dir)

        return result

    def _predict(self, img: np.ndarray, source_label: str) -> ClassificationResult:
        """Run the model on an in-memory BGR frame and build the result."""
        results = self.model(img, device=self.device, verbose=False)
        probs = results[0].probs

        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        predicted_class = CLASS_NAMES.get(class_id, f"class_{class_id}")

        all_probs = {
            CLASS_NAMES[i]: float(probs.data[i])
            for i in CLASS_NAMES
            if i < len(probs.data)
        }

        structure_type = STRUCTURE_MAP.get(predicted_class, "wall")
        condition = CONDITION_MAP.get(predicted_class, "uncracked")
        is_damaged = condition == "cracked" and predicted_class != OTHER_CLASS

        if predicted_class == OTHER_CLASS:
            severity: Optional[DamageSeverity] = DamageSeverity.UNCERTAIN
        elif is_damaged:
            # Conservative confidence-based fallback; refined later when a
            # physical measurement is available.
            if confidence < self.confidence_threshold:
                severity = DamageSeverity.UNCERTAIN
            else:
                severity = assess_severity_from_confidence(confidence, True)
        else:
            severity = DamageSeverity.NONE

        # Severity is never NONE while the verdict is unresolved: an
        # out-of-scope (``z_other``) prediction carries UNCERTAIN severity, so
        # it must not be reported with safety_factor 1.0 -- that contradicted
        # app.py, which uses SAFETY_FACTORS[UNCERTAIN] = 0.90 for the same
        # prediction and told the user it is "not a healthy verdict".
        safety_factor = (
            1.0
            if severity == DamageSeverity.NONE
            else get_safety_factor(severity)
        )
        recommendation = self._generate_recommendation(
            structure_type, condition, severity, confidence
        )

        return ClassificationResult(
            image_path=source_label,
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probs,
            structure_type=structure_type,
            condition=condition,
            severity=severity,
            recommendation=recommendation,
            is_damaged=is_damaged,
            safety_factor=safety_factor,
        )

    # ------------------------------------------------------------------
    # Batch / video / stream
    # ------------------------------------------------------------------

    def classify_batch(
        self,
        image_dir: str,
        output_dir: str = "runs/inference",
        save_report: bool = True,
        enable_measurement: bool = False,
        distance_mm: float = 2000,
        focal_length_px: float = 800,
    ) -> List[ClassificationResult]:
        """
        Recursively classify all images in a directory.

        Args:
            image_dir: Root directory to scan.
            output_dir: Output directory for artifacts.
            save_report: Write consolidated CSV + summary.txt.
            enable_measurement: Run crack measurement on cracked predictions.
            distance_mm / focal_length_px: Measurement camera parameters.

        Returns:
            Successfully processed results (failures are logged, not raised).
        """
        image_paths: List[Path] = []
        root = Path(image_dir)
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(root.rglob(ext))

        logger.info("Found %d images in %s", len(image_paths), image_dir)

        all_results: List[ClassificationResult] = []
        for i, img_path in enumerate(sorted(image_paths), 1):
            try:
                result = self.classify_image(
                    str(img_path),
                    save_output=True,
                    output_dir=output_dir,
                    enable_measurement=enable_measurement,
                    distance_mm=distance_mm,
                    focal_length_px=focal_length_px,
                )
                all_results.append(result)
            except Exception as exc:
                logger.error("Error processing %s: %s", img_path, exc)
            if i % 25 == 0:
                logger.info("Progress: %d/%d", i, len(image_paths))

        if save_report and all_results:
            self._save_consolidated_report(all_results, output_dir)

        return all_results

    def classify_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        sample_interval: int = 5,
    ) -> List[ClassificationResult]:
        """
        Classify sampled frames of a video file (frames stay in memory).

        Args:
            video_path: Input video file.
            output_path: Optional annotated-output video path.
            sample_interval: Process every Nth frame.

        Returns:
            Per-sampled-frame results.

        Raises:
            ValueError: If the video cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            "Video: %dx%d @ %.1ffps, %d frames", width, height, fps, total
        )

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps / max(sample_interval, 1),
                (width, height),
            )

        frame_results: List[ClassificationResult] = []
        frame_count = processed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % sample_interval != 0:
                    continue

                try:
                    result = self._predict(
                        frame, source_label=f"{video_path}#frame_{frame_count}"
                    )
                    frame_results.append(result)
                    processed += 1
                    if writer:
                        writer.write(self._annotate_image(frame, result))
                    if processed % 10 == 0:
                        logger.info("Processed %d frames...", processed)
                except Exception as exc:
                    logger.error("Error processing frame %d: %s", frame_count, exc)
        finally:
            cap.release()
            if writer:
                writer.release()

        logger.info("Video classification complete: %d frames processed.", processed)
        return frame_results

    def process_stream(
        self,
        source: Union[int, str] = 0,
        display: bool = True,
        save_video: bool = False,
        output_path: str = "runs/inference/stream_output.mp4",
    ) -> None:
        """
        Real-time classification over webcam / RTSP (frames stay in memory).

        Args:
            source: Webcam index or RTSP/HTTP stream URL.
            display: Show a live OpenCV window (requires a GUI environment).
            save_video: Record the annotated stream to ``output_path``.
            output_path: Output recording path.

        Raises:
            ValueError: If the stream cannot be opened.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open stream: {source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        writer = None
        if save_video:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

        frame_count = 0
        logger.info("Stream started (%s). Press 'q' to quit.", source)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame capture failed, retrying...")
                    time.sleep(0.1)
                    continue

                frame_count += 1
                try:
                    result = self._predict(
                        frame, source_label=f"{source}#frame_{frame_count}"
                    )
                    frame = self._annotate_image(frame, result)
                except Exception as exc:
                    logger.error("Frame classification error: %s", exc)

                if writer:
                    writer.write(frame)
                if display:
                    cv2.putText(
                        frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    )
                    cv2.imshow("SHM Classification", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User interrupted stream")
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        logger.info("Stream complete: %d frames processed.", frame_count)

    # ------------------------------------------------------------------
    # Assessment / rendering / persistence helpers
    # ------------------------------------------------------------------

    def _generate_recommendation(
        self,
        structure_type: str,
        condition: str,
        severity: Optional[DamageSeverity],
        confidence: float,
    ) -> str:
        """Compose a recommendation with element-specific thresholds."""
        if structure_type == "z":
            return (
                "OUT OF SCOPE: the image does not look like one of the trained "
                "structure surfaces (deck / pavement / wall). Not a healthy "
                "verdict -- inspect manually or extend the training data."
            )
        if condition == "uncracked":
            return (
                f"{structure_type.upper()}: No damage detected. "
                f"Continue routine inspection schedule."
            )

        thresholds = get_structure_thresholds(structure_type)
        rec = (
            f"{structure_type.upper()} CRACK DETECTED (confidence: {confidence:.1%}). "
            f"{get_recommendation(severity)} "
            f"Critical threshold for {structure_type}: {thresholds.critical_width}mm. "
            f"Max allowable: {thresholds.max_allowable}mm."
        )
        if severity == DamageSeverity.UNCERTAIN:
            rec += (
                " WARNING: Low model confidence. Manual inspection and "
                "physical measurement required."
            )
        return rec

    def _annotate_image(
        self, img: np.ndarray, result: ClassificationResult
    ) -> np.ndarray:
        """Render classification HUD (panel + probability bars) on a copy."""
        img = img.copy()
        h, w = img.shape[:2]

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        color = CLASS_COLORS.get(result.predicted_class, (0, 0, 255))
        cv2.putText(img, result.predicted_class.upper(), (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(img, f"Confidence: {result.confidence:.1%}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            img,
            f"Structure: {result.structure_type} | Condition: {result.condition}",
            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )

        if result.is_damaged and result.severity is not None:
            safety_text = (
                f"SAFETY FACTOR: {result.safety_factor:.2f} | "
                f"{result.severity.value.upper()}"
            )
            cv2.putText(img, safety_text, (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        self._draw_probability_bars(img, result)
        return img

    @staticmethod
    def _draw_probability_bars(
        img: np.ndarray, result: ClassificationResult
    ) -> None:
        """Draw in-image probability bars for all classes."""
        h, w = img.shape[:2]
        bar_x = max(w - 300, 10)
        bar_y_start = 150
        bar_height, bar_spacing, max_bar_width = 25, 30, 200

        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (bar_x - 10, bar_y_start - 30),
            (w - 10, bar_y_start + len(result.all_probabilities) * bar_spacing + 10),
            (0, 0, 0), -1,
        )
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        cv2.putText(img, "CLASS PROBABILITIES", (bar_x, bar_y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for idx, (class_name, prob) in enumerate(result.all_probabilities.items()):
            y = bar_y_start + idx * bar_spacing
            bar_width = int(prob * max_bar_width)
            color = (0, 255, 0) if "uncracked" in class_name else (0, 0, 255)
            cv2.rectangle(img, (bar_x, y), (bar_x + bar_width, y + bar_height),
                          color, -1)
            cv2.rectangle(img, (bar_x, y), (bar_x + max_bar_width, y + bar_height),
                          (255, 255, 255), 1)
            cv2.putText(img, f"{class_name}: {prob:.1%}", (bar_x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _save_single_output(
        self,
        img: np.ndarray,
        result: ClassificationResult,
        output_dir: str,
    ) -> None:
        """Persist annotated image + JSON report for one image."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = Path(result.image_path).stem

        annotated = self._annotate_image(img, result)
        img_path = out / f"{base}_classified.jpg"
        cv2.imwrite(str(img_path), annotated)

        json_path = out / f"{base}_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            payload = result.to_dict()
            payload["timestamp"] = datetime.now().isoformat()
            json.dump(payload, f, indent=2, default=str)

        logger.info("Saved: %s, %s", img_path, json_path)

    def _save_consolidated_report(
        self, results: List[ClassificationResult], output_dir: str
    ) -> None:
        """Write consolidated CSV + human-readable summary.txt."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        csv_path = out / "consolidated_report.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Image", "Predicted_Class", "Confidence", "Structure_Type",
                "Condition", "Severity", "Safety_Factor", "Is_Damaged",
                "Measured_Width_mm", "Recommendation",
            ])
            for r in results:
                writer.writerow([
                    r.image_path,
                    r.predicted_class,
                    f"{r.confidence:.4f}",
                    r.structure_type,
                    r.condition,
                    r.severity.value if r.severity else "N/A",
                    f"{r.safety_factor:.2f}",
                    r.is_damaged,
                    (
                        f"{r.measurement['avg_width_mm']:.3f}"
                        if r.measurement.get("measured") else ""
                    ),
                    r.recommendation,
                ])

        total = len(results)
        damaged = sum(1 for r in results if r.is_damaged)
        summary_path = out / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("STRUCTURAL HEALTH MONITORING - BATCH CLASSIFICATION SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Images Analyzed: {total}\n")
            f.write(f"Damaged Structures: {damaged} ({damaged / total:.1%})\n")
            f.write(f"Healthy Structures: {total - damaged} ({(total - damaged) / total:.1%})\n")
            f.write("\nPer-Class Breakdown:\n")

            class_counts: Dict[str, int] = {}
            for r in results:
                class_counts[r.predicted_class] = (
                    class_counts.get(r.predicted_class, 0) + 1
                )
            for class_name, count in sorted(class_counts.items()):
                f.write(f"  {class_name:25s}: {count:5d} ({count / total:.1%})\n")

        logger.info("Saved consolidated report: %s", csv_path)
        logger.info("Saved summary: %s", summary_path)


def main() -> None:
    """CLI entrypoint: single image / batch / video / stream modes."""
    parser = argparse.ArgumentParser(
        description="SHM Classification - Damage Detection & Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/inference.py --image sample.jpg --weights best.pt
  python src/inference.py --source data/raw/ --weights best.pt --save-report
  python src/inference.py --video inspection.mp4 --weights best.pt --output runs/inference
  python src/inference.py --stream 0 --weights best.pt
  python src/inference.py --image sample.jpg --weights best.pt --enable-measurement --distance-mm 2000 --focal-px 800
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Single image path")
    group.add_argument("--source", type=str, help="Directory of images (recursive)")
    group.add_argument("--video", type=str, help="Video file path")
    group.add_argument(
        "--stream", type=str, help="Stream source (0 for webcam, or RTSP URL)"
    )

    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.75,
                        help="Minimum confidence threshold (default: 0.75)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device ('auto' = GPU if available else CPU, "
                             "'0' first GPU, 'cpu' for CPU)")

    parser.add_argument("--output", type=str, default="runs/inference",
                        help="Output directory")
    parser.add_argument("--save-report", action="store_true",
                        help="Save consolidated batch report")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable live display (headless stream mode)")
    parser.add_argument("--save-video", action="store_true",
                        help="Record annotated output (stream/video mode)")
    parser.add_argument("--sample-interval", type=int, default=5,
                        help="Process every Nth frame (video mode)")

    parser.add_argument("--enable-measurement", action="store_true",
                        help="Run photogrammetric crack measurement")
    parser.add_argument("--distance-mm", type=float, default=2000,
                        help="Camera-to-surface distance in mm (default: 2000)")
    parser.add_argument("--focal-px", type=float, default=800,
                        help="Focal length in pixels (default: 800)")

    args = parser.parse_args()

    classifier = StructuralClassifier(
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
    )

    if args.image:
        result = classifier.classify_image(
            args.image,
            save_output=True,
            output_dir=args.output,
            enable_measurement=args.enable_measurement,
            distance_mm=args.distance_mm,
            focal_length_px=args.focal_px,
        )
        print("=" * 70)
        print(f"Image: {result.image_path}")
        print(f"Predicted Class: {result.predicted_class}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Structure Type: {result.structure_type}")
        print(f"Condition: {result.condition}")
        if result.is_damaged:
            print(f"Severity: {result.severity.value}")
            print(f"Safety Factor: {result.safety_factor:.2f}")
        if result.measurement.get("measured"):
            print(
                f"Measured Width: {result.measurement['avg_width_mm']:.3f}mm "
                f"(max {result.measurement['max_width_mm']:.3f}mm)"
            )
        print(f"Recommendation: {result.recommendation}")
        print("=" * 70)

    elif args.source:
        results = classifier.classify_batch(
            args.source,
            output_dir=args.output,
            save_report=args.save_report,
            enable_measurement=args.enable_measurement,
            distance_mm=args.distance_mm,
            focal_length_px=args.focal_px,
        )
        print(f"Processed {len(results)} images")

    elif args.video:
        results = classifier.classify_video(
            args.video,
            output_path=(
                os.path.join(args.output, "output_video.mp4")
                if args.save_video else None
            ),
            sample_interval=args.sample_interval,
        )
        print(f"Processed {len(results)} frames")

    elif args.stream:
        stream_source = (
            int(args.stream) if str(args.stream).isdigit() else args.stream
        )
        classifier.process_stream(
            source=stream_source,
            display=not args.no_display,
            save_video=args.save_video,
            output_path=os.path.join(args.output, "stream_output.mp4"),
        )


if __name__ == "__main__":
    main()
