#!/usr/bin/env python3
"""
Structural Health Monitoring - Classification Inference & Analysis Pipeline
============================================================================
Analyzes concrete structure images and classifies them into damage categories
with structural engineering assessment.

Usage:
    # Single image
    python src/inference.py --image data/raw/deck/cracked/sample.jpg --weights runs/classify/shm_classification/weights/best.pt
    
    # Batch processing
    python src/inference.py --source data/raw/ --weights best.pt --save-report
    
    # Video analysis
    python src/inference.py --video inspection.mp4 --weights best.pt --output results/
    
    # Real-time webcam
    python src/inference.py --stream 0 --weights best.pt

Features:
- Single image, batch, video, and real-time stream processing
- Structural condition assessment per class
- Safety rating based on structure type and damage condition
- Annotated output with classification results
- JSON/CSV report generation
- Confidence threshold filtering
"""

import argparse
import json
import csv
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from ultralytics import YOLO

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eng_utils import (
    DamageSeverity, DamageType, RECOMMENDATIONS,
    structural_safety_rating, pixels_to_mm
)

from config import (
    DamageSeverity,
    get_safety_factor,
    get_recommendation,
    UNCERTAINTY_THRESHOLD,
)
from crack_measurement import estimate_crack_width

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Classification result with structural assessment."""
    image_path: str
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    structure_type: str          # deck, pavement, wall
    condition: str               # cracked, uncracked
    severity: Optional[DamageSeverity]
    recommendation: str
    is_damaged: bool
    safety_factor: float


class StructuralClassifier:
    """
    Main class for structural image classification combining YOLO 
    classification with civil engineering assessment.
    """
    
    # Class mapping
    CLASS_NAMES = {
        0: 'deck_cracked',
        1: 'deck_uncracked',
        2: 'pavement_cracked',
        3: 'pavement_uncracked',
        4: 'wall_cracked',
        5: 'wall_uncracked'
    }
    
    # Structure type mapping
    STRUCTURE_MAP = {
        'deck_cracked': 'deck',
        'deck_uncracked': 'deck',
        'pavement_cracked': 'pavement',
        'pavement_uncracked': 'pavement',
        'wall_cracked': 'wall',
        'wall_uncracked': 'wall'
    }
    
    # Condition mapping
    CONDITION_MAP = {
        'deck_cracked': 'cracked',
        'deck_uncracked': 'uncracked',
        'pavement_cracked': 'cracked',
        'pavement_uncracked': 'uncracked',
        'wall_cracked': 'cracked',
        'wall_uncracked': 'uncracked'
    }
    
    # Color mapping for visualization (BGR)
    CLASS_COLORS = {
        'deck_cracked': (0, 0, 255),         # Red
        'deck_uncracked': (0, 255, 0),       # Green
        'pavement_cracked': (0, 140, 255),   # Orange
        'pavement_uncracked': (0, 255, 0),   # Green
        'wall_cracked': (0, 0, 255),         # Red
        'wall_uncracked': (0, 255, 0)        # Green
    }
    
    # Safety thresholds per structure type (crack width in mm)
    SAFETY_THRESHOLDS = {
        'deck': {
            'critical_width': 0.3,
            'max_allowable': 0.5,
            'description': 'Bridge deck or floor slab'
        },
        'pavement': {
            'critical_width': 6.0,
            'max_allowable': 12.0,
            'description': 'Road or sidewalk pavement'
        },
        'wall': {
            'critical_width': 0.3,
            'max_allowable': 1.0,
            'description': 'Concrete or masonry wall'
        }
    }
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.75,
        device: str = '0'
    ):
        """
        Initialize the structural classifier.
        
        Args:
            model_path: Path to trained YOLO classification weights
            confidence_threshold: Minimum confidence for reliable prediction
            device: Computation device ('0' for GPU, 'cpu' for CPU)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Load model
        logger.info(f"Loading classification model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        logger.info("Model loaded successfully")
        
        # Results storage
        self.results: List[ClassificationResult] = []
        
    def classify_image(
        self,
        image_path: str,
        save_output: bool = True,
        output_dir: str = 'runs/inference',
        distance_mm: float = 2000,
        focal_length_px: float = 800,
    ) -> ClassificationResult:
        """
        Classify a single image and perform structural assessment.
        
        Args:
            image_path: Path to input image
            save_output: Whether to save annotated image
            output_dir: Directory for output files
            
        Returns:
            ClassificationResult with assessment
        """
        logger.info(f"Classifying image: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run classification
        results = self.model(image_path, device=self.device)
        
        # Extract classification results
        result = results[0]
        probs = result.probs
        
        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        predicted_class = self.CLASS_NAMES[class_id]
        
        # Get all probabilities
        all_probs = {}
        for idx, name in self.CLASS_NAMES.items():
            all_probs[name] = float(probs.data[idx])
        
        # Determine structure type and condition
        structure_type = self.STRUCTURE_MAP[predicted_class]
        condition = self.CONDITION_MAP[predicted_class]
        is_damaged = (condition == 'cracked')
        
        # Assess severity
                # Assess severity
        if is_damaged:
            severity = self._assess_crack_severity(
                structure_type, confidence, img, distance_mm, focal_length_px
            )
        else:
            severity = DamageSeverity.NONE
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            structure_type, condition, severity, confidence
        )
        
        # Calculate safety factor
        safety_factor = self._calculate_safety_factor(severity, is_damaged)
        
        # Create result
        classification_result = ClassificationResult(
            image_path=image_path,
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probs,
            structure_type=structure_type,
            condition=condition,
            severity=severity,
            recommendation=recommendation,
            is_damaged=is_damaged,
            safety_factor=safety_factor
        )
        
        self.results.append(classification_result)
        
        # Save output
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save annotated image
            base_name = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{base_name}_classified.jpg")
            annotated = self._annotate_image(img, classification_result)
            cv2.imwrite(output_path, annotated)
            logger.info(f"Saved annotated image: {output_path}")
            
            # Save JSON report
            json_path = os.path.join(output_dir, f"{base_name}_report.json")
            with open(json_path, 'w') as f:
                json.dump(self._result_to_dict(classification_result), f, indent=2)
            logger.info(f"Saved JSON report: {json_path}")
        
        return classification_result
    
    def classify_batch(
        self,
        image_dir: str,
        output_dir: str = 'runs/inference',
        save_report: bool = True
    ) -> List[ClassificationResult]:
        """
        Classify multiple images in a directory.
        
        Args:
            image_dir: Directory containing images
            output_dir: Output directory
            save_report: Whether to save consolidated report
            
        Returns:
            List of ClassificationResult
        """
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(Path(image_dir).glob(ext))
        
        logger.info(f"Found {len(image_paths)} images in {image_dir}")
        
        all_results = []
        for img_path in image_paths:
            try:
                result = self.classify_image(str(img_path), save_output=True, output_dir=output_dir)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
        
        # Save consolidated report
        if save_report and all_results:
            self._save_consolidated_report(all_results, output_dir)
        
        return all_results
    
    def classify_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        sample_interval: int = 5
    ) -> List[ClassificationResult]:
        """
        Classify video frames.
        
        Args:
            video_path: Path to video file
            output_path: Path for output video
            sample_interval: Process every Nth frame
            
        Returns:
            List of frame classification results
        """
        logger.info(f"Classifying video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps / sample_interval, (width, height))
        
        frame_results = []
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % sample_interval != 0:
                continue
            
            # Save temporary frame
            temp_path = f"/tmp/frame_{frame_count}.jpg"
            cv2.imwrite(temp_path, frame)
            
            try:
                result = self.classify_image(temp_path, save_output=False)
                result.image_path = f"{video_path}#frame_{frame_count}"
                frame_results.append(result)
                processed_count += 1
                
                # Annotate frame
                annotated = self._annotate_image(frame, result)
                if writer:
                    writer.write(annotated)
                
                if processed_count % 10 == 0:
                    logger.info(f"Processed {processed_count} frames...")
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        cap.release()
        if writer:
            writer.release()
        
        logger.info(f"Video classification complete. Processed {processed_count} frames.")
        return frame_results
    
    def process_stream(
        self,
        source: Union[int, str] = 0,
        display: bool = True,
        save_video: bool = False,
        output_path: str = 'runs/inference/stream_output.mp4'
    ):
        """
        Process real-time video stream.
        
        Args:
            source: Video source (0 for webcam, URL for IP camera)
            display: Show live display
            save_video: Save output video
            output_path: Output video path
        """
        logger.info(f"Starting stream processing from source: {source}")
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open stream: {source}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        writer = None
        if save_video:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        logger.info("Stream processing started. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame capture failed, retrying...")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Save temp and classify
            temp_path = "/tmp/stream_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            try:
                result = self.classify_image(temp_path, save_output=False)
                annotated = self._annotate_image(frame, result)
                frame = annotated
                
                # Add FPS counter
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                logger.error(f"Frame classification error: {e}")
            
            if writer:
                writer.write(frame)
            
            if display:
                cv2.imshow('SHM Classification', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User interrupted stream")
                    break
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Stream processing complete. Processed {frame_count} frames.")
    
        def _assess_crack_severity(
        self,
        structure_type: str,
        confidence: float,
        image: np.ndarray = None,
        distance_mm: float = 2000,
        focal_length_px: float = 800,
    ) -> DamageSeverity:
         """
        Assess crack severity.
        If image is provided, attempts physical measurement first.
        Otherwise falls back to confidence-based assessment.
        """
        # Try physical measurement first if image available
        if image is not None:
            try:
                measurement = estimate_crack_width(image, distance_mm, focal_length_px)
                if measurement["measured"]:
                    logger.info(f"Physical measurement: {measurement['message']}")
                    return measurement["severity"]
            except Exception as e:
                logger.warning(f"Physical measurement failed, falling back to confidence: {e}")
        
        # Fallback to confidence-based assessment
        return assess_severity_from_confidence(confidence, is_damaged=True)
    
    def _generate_recommendation(
        self,
        structure_type: str,
        condition: str,
        severity: Optional[DamageSeverity],
        confidence: float
    ) -> str:
        """Generate structural recommendation based on classification."""
        if condition == 'uncracked':
            return f"{structure_type.upper()}: No damage detected. Continue routine inspection schedule."
        
        thresholds = self.SAFETY_THRESHOLDS.get(structure_type, self.SAFETY_THRESHOLDS['wall'])
        base_rec = get_recommendation(severity)
        
        rec = (
            f"{structure_type.upper()} CRACK DETECTED (confidence: {confidence:.1%}). "
            f"{base_rec} "
            f"Critical threshold for {structure_type}: {thresholds['critical_width']}mm. "
            f"Max allowable: {thresholds['max_allowable']}mm."
        )
        
        # Add uncertainty warning
        if severity == DamageSeverity.UNCERTAIN:
            rec += " WARNING: Low model confidence. Manual inspection and physical measurement required."
        
        return rec
    
    def _calculate_safety_factor(
        self,
        severity: Optional[DamageSeverity],
        is_damaged: bool
    ) -> float:
        """Calculate safety factor using centralized config."""
        if not is_damaged:
            return 1.0
        return get_safety_factor(severity)
    
    def _annotate_image(
        self,
        img: np.ndarray,
        result: ClassificationResult
    ) -> np.ndarray:
        """Annotate image with classification results."""
        h, w = img.shape[:2]
        
        # Create overlay for top panel
        overlay = img.copy()
        panel_height = 120
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        
        # Color based on condition
        color = (0, 255, 0) if not result.is_damaged else (0, 0, 255)
        
        # Main classification result
        label = f"{result.predicted_class.upper()}"
        cv2.putText(img, label, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Confidence
        conf_text = f"Confidence: {result.confidence:.1%}"
        cv2.putText(img, conf_text, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Structure type and condition
        info_text = f"Structure: {result.structure_type} | Condition: {result.condition}"
        cv2.putText(img, info_text, (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Safety indicator at bottom
        if result.is_damaged:
            safety_text = f"SAFETY FACTOR: {result.safety_factor:.2f} | {result.severity.value.upper()}"
            cv2.putText(img, safety_text, (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add probability bar chart on the right side
        self._draw_probability_bars(img, result)
        
        return img
    
    def _draw_probability_bars(self, img: np.ndarray, result: ClassificationResult):
        """Draw probability bars for all classes."""
        h, w = img.shape[:2]
        bar_x = w - 300
        bar_y_start = 150
        bar_height = 25
        bar_spacing = 30
        max_bar_width = 200
        
        # Background
        overlay = img.copy()
        cv2.rectangle(overlay, (bar_x - 10, bar_y_start - 30), 
                     (w - 10, bar_y_start + len(result.all_probabilities) * bar_spacing + 10),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        cv2.putText(img, "CLASS PROBABILITIES", (bar_x, bar_y_start - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for idx, (class_name, prob) in enumerate(result.all_probabilities.items()):
            y = bar_y_start + idx * bar_spacing
            bar_width = int(prob * max_bar_width)
            
            # Color based on condition
            color = (0, 255, 0) if 'uncracked' in class_name else (0, 0, 255)
            
            # Draw bar
            cv2.rectangle(img, (bar_x, y), (bar_x + bar_width, y + bar_height), color, -1)
            cv2.rectangle(img, (bar_x, y), (bar_x + max_bar_width, y + bar_height), (255, 255, 255), 1)
            
            # Label
            label = f"{class_name}: {prob:.1%}"
            cv2.putText(img, label, (bar_x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _result_to_dict(self, result: ClassificationResult) -> Dict:
        """Convert ClassificationResult to dictionary."""
        return {
            'image_path': result.image_path,
            'predicted_class': result.predicted_class,
            'confidence': result.confidence,
            'all_probabilities': result.all_probabilities,
            'structure_type': result.structure_type,
            'condition': result.condition,
            'severity': result.severity.value if result.severity else None,
            'recommendation': result.recommendation,
            'is_damaged': result.is_damaged,
            'safety_factor': result.safety_factor,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_consolidated_report(
        self,
        results: List[ClassificationResult],
        output_dir: str
    ):
        """Save consolidated CSV report."""
        csv_path = os.path.join(output_dir, 'consolidated_report.csv')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Image', 'Predicted_Class', 'Confidence', 'Structure_Type',
                'Condition', 'Severity', 'Safety_Factor', 'Is_Damaged',
                'Recommendation'
            ])
            
            for result in results:
                writer.writerow([
                    result.image_path,
                    result.predicted_class,
                    f"{result.confidence:.4f}",
                    result.structure_type,
                    result.condition,
                    result.severity.value if result.severity else 'N/A',
                    f"{result.safety_factor:.2f}",
                    result.is_damaged,
                    result.recommendation
                ])
        
        # Summary statistics
        total = len(results)
        damaged = sum(1 for r in results if r.is_damaged)
        
        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("STRUCTURAL HEALTH MONITORING - BATCH CLASSIFICATION SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Total Images Analyzed: {total}\n")
            f.write(f"Damaged Structures: {damaged} ({damaged/total*100:.1f}%)\n")
            f.write(f"Healthy Structures: {total-damaged} ({(total-damaged)/total*100:.1f}%)\n")
            f.write("\nPer-Class Breakdown:\n")
            
            class_counts = {}
            for r in results:
                class_counts[r.predicted_class] = class_counts.get(r.predicted_class, 0) + 1
            
            for class_name, count in sorted(class_counts.items()):
                f.write(f"  {class_name:25s}: {count:5d} ({count/total*100:.1f}%)\n")
        
        logger.info(f"Saved consolidated report: {csv_path}")
        logger.info(f"Saved summary: {summary_path}")


def main():
    """Main entry point for classification inference pipeline."""
    parser = argparse.ArgumentParser(
        description="SHM Classification - Damage Detection & Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python src/inference.py --image data/raw/deck/cracked/sample.jpg --weights runs/classify/shm_classification/weights/best.pt
  
  # Batch processing
  python src/inference.py --source data/raw/ --weights best.pt --save-report
  
  # Video analysis
  python src/inference.py --video inspection.mp4 --weights best.pt --output results/
  
  # Real-time webcam
  python src/inference.py --stream 0 --weights best.pt
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Single image path')
    input_group.add_argument('--source', type=str, help='Directory of images')
    input_group.add_argument('--video', type=str, help='Video file path')
    input_group.add_argument('--stream', type=str, help='Stream source (0 for webcam, URL for IP)')
    
    # Model and parameters
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.75,
                       help='Minimum confidence threshold (default: 0.75)')
    parser.add_argument('--device', type=str, default='0',
                       help='Device (0 for GPU, cpu for CPU)')
    
    # Output options
    parser.add_argument('--output', type=str, default='runs/inference',
                       help='Output directory')
    parser.add_argument('--save-report', action='store_true',
                       help='Save detailed report')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display (headless mode)')
    parser.add_argument('--save-video', action='store_true',
                       help='Save output video (stream mode)')
    
    # Video options
    parser.add_argument('--sample-interval', type=int, default=5,
                       help='Process every Nth frame (video only)')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = StructuralClassifier(
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device
    )
    
    # Process based on input type
    if args.image:
        result = classifier.classify_image(
            args.image,
            save_output=True,
            output_dir=args.output
        )
        print("\n" + "=" * 70)
        print("CLASSIFICATION RESULTS")
        print("=" * 70)
        print(f"Image: {result.image_path}")
        print(f"Predicted Class: {result.predicted_class}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Structure Type: {result.structure_type}")
        print(f"Condition: {result.condition}")
        if result.is_damaged:
            print(f"Severity: {result.severity.value}")
            print(f"Safety Factor: {result.safety_factor:.2f}")
        print(f"Recommendation: {result.recommendation}")
        print("=" * 70)
        
    elif args.source:
        results = classifier.classify_batch(
            args.source,
            output_dir=args.output,
            save_report=args.save_report
        )
        print(f"\nProcessed {len(results)} images")
        
    elif args.video:
        results = classifier.classify_video(
            args.video,
            output_path=os.path.join(args.output, 'output_video.mp4'),
            sample_interval=args.sample_interval
        )
        print(f"\nProcessed {len(results)} frames")
        
    elif args.stream:
        stream_source = int(args.stream) if args.stream.isdigit() else args.stream
        classifier.process_stream(
            source=stream_source,
            display=not args.no_display,
            save_video=args.save_video,
            output_path=os.path.join(args.output, 'stream_output.mp4')
        )


if __name__ == '__main__':
    main()
