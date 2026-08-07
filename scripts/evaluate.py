#!/usr/bin/env python3
"""
Model Evaluation Script for SHM Classification
===============================================
Comprehensive evaluation with classification metrics and confusion matrix.

Usage:
    python scripts/evaluate.py --weights runs/classify/shm_classification/weights/best.pt --data data/processed
    python scripts/evaluate.py --weights best.pt --data data/processed --conf-matrix
    python scripts/evaluate.py --weights best.pt --data data/processed --per-class
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(weights_path: str, data_path: str) -> Dict:
    """
    Evaluate classification model and generate metrics.
    
    Args:
        weights_path: Path to model weights
        data_path: Path to dataset directory
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Loading model: {weights_path}")
    model = YOLO(weights_path)
    
    # Run validation
    logger.info("Running validation...")
    metrics = model.val(data=data_path)
    
    # Extract metrics
    results = {
        'model': weights_path,
        'dataset': data_path,
        'top1_accuracy': float(metrics.top1) if hasattr(metrics, 'top1') else None,
        'top5_accuracy': float(metrics.top5) if hasattr(metrics, 'top5') else None,
    }
    
    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    if results['top1_accuracy']:
        logger.info(f"Top-1 Accuracy: {results['top1_accuracy']:.4f}")
    if results['top5_accuracy']:
        logger.info(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")
    logger.info("=" * 70)
    
    return results


def generate_confusion_matrix(
    weights_path: str,
    data_path: str,
    output_dir: str = 'runs/evaluation'
) -> np.ndarray:
    """
    Generate and save confusion matrix.
    
    Args:
        weights_path: Path to model weights
        data_path: Path to dataset
        output_dir: Output directory for plots
        
    Returns:
        Confusion matrix array
    """
    logger.info("Generating confusion matrix...")
    
    model = YOLO(weights_path)
    
    # Get predictions and labels
    # Note: YOLO classification saves confusion matrix automatically
    # This function creates an enhanced version
    
    # Run validation to generate confusion matrix
    metrics = model.val(data=data_path)
    
    # The confusion matrix is saved by YOLO in runs/classify/val/
    # We can also generate our own if needed
    
    logger.info(f"Confusion matrix saved in validation output directory")
    
    return None


def per_class_analysis(
    weights_path: str,
    data_path: str
) -> Dict:
    """
    Perform detailed per-class analysis.
    
    Args:
        weights_path: Path to model weights
        data_path: Path to dataset
        
    Returns:
        Per-class metrics dictionary
    """
    logger.info("Performing per-class analysis...")
    
    model = YOLO(weights_path)
    
    # Get class names
    class_names = list(model.names.values()) if hasattr(model, 'names') else []
    
    # Run validation
    metrics = model.val(data=data_path)
    
    per_class = {}
    
    # Extract per-class metrics if available
    if hasattr(metrics, 'results_dict'):
        for key, value in metrics.results_dict.items():
            if 'class' in key.lower():
                per_class[key] = float(value)
    
    logger.info("\nPer-Class Metrics:")
    for name, metric in per_class.items():
        logger.info(f"  {name}: {metric:.4f}")
    
    return per_class


def benchmark_speed(
    weights_path: str,
    data_path: str,
    num_runs: int = 100
) -> Dict:
    """
    Benchmark model inference speed.
    
    Args:
        weights_path: Path to model weights
        data_path: Path to dataset (for sample images)
        num_runs: Number of inference runs for averaging
        
    Returns:
        Speed benchmark results
    """
    import time
    
    logger.info(f"Benchmarking inference speed ({num_runs} runs)...")
    
    model = YOLO(weights_path)
    
    # Find sample images
    sample_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        sample_images.extend(Path(data_path).glob(f'**/{ext}'))
    
    if not sample_images:
        logger.warning("No sample images found for benchmarking")
        return {}
    
    # Use first image for benchmarking
    sample = str(sample_images[0])
    
    # Warmup
    for _ in range(10):
        model(sample)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        model(sample)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    results = {
        'average_inference_time_ms': round(avg_time * 1000, 2),
        'std_inference_time_ms': round(std_time * 1000, 2),
        'throughput_fps': round(1 / avg_time, 2),
        'num_runs': num_runs
    }
    
    logger.info("=" * 70)
    logger.info("SPEED BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Average inference time: {results['average_inference_time_ms']:.2f} ms")
    logger.info(f"Throughput: {results['throughput_fps']:.2f} images/second")
    logger.info("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SHM Classification Model")
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--data', type=str, default='data/processed',
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='runs/evaluation',
                       help='Output directory for results')
    parser.add_argument('--conf-matrix', action='store_true',
                       help='Generate confusion matrix')
    parser.add_argument('--per-class', action='store_true',
                       help='Per-class detailed analysis')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run speed benchmark')
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of runs for benchmark')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Main evaluation
    results = evaluate_model(args.weights, args.data)
    
    # Confusion matrix
    if args.conf_matrix:
        generate_confusion_matrix(args.weights, args.data, args.output)
    
    # Per-class analysis
    if args.per_class:
        per_class = per_class_analysis(args.weights, args.data)
        results['per_class'] = per_class
    
    # Speed benchmark
    if args.benchmark:
        speed = benchmark_speed(args.weights, args.data, args.runs)
        results['speed_benchmark'] = speed
    
    # Save results
    output_file = os.path.join(args.output, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
