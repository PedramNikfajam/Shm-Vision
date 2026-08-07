#!/usr/bin/env python3
"""
YOLOv8 Classification Training Pipeline for Structural Health Monitoring
=========================================================================
Professional training script for multi-class concrete damage classification.

Classes:
    0: deck_cracked      - Concrete deck with cracks
    1: deck_uncracked    - Healthy concrete deck
    2: pavement_cracked  - Pavement/road with cracks
    3: pavement_uncracked- Healthy pavement/road
    4: wall_cracked      - Wall with cracks
    5: wall_uncracked    - Healthy wall

Usage:
    # Train with default settings
    python src/train.py --data data/processed --epochs 150 --imgsz 224
    
    # Use hyperparameter config file
    python src/train.py --config config/hyperparams.yaml
    
    # Resume interrupted training
    python src/train.py --resume runs/classify/shm_classification/weights/last.pt
    
    # Validate trained model
    python src/train.py --validate --weights runs/classify/shm_classification/weights/best.pt
    
    # Export to ONNX for deployment
    python src/train.py --export --weights best.pt --format onnx
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
from ultralytics import YOLO

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
os.makedirs(PROJECT_ROOT / 'runs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'runs' / 'training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def load_hyperparams(config_path: str) -> Dict[str, Any]:
    """Load hyperparameters from YAML configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def setup_training_args(args: argparse.Namespace, hyperparams: Dict) -> Dict[str, Any]:
    """Merge CLI arguments with hyperparameters from config file."""
    training_args = {}
    
    if hyperparams:
        # Model config
        model_cfg = hyperparams.get('model', {})
        training_args['model'] = args.model if args.model else model_cfg.get('base_weights', 'yolov8n-cls.pt')
        
        # Training config
        train_cfg = hyperparams.get('training', {})
        training_args['epochs'] = args.epochs if args.epochs else train_cfg.get('epochs', 150)
        training_args['imgsz'] = args.imgsz if args.imgsz else train_cfg.get('imgsz', 224)
        training_args['batch'] = args.batch if args.batch else train_cfg.get('batch', 64)
        training_args['patience'] = train_cfg.get('patience', 20)
        training_args['device'] = args.device if args.device else train_cfg.get('device', '0')
        training_args['workers'] = args.workers if args.workers is not None else train_cfg.get('workers', 8)
        training_args['cache'] = train_cfg.get('cache', True)
        
        # Optimizer config
        opt_cfg = hyperparams.get('optimizer', {})
        training_args['optimizer'] = opt_cfg.get('name', 'AdamW')
        training_args['lr0'] = opt_cfg.get('lr0', 0.001)
        training_args['lrf'] = opt_cfg.get('lrf', 0.01)
        training_args['momentum'] = opt_cfg.get('momentum', 0.937)
        training_args['weight_decay'] = opt_cfg.get('weight_decay', 0.0005)
        training_args['warmup_epochs'] = opt_cfg.get('warmup_epochs', 3.0)
        
        # Augmentation config
        aug_cfg = hyperparams.get('augmentation', {})
        training_args['hsv_h'] = aug_cfg.get('hsv_h', 0.015)
        training_args['hsv_s'] = aug_cfg.get('hsv_s', 0.7)
        training_args['hsv_v'] = aug_cfg.get('hsv_v', 0.4)
        training_args['degrees'] = aug_cfg.get('degrees', 15.0)
        training_args['translate'] = aug_cfg.get('translate', 0.1)
        training_args['scale'] = aug_cfg.get('scale', 0.5)
        training_args['shear'] = aug_cfg.get('shear', 2.0)
        training_args['flipud'] = aug_cfg.get('flipud', 0.0)
        training_args['fliplr'] = aug_cfg.get('fliplr', 0.5)
        training_args['mixup'] = aug_cfg.get('mixup', 0.1)
        training_args['erasing'] = aug_cfg.get('erasing', 0.4)
        training_args['auto_augment'] = aug_cfg.get('auto_augment', 'randaugment')
        
        # Loss config (class weights for imbalanced data)
        loss_cfg = hyperparams.get('loss', {})
        training_args['label_smoothing'] = loss_cfg.get('label_smoothing', 0.1)
        
        # Validation config
        val_cfg = hyperparams.get('validation', {})
        training_args['project'] = val_cfg.get('project', 'runs/classify')
        training_args['name'] = args.name if args.name else val_cfg.get('name', 'shm_classification')
        training_args['exist_ok'] = val_cfg.get('exist_ok', False)
        training_args['verbose'] = val_cfg.get('verbose', True)
        training_args['plots'] = val_cfg.get('plots', True)
        training_args['save_period'] = val_cfg.get('save_period', 10)
        
    else:
        # Use CLI arguments only
        training_args['model'] = args.model or 'yolov8n-cls.pt'
        training_args['epochs'] = args.epochs or 150
        training_args['imgsz'] = args.imgsz or 224
        training_args['batch'] = args.batch or 64
        training_args['patience'] = 20
        training_args['device'] = args.device or '0'
        training_args['workers'] = args.workers if args.workers is not None else 8
        training_args['cache'] = True
        training_args['project'] = 'runs/classify'
        training_args['name'] = args.name or 'shm_classification'
        training_args['exist_ok'] = False
        training_args['verbose'] = True
        training_args['plots'] = True
    
    # Data path
    training_args['data'] = args.data or 'data/processed'
    
    return training_args


def log_training_config(args: Dict[str, Any]) -> None:
    """Log training configuration for reproducibility."""
    logger.info("=" * 70)
    logger.info("STRUCTURAL HEALTH MONITORING - CLASSIFICATION TRAINING")
    logger.info("=" * 70)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info("-" * 70)
    
    for key, value in sorted(args.items()):
        if key not in ['data', 'project', 'name']:
            logger.info(f"  {key}: {value}")
    logger.info("-" * 70)


def train_model(args: Dict[str, Any]) -> None:
    """Execute YOLOv8 classification training."""
    try:
        # Initialize model
        logger.info(f"Loading classification model: {args['model']}")
        model = YOLO(args['model'])
        
        # Log model info
        logger.info(f"Model task: {model.task}")
        logger.info(f"Model classes: {getattr(model, 'names', 'N/A')}")
        
        # Prepare training arguments
        train_kwargs = {
            'data': args['data'],
            'epochs': args['epochs'],
            'imgsz': args['imgsz'],
            'batch': args['batch'],
            'patience': args.get('patience', 20),
            'device': args['device'],
            'workers': args.get('workers', 8),
            'cache': args.get('cache', True),
            'project': args['project'],
            'name': args['name'],
            'exist_ok': args.get('exist_ok', False),
            'verbose': args.get('verbose', True),
            'plots': args.get('plots', True),
            'save_period': args.get('save_period', 10),
        }
        
        # Add optimizer settings
        for param in ['optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs']:
            if param in args:
                train_kwargs[param] = args[param]
        
        # Add augmentation settings
        for param in ['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 
                     'scale', 'shear', 'flipud', 'fliplr', 'mixup', 
                     'erasing', 'auto_augment']:
            if param in args:
                train_kwargs[param] = args[param]
        
        # Add loss settings
        if 'label_smoothing' in args:
            train_kwargs['label_smoothing'] = args['label_smoothing']
        
        # Log final training arguments
        logger.info("Training with parameters:")
        for key, value in train_kwargs.items():
            logger.info(f"  {key}: {value}")
        
        # Start training
        logger.info("Starting classification training...")
        results = model.train(**train_kwargs)
        
        # Log results
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
        # Extract metrics from results
        if hasattr(results, 'results_dict'):
            logger.info(f"Final Accuracy: {results.results_dict.get('metrics/accuracy_top1', 'N/A')}")
            logger.info(f"Top-5 Accuracy: {results.results_dict.get('metrics/accuracy_top5', 'N/A')}")
        
        weights_dir = f"{args['project']}/{args['name']}/weights"
        logger.info(f"Best model saved to: {weights_dir}/best.pt")
        logger.info(f"Last model saved to: {weights_dir}/last.pt")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


def resume_training(resume_path: str) -> None:
    """Resume training from a checkpoint."""
    try:
        logger.info(f"Resuming training from: {resume_path}")
        model = YOLO(resume_path)
        results = model.train(resume=True)
        logger.info("Training resumed and completed successfully")
    except Exception as e:
        logger.error(f"Failed to resume training: {str(e)}", exc_info=True)
        raise


def validate_model(model_path: str, data_path: str) -> None:
    """Validate a trained classification model."""
    try:
        logger.info(f"Validating model: {model_path}")
        model = YOLO(model_path)
        metrics = model.val(data=data_path)
        
        logger.info("=" * 70)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 70)
        
        if hasattr(metrics, 'top1'):
            logger.info(f"Top-1 Accuracy: {metrics.top1:.4f}")
        if hasattr(metrics, 'top5'):
            logger.info(f"Top-5 Accuracy: {metrics.top5:.4f}")
        
        # Per-class accuracy if available
        if hasattr(metrics, 'results_dict'):
            for key, value in metrics.results_dict.items():
                if 'accuracy' in key.lower():
                    logger.info(f"{key}: {value:.4f}")
        
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}", exc_info=True)
        raise


def export_model(model_path: str, format: str = 'onnx') -> None:
    """Export trained model to deployment format."""
    try:
        logger.info(f"Exporting model to {format.upper()}: {model_path}")
        model = YOLO(model_path)
        
        # Export with appropriate parameters
        export_kwargs = {'format': format}
        if format == 'onnx':
            export_kwargs['opset'] = 12
        
        model.export(**export_kwargs)
        logger.info(f"Model exported successfully to {format.upper()} format")
    except Exception as e:
        logger.error(f"Export failed: {str(e)}", exc_info=True)
        raise


def benchmark_model(model_path: str, data_path: str) -> None:
    """Benchmark model performance (speed and accuracy)."""
    try:
        logger.info(f"Benchmarking model: {model_path}")
        model = YOLO(model_path)
        
        # Run validation to get accuracy
        metrics = model.val(data=data_path)
        
        # Run prediction on a few images to measure speed
        import time
        test_images = list(Path(data_path).glob('**/*.jpg'))[:10]
        
        if test_images:
            start = time.time()
            for img in test_images:
                model(img)
            elapsed = time.time() - start
            avg_time = elapsed / len(test_images)
            
            logger.info("=" * 70)
            logger.info("BENCHMARK RESULTS")
            logger.info("=" * 70)
            logger.info(f"Average inference time: {avg_time*1000:.2f} ms/image")
            logger.info(f"Throughput: {1/avg_time:.2f} images/second")
            if hasattr(metrics, 'top1'):
                logger.info(f"Top-1 Accuracy: {metrics.top1:.4f}")
            logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}", exc_info=True)


def main():
    """Main entry point for classification training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 Classification for Structural Health Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python src/train.py --data data/processed
  
  # Train with custom epochs and batch size
  python src/train.py --data data/processed --epochs 200 --batch 128
  
  # Train on CPU or fix Windows multiprocessing error
  python src/train.py --data data/processed --device cpu --workers 0
  
  # Use hyperparameter config file
  python src/train.py --config config/hyperparams.yaml
  
  # Resume interrupted training
  python src/train.py --resume runs/classify/shm_classification/weights/last.pt
  
  # Validate trained model
  python src/train.py --validate --weights runs/classify/shm_classification/weights/best.pt
  
  # Export to ONNX
  python src/train.py --export --weights best.pt --format onnx
  
  # Benchmark model performance
  python src/train.py --benchmark --weights best.pt --data data/processed
        """
    )
    
    # Main arguments
    parser.add_argument('--data', type=str, default='data/processed',
                       help='Path to processed dataset directory')
    parser.add_argument('--config', type=str, default='config/hyperparams.yaml',
                       help='Path to hyperparameters YAML config')
    parser.add_argument('--model', type=str,
                       help='Base model weights (yolov8n-cls.pt, yolov8s-cls.pt, etc.)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, help='Input image size (224 recommended)')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--device', type=str, help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--workers', type=int, help='Number of dataloader workers (0 to disable multiprocessing)')
    parser.add_argument('--name', type=str, help='Experiment name')
    
    # Actions
    parser.add_argument('--resume', type=str, help='Resume from checkpoint path')
    parser.add_argument('--validate', action='store_true', help='Validate model')
    parser.add_argument('--weights', type=str, help='Model weights for validation/export')
    parser.add_argument('--export', action='store_true', help='Export model')
    parser.add_argument('--format', type=str, default='onnx',
                       help='Export format (onnx, torchscript, engine)')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark model')
    
    args = parser.parse_args()
    
    # Create runs directory
    os.makedirs(PROJECT_ROOT / 'runs', exist_ok=True)
    
    # Handle resume
    if args.resume:
        resume_training(args.resume)
        return
    
    # Handle validation
    if args.validate:
        if not args.weights:
            parser.error("--validate requires --weights")
        validate_model(args.weights, args.data)
        return
    
    # Handle export
    if args.export:
        if not args.weights:
            parser.error("--export requires --weights")
        export_model(args.weights, args.format)
        return
    
    # Handle benchmark
    if args.benchmark:
        if not args.weights:
            parser.error("--benchmark requires --weights")
        benchmark_model(args.weights, args.data)
        return
    
    # Normal training
    hyperparams = load_hyperparams(args.config)
    training_args = setup_training_args(args, hyperparams)
    log_training_config(training_args)
    train_model(training_args)


if __name__ == '__main__':
    main()
