#!/usr/bin/env python3
"""
Dataset Preparation Script for SHM Classification
=================================================
Converts raw organized images to YOLO classification format.

Expected Input Structure:
    data/raw/
    ├── deck/
    │   ├── cracked/          (1500+ images)
    │   └── uncracked/        (10000+ images)
    ├── pavements/
    │   ├── cracked/          (1500+ images)
    │   └── uncracked/        (10000+ images)
    └── walls/
        ├── cracked/          (1500+ images)
        └── uncracked/        (10000+ images)

Output Structure (YOLO Classification Format):
    data/processed/
    ├── train/
    │   ├── deck_cracked/
    │   ├── deck_uncracked/
    │   ├── pavement_cracked/
    │   ├── pavement_uncracked/
    │   ├── wall_cracked/
    │   └── wall_uncracked/
    ├── val/
    │   └── (same structure)
    └── test/
        └── (same structure)

Usage:
    python scripts/prepare_data.py --raw data/raw/ --output data/processed/
    python scripts/prepare_data.py --raw data/raw/ --split 0.7 0.15 0.15 --seed 42
    python scripts/prepare_data.py --raw data/raw/ --balance --max-per-class 2000
"""

import argparse
import logging
import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenCV (cv2) not installed. Image dimension reading will be skipped.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mapping from folder structure to class names
STRUCTURE_TYPES = ['deck', 'pavements', 'walls']
CONDITIONS = ['cracked', 'uncracked']
#: Raw folders that feed the optional out-of-scope "z_other" abstain class.
#: Any image found under them (recursively, any depth) belongs to it.
OTHER_FOLDERS = {'other', 'others', 'unknown'}


def get_class_name(structure: str, condition: str) -> str:
    """
    Generate standardized class name.
    
    Args:
        structure: Structure type (deck, pavements, walls)
        condition: Condition (cracked, uncracked)
        
    Returns:
        Standardized class name (e.g., 'deck_cracked', 'pavement_uncracked')
    """
    # Normalize plural structure names (pavements -> pavement, walls -> wall)
    structure = structure.lower().rstrip('s')
    condition = condition.lower()
    return f"{structure}_{condition}"


def discover_raw_data(raw_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover all images in raw directory organized by structure/condition.
    
    Args:
        raw_dir: Path to raw data directory
        
    Returns:
        Dictionary mapping class_name -> list of image paths
        
    Raises:
        ValueError: If raw directory structure is invalid
    """
    if not raw_dir.exists():
        raise ValueError(f"Raw directory not found: {raw_dir}")
    
    class_images = defaultdict(list)
    
    for structure_dir in raw_dir.iterdir():
        if not structure_dir.is_dir():
            continue

        structure_name = structure_dir.name.lower()

        # Optional abstain class: flat or nested image dump.
        if structure_name in OTHER_FOLDERS:
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
            for ext in image_extensions:
                class_images['z_other'].extend(structure_dir.rglob(ext))
            logger.info(
                "Found %d out-of-scope images for class: z_other",
                len(class_images['z_other']),
            )
            continue

        # Check if it's a valid structure type
        if structure_name not in [s.lower() for s in STRUCTURE_TYPES]:
            logger.warning(f"Skipping unknown directory: {structure_name}")
            continue
        
        # Look for condition subdirectories
        for condition_dir in structure_dir.iterdir():
            if not condition_dir.is_dir():
                continue
            
            condition_name = condition_dir.name.lower()
            
            if condition_name not in [c.lower() for c in CONDITIONS]:
                logger.warning(f"Skipping unknown condition: {condition_name}")
                continue
            
            class_name = get_class_name(structure_name, condition_name)
            
            # Find all images
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
            for ext in image_extensions:
                class_images[class_name].extend(condition_dir.glob(ext))
            
            logger.info(f"Found {len(class_images[class_name])} images for class: {class_name}")
    
    if not class_images:
        raise ValueError(
            f"No valid images found in {raw_dir}. "
            f"Expected structure: raw/<deck|pavements|walls>/<cracked|uncracked>/"
        )
    
    return dict(class_images)


def balance_classes(
    class_images: Dict[str, List[Path]],
    max_per_class: Optional[int] = None,
    min_per_class: Optional[int] = None
) -> Dict[str, List[Path]]:
    """
    Balance dataset by limiting or augmenting class sizes.
    
    Args:
        class_images: Dictionary of class -> image paths
        max_per_class: Maximum images per class (None = no limit)
        min_per_class: Minimum images per class (None = no minimum)
        
    Returns:
        Balanced class_images dictionary
    """
    balanced = {}
    
    for class_name, images in class_images.items():
        if max_per_class and len(images) > max_per_class:
            # Randomly sample to max_per_class
            balanced[class_name] = random.sample(images, max_per_class)
            logger.info(f"Balanced {class_name}: {len(images)} -> {max_per_class}")
        elif min_per_class and len(images) < min_per_class:
            # Oversample by repeating images
            n_repeat = (min_per_class // len(images)) + 1
            repeated = (images * n_repeat)[:min_per_class]
            balanced[class_name] = repeated
            logger.info(f"Oversampled {class_name}: {len(images)} -> {min_per_class}")
        else:
            balanced[class_name] = images
    
    return balanced


def split_dataset(
    images: List[Path],
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split images into train/val/test sets, grouped by surface ID.

    SDNET2018 tiles are named ``<surface-ID>-<frame>.jpg``: every frame of
    one ID is a crop of the SAME surface. Random per-image splits leak those
    near-duplicates across train/val/test and inflate accuracy while the
    model still fails on real photos. This split assigns whole groups of
    frames to one split; images without a ``-`` form their own group.

    Args:
        images: List of image paths.
        split_ratios: (train, val, test) ratios.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_images, val_images, test_images)
    """
    groups: Dict[str, List[Path]] = defaultdict(list)
    for img in images:
        stem = img.stem
        # Undo copy_images de-duplication suffixes: "001-114_1" -> "001-114".
        if "_" in stem:
            prefix, suffix = stem.rsplit("_", 1)
            if suffix.isdigit():
                stem = prefix
        groups[stem.split("-")[0]].append(img)

    random.seed(seed)
    group_keys = sorted(groups)
    random.shuffle(group_keys)

    n_total = len(group_keys)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])

    train, val, test = [], [], []
    for i, key in enumerate(group_keys):
        bucket = train if i < n_train else val if i < n_train + n_val else test
        bucket.extend(groups[key])

    logger.info(
        "Grouped split: %d surface IDs -> train %d / val %d / test %d "
        "(leak-free: no surface crosses splits)",
        n_total, n_train, n_val, n_total - n_train - n_val,
    )
    return train, val, test


def copy_images(
    images: List[Path],
    dest_dir: Path,
    class_name: str
) -> int:
    """
    Copy images to destination directory.
    
    Args:
        images: List of image paths to copy
        dest_dir: Destination directory
        class_name: Class name (subfolder)
        
    Returns:
        Number of images copied
    """
    class_dir = dest_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    for img_path in images:
        dest_path = class_dir / img_path.name
        # Handle duplicate names
        counter = 1
        while dest_path.exists():
            dest_path = class_dir / f"{img_path.stem}_{counter}{img_path.suffix}"
            counter += 1
        
        shutil.copy2(img_path, dest_path)
        copied += 1
    
    return copied


def create_dataset_statistics(
    class_images: Dict[str, List[Path]],
    output_dir: Path
) -> Dict:
    """
    Generate and save dataset statistics.
    
    Args:
        class_images: Dictionary of all class images
        output_dir: Output directory
        
    Returns:
        Statistics dictionary
    """
    stats = {
        'total_images': sum(len(imgs) for imgs in class_images.values()),
        'total_classes': len(class_images),
        'classes': {}
    }
    
    for class_name, images in sorted(class_images.items()):
        # Get image dimensions from first image (if cv2 available)
        h, w = 0, 0
        if CV2_AVAILABLE:
            sample_img = cv2.imread(str(images[0]))
            if sample_img is not None:
                h, w = sample_img.shape[:2]
        
        stats['classes'][class_name] = {
            'count': len(images),
            'percentage': round(len(images) / stats['total_images'] * 100, 2),
            'sample_dimensions': [w, h]
        }
    
    # Save statistics
    import json
    stats_path = output_dir / 'dataset_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Dataset statistics saved to {stats_path}")
    return stats


def print_summary(
    split_stats: Dict[str, Dict[str, int]],
    output_dir: Path
):
    """
    Print formatted summary of dataset preparation.
    
    Args:
        split_stats: Dictionary of split -> class -> count
        output_dir: Output directory
    """
    print("\n" + "=" * 70)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 70)
    print(f"Output Directory: {output_dir}")
    print("-" * 70)
    
    for split_name, classes in split_stats.items():
        total = sum(classes.values())
        print(f"\n{split_name.upper()} Split:")
        print(f"  Total Images: {total}")
        for class_name, count in sorted(classes.items()):
            print(f"    {class_name:25s}: {count:5d} images")
    
    print("\n" + "=" * 70)
    print("Dataset preparation complete!")
    print(f"You can now train with: python src/train.py --data data/processed")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare YOLO Classification Dataset for SHM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (80/10/10 split)
  python scripts/prepare_data.py --raw data/raw/ --output data/processed/
  
  # Custom split ratios
  python scripts/prepare_data.py --raw data/raw/ --split 0.7 0.15 0.15
  
  # Balance classes (limit to 2000 per class)
  python scripts/prepare_data.py --raw data/raw/ --balance --max-per-class 2000
  
  # Oversample minority classes
  python scripts/prepare_data.py --raw data/raw/ --balance --min-per-class 3000
  
  # Different seed for reproducibility
  python scripts/prepare_data.py --raw data/raw/ --seed 123
        """
    )
    
    parser.add_argument('--raw', type=str, default='data/raw',
                       help='Raw data directory path')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory for processed dataset')
    parser.add_argument('--split', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                       help='Train/val/test split ratios (must sum to 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--balance', action='store_true',
                       help='Balance classes by limiting or oversampling')
    parser.add_argument('--max-per-class', type=int, default=None,
                       help='Maximum images per class (for balancing)')
    parser.add_argument('--min-per-class', type=int, default=None,
                       help='Minimum images per class (for oversampling)')
    parser.add_argument('--clean', action='store_true',
                       help='Clean output directory before processing')
    
    args = parser.parse_args()

    # Seed before balancing so class sampling is reproducible too
    # (split_dataset re-seeds with the same value before splitting).
    random.seed(args.seed)

    # Validate split ratios
    if abs(sum(args.split) - 1.0) > 0.001:
        parser.error(f"Split ratios must sum to 1.0, got: {sum(args.split)}")
    
    raw_dir = Path(args.raw)
    output_dir = Path(args.output)
    
    # Clean output directory if requested
    if args.clean and output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Discover raw data
    logger.info("Step 1: Discovering raw data...")
    class_images = discover_raw_data(raw_dir)
    
    # Step 2: Balance classes if requested
    if args.balance:
        logger.info("Step 2: Balancing classes...")
        class_images = balance_classes(
            class_images,
            max_per_class=args.max_per_class,
            min_per_class=args.min_per_class
        )
    
    # Step 3: Split and copy
    logger.info("Step 3: Splitting and copying images...")
    split_stats = {'train': defaultdict(int), 'val': defaultdict(int), 'test': defaultdict(int)}
    
    for class_name, images in class_images.items():
        logger.info(f"Processing class: {class_name} ({len(images)} images)")
        
        # Split
        train_imgs, val_imgs, test_imgs = split_dataset(
            images,
            split_ratios=tuple(args.split),
            seed=args.seed
        )
        
        # Copy to respective directories
        train_copied = copy_images(train_imgs, output_dir / 'train', class_name)
        val_copied = copy_images(val_imgs, output_dir / 'val', class_name)
        test_copied = copy_images(test_imgs, output_dir / 'test', class_name)
        
        split_stats['train'][class_name] = train_copied
        split_stats['val'][class_name] = val_copied
        split_stats['test'][class_name] = test_copied
        
        logger.info(f"  Train: {train_copied}, Val: {val_copied}, Test: {test_copied}")
    
    # Step 4: Generate statistics
    logger.info("Step 4: Generating statistics...")
    stats = create_dataset_statistics(class_images, output_dir)
    
    # Print summary
    print_summary(split_stats, output_dir)
    
    # Validate dataset
    logger.info("Validating dataset...")
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        logger.info(f"  {split}: {len(class_dirs)} classes found")
    
    logger.info("Dataset preparation completed successfully!")


if __name__ == '__main__':
    main()
