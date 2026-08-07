#!/usr/bin/env python3
"""
Crack Measurement Module for SHM Vision
========================================
Estimates physical crack width from images using computer vision
and photogrammetric principles.

Workflow:
1. Preprocess image (grayscale, blur, CLAHE)
2. Detect crack edges (Canny + morphological operations)
3. Skeletonize to get crack centerline
4. Measure perpendicular width along skeleton
5. Convert pixels to mm using camera parameters

Usage:
    from crack_measurement import estimate_crack_width
    width_mm = estimate_crack_width(image, distance_mm=2000, focal_length_px=800)
"""

import logging
import math
from typing import Tuple, Optional, List, Dict

import cv2
import numpy as np

from config import (
    CRACK_MEASUREMENT,
    DEFAULT_CAMERA_DISTANCE_MM,
    DEFAULT_FOCAL_LENGTH_PX,
    assess_severity_from_width,
    DamageSeverity,
)

logger = logging.getLogger(__name__)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for crack detection.
    Enhances contrast and reduces noise.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Gaussian blur to reduce noise
    ksize = CRACK_MEASUREMENT["gaussian_blur_kernel"]
    blurred = cv2.GaussianBlur(enhanced, (ksize, ksize), 0)

    return blurred


def overlay_crack_mask(
    image: np.ndarray, mask: np.ndarray, color: tuple = (0, 0, 255), thickness: int = 2
) -> np.ndarray:
    """
    Draw crack mask on image with semi-transparent red overlay.

    Args:
        image: Original BGR image
        mask: Binary crack mask
        color: BGR color, default red (0, 0, 255)
        thickness: Line thickness in pixels

    Returns:
        Image with crack highlighted in red
    """
    result = image.copy()

    # Dilate mask slightly to make crack visible but not too thick
    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        thick_mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        thick_mask = mask

    # Semi-transparent overlay (alpha blend) - looks better and more accurate
    overlay = result.copy()
    overlay[thick_mask > 0] = color
    result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

    return result


def detect_crack_mask(preprocessed: np.ndarray) -> np.ndarray:
    """
    Detect crack regions using thresholding (cracks are dark).
    Returns a binary mask where crack pixels are 255.
    """
    # Invert: cracks (dark) become bright
    inverted = 255 - preprocessed

    # Otsu threshold to separate dark regions (crack) from background
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove very small noise blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Close small gaps inside crack
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Skeletonize binary mask to get 1-pixel wide centerline.
    Uses Zhang-Suen thinning algorithm via OpenCV ximgproc if available,
    otherwise falls back to iterative erosion.
    """
    # Try OpenCV contrib ximgproc if available
    try:
        from cv2 import ximgproc

        skeleton = ximgproc.thinning(mask)
        return skeleton
    except (ImportError, AttributeError):
        pass

    # Fallback: iterative thinning
    skeleton = np.zeros(mask.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = mask.copy()

    while True:
        eroded = cv2.erode(temp, element)
        dilated = cv2.dilate(eroded, element)
        temp_skel = cv2.subtract(temp, dilated)
        skeleton = cv2.bitwise_or(skeleton, temp_skel)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    return skeleton


def measure_crack_width_pixels(
    mask: np.ndarray, skeleton: np.ndarray
) -> Tuple[float, float, int]:
    """
    Measure crack width along the skeleton using distance transform.

    Returns:
        (max_width_px, avg_width_px, measured_points)
    """
    # Distance transform: each pixel value = distance to nearest background
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Get distances along skeleton points
    skeleton_coords = np.argwhere(skeleton > 0)

    if len(skeleton_coords) == 0:
        return 0.0, 0.0, 0

    widths = []
    for y, x in skeleton_coords:
        # Distance to edge is half the width at that point
        half_width = dist_transform[y, x]
        full_width = half_width * 2
        widths.append(full_width)

    if not widths:
        return 0.0, 0.0, 0

    widths_arr = np.array(widths)
    max_w = float(np.max(widths_arr))
    avg_w = float(np.mean(widths_arr))

    # Filter outliers for more robust average
    q75, q25 = np.percentile(widths_arr, [75, 25])
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    filtered = widths_arr[(widths_arr >= lower) & (widths_arr <= upper)]

    if len(filtered) > 0:
        avg_w = float(np.mean(filtered))

    return max_w, avg_w, len(widths)


def pixels_to_mm(
    pixel_width: float,
    distance_mm: float,
    focal_length_px: float,
) -> float:
    """
    Convert pixel width to millimeters using photogrammetry.
    Formula: W_mm = W_px * (D / f)
    """
    if focal_length_px <= 0 or distance_mm <= 0:
        raise ValueError("focal_length and distance must be positive")
    if pixel_width < 0:
        raise ValueError("pixel_width cannot be negative")

    real_width_mm = pixel_width * (distance_mm / focal_length_px)
    return round(real_width_mm, 3)


def estimate_crack_width(
    image: np.ndarray,
    distance_mm: float = DEFAULT_CAMERA_DISTANCE_MM,
    focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX,
    return_debug_info: bool = False,
) -> Dict:
    """
    Main function: estimate crack width from image.

    Args:
        image: Input image (BGR or grayscale)
        distance_mm: Distance from camera to structure in mm
        focal_length_px: Camera focal length in pixels
        return_debug_info: If True, returns debug images and measurements

    Returns:
        Dictionary with:
            - max_width_mm: Maximum estimated crack width
            - avg_width_mm: Average estimated crack width
            - severity: DamageSeverity based on physical width
            - safety_factor: Calculated safety factor
            - measured: Whether a crack was actually measured
            - debug_images: Optional dict with visualization images
    """
    try:
        # Preprocess
        preprocessed = preprocess_image(image)

        # Detect crack mask
        mask = detect_crack_mask(preprocessed)

        # Check if any crack was detected
        if cv2.countNonZero(mask) < CRACK_MEASUREMENT["min_crack_length_px"]:
            # Create red crack overlay on original image
            overlay = overlay_crack_mask(image, mask, color=(0, 0, 255), thickness=2)
            result = {
                "max_width_mm": 0.0,
                "avg_width_mm": 0.0,
                "severity": DamageSeverity.NONE,
                "safety_factor": 1.0,
                "measured": False,
                "message": "No crack detected in image for measurement.",
            }
            if return_debug_info:
                result["debug_images"] = {
                    "preprocessed": preprocessed,
                    "mask": mask,
                }
            return result

        # Skeletonize
        skeleton = skeletonize_mask(mask)

        # Measure width in pixels
        max_w_px, avg_w_px, num_points = measure_crack_width_pixels(mask, skeleton)

        # Convert to mm
        max_w_mm = pixels_to_mm(max_w_px, distance_mm, focal_length_px)
        avg_w_mm = pixels_to_mm(avg_w_px, distance_mm, focal_length_px)

        # Sanity check: if measured width is unrealistically large (>20mm),
        # it's likely noise/texture falsely detected as a crack
        if avg_w_mm > 50.0:
            return {
                "max_width_mm": max_w_mm,
                "avg_width_mm": avg_w_mm,
                "severity": DamageSeverity.UNCERTAIN,
                "safety_factor": get_safety_factor(DamageSeverity.UNCERTAIN),
                "measured": False,
                "overlay_image": None,
                "message": (
                    f"Measurement rejected: avg={avg_w_mm:.1f}mm exceeds realistic crack width. "
                    f"Likely noise/texture detected as crack."
                ),
            }
        # Assess severity based on physical width
        severity = assess_severity_from_width(avg_w_mm)
        from config import get_safety_factor

        safety_factor = get_safety_factor(severity)

        # Create red crack overlay on original image
        overlay = overlay_crack_mask(image, mask, color=(0, 0, 255), thickness=2)

        result = {
            "max_width_mm": max_w_mm,
            "avg_width_mm": avg_w_mm,
            "max_width_px": max_w_px,
            "avg_width_px": avg_w_px,
            "measured_points": num_points,
            "severity": severity,
            "safety_factor": safety_factor,
            "measured": True,
            "distance_mm": distance_mm,
            "focal_length_px": focal_length_px,
            "overlay_image": overlay,
            "message": (
                f"Crack measured: avg={avg_w_mm:.3f}mm, max={max_w_mm:.3f}mm "
                f"(based on {num_points} points)"
            ),
        }

        if return_debug_info:
            # Create visualization
            vis = (
                image.copy()
                if len(image.shape) == 3
                else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            )
            # Overlay mask in red
            overlay_vis = vis.copy()
            overlay_vis[mask > 0] = (0, 0, 255)
            vis = cv2.addWeighted(vis, 0.7, overlay_vis, 0.3, 0)
            # Draw skeleton in green
            vis[skeleton > 0] = (0, 255, 0)

            result["debug_images"] = {
                "preprocessed": preprocessed,
                "mask": mask,
                "skeleton": skeleton,
                "visualization": vis,
            }

        logger.info(result["message"])
        return result

    except Exception as e:
        logger.error(f"Crack measurement failed: {e}")
        return {
            "max_width_mm": 0.0,
            "avg_width_mm": 0.0,
            "severity": DamageSeverity.UNCERTAIN,
            "safety_factor": get_safety_factor(DamageSeverity.UNCERTAIN),
            "measured": False,
            "message": f"Measurement error: {str(e)}",
        }


def annotate_measurement(
    image: np.ndarray,
    measurement: Dict,
    x_offset: int = 20,
    y_offset: int = 140,
) -> np.ndarray:
    """
    Draw measurement results on image.
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Background panel
    overlay = img.copy()
    panel_h = 120
    cv2.rectangle(overlay, (0, y_offset - 20), (w, y_offset + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # Text
    color = (
        (0, 255, 0)
        if measurement["severity"] == DamageSeverity.NONE
        else (
            (0, 165, 255)
            if measurement["severity"]
            in [DamageSeverity.MINOR, DamageSeverity.MODERATE]
            else (0, 0, 255)
        )
    )

    cv2.putText(
        img,
        "PHYSICAL MEASUREMENT",
        (x_offset, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    if measurement["measured"]:
        line1 = f"Avg Width: {measurement['avg_width_mm']:.3f} mm"
        line2 = f"Max Width: {measurement['max_width_mm']:.3f} mm"
        line3 = f"Severity: {measurement['severity'].value.upper()} | SF: {measurement['safety_factor']:.2f}"
    else:
        line1 = measurement.get("message", "No measurement available")
        line2 = "Using classification-based estimate only."
        line3 = ""

    cv2.putText(
        img, line1, (x_offset, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )
    cv2.putText(
        img,
        line2,
        (x_offset, y_offset + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        img, line3, (x_offset, y_offset + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )

    return img


if __name__ == "__main__":
    # Simple test
    print("Crack Measurement Module - Validation")
    print("=" * 50)

    # Test pixels_to_mm
    w = pixels_to_mm(50, 2000, 800)
    print(f"50px @ 2m, f=800px -> {w} mm (expected: 125.0)")
    assert abs(w - 125.0) < 0.01

    # Test severity assessment
    sev = assess_severity_from_width(0.05)
    print(f"0.05mm crack -> {sev.value} (expected: minor)")
    assert sev == DamageSeverity.MINOR

    sev = assess_severity_from_width(0.5)
    print(f"0.5mm crack -> {sev.value} (expected: severe)")
    assert sev == DamageSeverity.SEVERE

    sev = assess_severity_from_width(5.0)
    print(f"5.0mm crack -> {sev.value} (expected: collapse_risk)")
    assert sev == DamageSeverity.COLLAPSE_RISK

    print("All tests passed!")
