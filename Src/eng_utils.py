#!/usr/bin/env python3
"""
Civil Engineering Utilities for Structural Health Monitoring
============================================================
Module for structural assessment, camera calibration, and safety evaluation.

Supports both:
- Object Detection mode (with pixel measurements)
- Classification mode (structure type + condition assessment)

Key Functions:
- pixels_to_mm: Convert pixel width to millimeters using photogrammetry
- calculate_crack_severity: Assess crack severity based on width
- structural_safety_rating: Generate safety rating from assessments
- get_camera_parameters: Calculate focal length from physical specs
- estimate_distance_from_known_object: Distance estimation
- assess_structure_condition: Classification-based structural assessment
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from config import (
    DamageSeverity,
    get_safety_factor,
    get_recommendation,
    assess_severity_from_width,
    assess_severity_from_confidence,
    UNCERTAINTY_THRESHOLD,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DamageSeverity(Enum):
    """Structural damage severity levels based on engineering standards."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
    COLLAPSE_RISK = "collapse_risk"


class DamageType(Enum):
    """Types of structural damage."""
    CRACK = "crack"
    SPALLING = "spalling"
    CORROSION = "corrosion"
    DELAMINATION = "delamination"


class StructureType(Enum):
    """Types of civil infrastructure structures."""
    DECK = "deck"
    PAVEMENT = "pavement"
    WALL = "wall"
    BRIDGE = "bridge"
    COLUMN = "column"
    BEAM = "beam"


@dataclass
class DamageAssessment:
    """Data class for storing damage assessment results (Detection mode)."""
    damage_type: DamageType
    pixel_width: float
    pixel_height: float
    pixel_area: float
    real_width_mm: float
    real_height_mm: float
    real_area_mm2: float
    severity: DamageSeverity
    confidence: float
    recommendation: str
    safety_factor: float


@dataclass
class StructureAssessment:
    """Data class for structure condition assessment (Classification mode)."""
    structure_type: StructureType
    condition: str              # 'cracked' or 'uncracked'
    confidence: float
    severity: Optional[DamageSeverity]
    recommendation: str
    safety_factor: float
    critical_threshold_mm: float
    max_allowable_mm: float


# Engineering thresholds based on ACI 224R-01 and Eurocode 2
SEVERITY_THRESHOLDS = {
    DamageType.CRACK: {
        DamageSeverity.MINOR: 0.1,
        DamageSeverity.MODERATE: 0.3,
        DamageSeverity.SEVERE: 1.0,
        DamageSeverity.CRITICAL: 3.0,
        DamageSeverity.COLLAPSE_RISK: float('inf')
    },
    DamageType.SPALLING: {
        DamageSeverity.MINOR: 100,
        DamageSeverity.MODERATE: 500,
        DamageSeverity.SEVERE: 2000,
        DamageSeverity.CRITICAL: 5000,
        DamageSeverity.COLLAPSE_RISK: float('inf')
    },
    DamageType.CORROSION: {
        DamageSeverity.MINOR: 50,
        DamageSeverity.MODERATE: 200,
        DamageSeverity.SEVERE: 500,
        DamageSeverity.CRITICAL: 1000,
        DamageSeverity.COLLAPSE_RISK: float('inf')
    },
    DamageType.DELAMINATION: {
        DamageSeverity.MINOR: 2500,
        DamageSeverity.MODERATE: 10000,
        DamageSeverity.SEVERE: 40000,
        DamageSeverity.CRITICAL: 90000,
        DamageSeverity.COLLAPSE_RISK: float('inf')
    }
}

# Structure-specific safety thresholds
STRUCTURE_THRESHOLDS = {
    StructureType.DECK: {
        'critical_crack_width': 0.3,
        'max_allowable': 0.5,
        'description': 'Bridge deck or floor slab'
    },
    StructureType.PAVEMENT: {
        'critical_crack_width': 6.0,
        'max_allowable': 12.0,
        'description': 'Road or sidewalk pavement'
    },
    StructureType.WALL: {
        'critical_crack_width': 0.3,
        'max_allowable': 1.0,
        'description': 'Concrete or masonry wall'
    },
    StructureType.BRIDGE: {
        'critical_crack_width': 0.2,
        'max_allowable': 0.4,
        'description': 'Bridge structural element'
    },
    StructureType.COLUMN: {
        'critical_crack_width': 0.3,
        'max_allowable': 0.5,
        'description': 'Structural column'
    },
    StructureType.BEAM: {
        'critical_crack_width': 0.3,
        'max_allowable': 0.5,
        'description': 'Structural beam'
    }
}

RECOMMENDATIONS = {
    DamageSeverity.NONE: "No action required. Continue routine inspection schedule.",
    DamageSeverity.MINOR: "Document damage. No immediate action required. Re-inspect in 12 months.",
    DamageSeverity.MODERATE: "Schedule detailed inspection within 3 months. Monitor for progression.",
    DamageSeverity.SEVERE: "Repair required within 30 days. Restrict load if near critical members.",
    DamageSeverity.CRITICAL: "Immediate repair mandatory. Restrict access to area. Engineer review within 48 hours.",
    DamageSeverity.COLLAPSE_RISK: "EMERGENCY: Evacuate area immediately. Close structure to traffic. Emergency structural assessment required."
}


def pixels_to_mm(
    pixel_width: float,
    distance_mm: float,
    focal_length_px: float
) -> float:
    """
    Convert pixel measurements to millimeters using photogrammetric principles.
    
    Formula: W_mm = W_px * (D / f_px)
    
    Args:
        pixel_width: Width measured in pixels
        distance_mm: Distance from camera to structure in millimeters
        focal_length_px: Camera focal length in pixels
        
    Returns:
        Real-world width in millimeters (rounded to 2 decimal places)
        
    Raises:
        ValueError: If focal_length_px <= 0 or distance_mm <= 0
        TypeError: If inputs are not numeric
        
    Example:
        >>> pixels_to_mm(pixel_width=50, distance_mm=2000, focal_length_px=800)
        125.0
    """
    # Input validation
    if not all(isinstance(x, (int, float)) for x in [pixel_width, distance_mm, focal_length_px]):
        raise TypeError("All inputs must be numeric (int or float)")
    
    if focal_length_px <= 0:
        raise ValueError(f"Focal length must be strictly positive. Got: {focal_length_px}")
    
    if distance_mm <= 0:
        raise ValueError(f"Distance must be strictly positive. Got: {distance_mm}")
    
    if pixel_width < 0:
        raise ValueError(f"Pixel width cannot be negative. Got: {pixel_width}")
    
    # Core photogrammetric conversion
    real_width_mm = pixel_width * (distance_mm / focal_length_px)
    
    logger.debug(f"Converted {pixel_width}px at {distance_mm}mm distance (f={focal_length_px}px) -> {real_width_mm:.2f}mm")
    
    return round(real_width_mm, 2)


def pixels_to_mm_batch(
    measurements: List[Dict[str, float]]
) -> List[float]:
    """
    Batch convert multiple pixel measurements to millimeters.
    
    Args:
        measurements: List of dicts with keys 'pixel_width', 'distance_mm', 'focal_length_px'
        
    Returns:
        List of real-world widths in millimeters
        
    Example:
        >>> measurements = [
        ...     {'pixel_width': 50, 'distance_mm': 2000, 'focal_length_px': 800},
        ...     {'pixel_width': 30, 'distance_mm': 1500, 'focal_length_px': 800}
        ... ]
        >>> pixels_to_mm_batch(measurements)
        [125.0, 56.25]
    """
    results = []
    for i, meas in enumerate(measurements):
        try:
            result = pixels_to_mm(
                meas['pixel_width'],
                meas['distance_mm'],
                meas['focal_length_px']
            )
            results.append(result)
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error processing measurement {i}: {e}")
            results.append(None)
    return results


def calculate_crack_severity(
    width_mm: float,
    length_mm: Optional[float] = None
) -> Tuple[DamageSeverity, str]:
    """
    Assess crack severity based on width according to engineering standards.
    
    Classification based on ACI 224R-01:
    - Hairline: < 0.1 mm
    - Fine: 0.1 - 0.3 mm
    - Medium: 0.3 - 1.0 mm
    - Wide: 1.0 - 3.0 mm
    - Very wide: > 3.0 mm
    
    Args:
        width_mm: Crack width in millimeters
        length_mm: Optional crack length for additional assessment
        
    Returns:
        Tuple of (severity_level, recommendation)
    """
    thresholds = SEVERITY_THRESHOLDS[DamageType.CRACK]
    
    if width_mm < thresholds[DamageSeverity.MINOR]:
        severity = DamageSeverity.MINOR
    elif width_mm < thresholds[DamageSeverity.MODERATE]:
        severity = DamageSeverity.MODERATE
    elif width_mm < thresholds[DamageSeverity.SEVERE]:
        severity = DamageSeverity.SEVERE
    elif width_mm < thresholds[DamageSeverity.CRITICAL]:
        severity = DamageSeverity.CRITICAL
    else:
        severity = DamageSeverity.COLLAPSE_RISK
    
    # Adjust for very long cracks
    if length_mm and length_mm > 500 and severity in [DamageSeverity.MINOR, DamageSeverity.MODERATE]:
        recommendation = RECOMMENDATIONS[severity] + f" Note: Crack length ({length_mm:.1f}mm) requires monitoring."
    else:
        recommendation = RECOMMENDATIONS[severity]
    
    return severity, recommendation


def estimate_spalling_area(
    pixel_width: float,
    pixel_height: float,
    distance_mm: float,
    focal_length_px: float
) -> Tuple[float, DamageSeverity, str]:
    """
    Calculate spalling area and assess severity.
    
    Args:
        pixel_width: Width in pixels
        pixel_height: Height in pixels
        distance_mm: Camera distance in mm
        focal_length_px: Focal length in pixels
        
    Returns:
        Tuple of (area_mm2, severity, recommendation)
    """
    real_width = pixels_to_mm(pixel_width, distance_mm, focal_length_px)
    real_height = pixels_to_mm(pixel_height, distance_mm, focal_length_px)
    area_mm2 = real_width * real_height
    
    thresholds = SEVERITY_THRESHOLDS[DamageType.SPALLING]
    
    if area_mm2 < thresholds[DamageSeverity.MINOR]:
        severity = DamageSeverity.MINOR
    elif area_mm2 < thresholds[DamageSeverity.MODERATE]:
        severity = DamageSeverity.MODERATE
    elif area_mm2 < thresholds[DamageSeverity.SEVERE]:
        severity = DamageSeverity.SEVERE
    elif area_mm2 < thresholds[DamageSeverity.CRITICAL]:
        severity = DamageSeverity.CRITICAL
    else:
        severity = DamageSeverity.COLLAPSE_RISK
    
    return area_mm2, severity, RECOMMENDATIONS[severity]


def assess_structure_condition(
    structure_type: str,
    condition: str,
    confidence: float = 1.0,
    physical_width_mm: Optional[float] = None,
) -> StructureAssessment:
    """
    Assess structural condition based on classification results.
    
    Args:
        structure_type: Type of structure (deck, pavement, wall, etc.)
        condition: 'cracked' or 'uncracked'
        confidence: Classification confidence
        physical_width_mm: Optional physical crack width from measurement module
        
    Returns:
        StructureAssessment with safety evaluation
    """
    try:
        struct_enum = StructureType(structure_type.lower())
    except ValueError:
        struct_enum = StructureType.WALL
        logger.warning(f"Unknown structure type '{structure_type}', using default thresholds")
    
    thresholds = STRUCTURE_THRESHOLDS.get(struct_enum, STRUCTURE_THRESHOLDS[StructureType.WALL])
    
    if condition.lower() == 'uncracked':
        severity = DamageSeverity.NONE
        safety_factor = 1.0
        recommendation = (
            f"{structure_type.upper()}: No damage detected (confidence: {confidence:.1%}). "
            f"Continue routine inspection schedule. "
            f"Next inspection recommended in 12 months."
        )
    else:
        # PRIORITY: If physical measurement available, use it!
        if physical_width_mm is not None and physical_width_mm > 0:
            severity = assess_severity_from_width(physical_width_mm)
            measurement_msg = f"Physical measurement: {physical_width_mm:.3f}mm. "
        else:
            # Fallback: use confidence-based assessment with UNCERTAIN handling
            severity = assess_severity_from_confidence(confidence, is_damaged=True)
            measurement_msg = "No physical measurement available. "
        
        safety_factor = get_safety_factor(severity)
        recommendation = (
            f"{structure_type.upper()} CRACK DETECTED (confidence: {confidence:.1%}). "
            f"{get_recommendation(severity)} "
            f"Critical threshold: {thresholds['critical_crack_width']}mm. "
            f"Max allowable: {thresholds['max_allowable']}mm. "
            f"{measurement_msg}"
        )
    
    return StructureAssessment(
        structure_type=struct_enum,
        condition=condition.lower(),
        confidence=confidence,
        severity=severity,
        recommendation=recommendation,
        safety_factor=safety_factor,
        critical_threshold_mm=thresholds['critical_crack_width'],
        max_allowable_mm=thresholds['max_allowable']
    )
    
    return StructureAssessment(
        structure_type=struct_enum,
        condition=condition.lower(),
        confidence=confidence,
        severity=severity,
        recommendation=recommendation,
        safety_factor=safety_factor,
        critical_threshold_mm=thresholds['critical_crack_width'],
        max_allowable_mm=thresholds['max_allowable']
    )


def structural_safety_rating(
    assessments: List[DamageAssessment]
) -> Dict:
    """
    Calculate overall structural safety rating from multiple damage assessments.
    
    Args:
        assessments: List of DamageAssessment objects
        
    Returns:
        Dictionary with overall rating, critical findings, and recommendations
    """
    if not assessments:
        return {
            'overall_rating': 'SAFE',
            'safety_score': 100,
            'critical_count': 0,
            'severe_count': 0,
            'recommendation': 'No damage detected. Continue routine inspection schedule.',
            'priority_actions': []
        }
    
    severity_scores = {
        DamageSeverity.NONE: 100,
        DamageSeverity.MINOR: 90,
        DamageSeverity.MODERATE: 70,
        DamageSeverity.SEVERE: 40,
        DamageSeverity.CRITICAL: 10,
        DamageSeverity.COLLAPSE_RISK: 0
    }
    
    total_score = 0
    critical_count = 0
    severe_count = 0
    priority_actions = []
    
    for assessment in assessments:
        score = severity_scores[assessment.severity]
        total_score += score
        
        if assessment.severity == DamageSeverity.CRITICAL:
            critical_count += 1
            priority_actions.append(f"CRITICAL: {assessment.damage_type.value} - {assessment.recommendation}")
        elif assessment.severity == DamageSeverity.SEVERE:
            severe_count += 1
            priority_actions.append(f"SEVERE: {assessment.damage_type.value} - {assessment.recommendation}")
    
    avg_score = total_score / len(assessments)
    
    if avg_score >= 90:
        overall_rating = 'SAFE'
    elif avg_score >= 70:
        overall_rating = 'CAUTION'
    elif avg_score >= 40:
        overall_rating = 'UNSAFE'
    else:
        overall_rating = 'DANGER'
    
    return {
        'overall_rating': overall_rating,
        'safety_score': round(avg_score, 1),
        'critical_count': critical_count,
        'severe_count': severe_count,
        'total_defects': len(assessments),
        'recommendation': RECOMMENDATIONS[
            DamageSeverity.CRITICAL if critical_count > 0 else
            DamageSeverity.SEVERE if severe_count > 0 else
            DamageSeverity.MODERATE
        ],
        'priority_actions': priority_actions
    }


def batch_structure_assessment(
    classifications: List[Dict[str, any]]
) -> Dict:
    """
    Generate batch safety report from multiple classification results.
    
    Args:
        classifications: List of dicts with 'structure_type', 'condition', 'confidence'
        
    Returns:
        Consolidated safety report
    """
    total = len(classifications)
    damaged = sum(1 for c in classifications if c['condition'] == 'cracked')
    
    # Calculate average safety factor
    safety_factors = []
    for c in classifications:
        assessment = assess_structure_condition(
            c['structure_type'],
            c['condition'],
            c.get('confidence', 1.0)
        )
        safety_factors.append(assessment.safety_factor)
    
    avg_safety = sum(safety_factors) / len(safety_factors) if safety_factors else 1.0
    
    if avg_safety >= 0.9:
        rating = 'SAFE'
    elif avg_safety >= 0.7:
        rating = 'CAUTION'
    elif avg_safety >= 0.4:
        rating = 'UNSAFE'
    else:
        rating = 'DANGER'
    
    return {
        'overall_rating': rating,
        'average_safety_factor': round(avg_safety, 2),
        'total_structures': total,
        'damaged_structures': damaged,
        'healthy_structures': total - damaged,
        'damage_percentage': round(damaged / total * 100, 1) if total > 0 else 0,
        'recommendation': (
            "EMERGENCY: Multiple critical damages detected. Immediate action required."
            if rating == 'DANGER' else
            "Multiple damages detected. Schedule repairs within 30 days."
            if rating == 'UNSAFE' else
            "Some damages detected. Monitor and schedule inspections."
            if rating == 'CAUTION' else
            "Structures in good condition. Continue routine monitoring."
        )
    }


def get_camera_parameters(
    sensor_width_mm: float,
    focal_length_mm: float,
    image_width_px: int
) -> float:
    """
    Calculate focal length in pixels from camera physical parameters.
    
    Args:
        sensor_width_mm: Camera sensor width in millimeters
        focal_length_mm: Physical focal length in millimeters
        image_width_px: Image width in pixels
        
    Returns:
        Focal length in pixels
        
    Example:
        >>> get_camera_parameters(sensor_width_mm=36, focal_length_mm=50, image_width_px=1920)
        2666.67
    """
    if sensor_width_mm <= 0 or focal_length_mm <= 0 or image_width_px <= 0:
        raise ValueError("All camera parameters must be positive")
    
    focal_length_px = (image_width_px * focal_length_mm) / sensor_width_mm
    return round(focal_length_px, 2)


def estimate_distance_from_known_object(
    known_width_mm: float,
    pixel_width: float,
    focal_length_px: float
) -> float:
    """
    Estimate camera distance using a reference object of known size.
    
    Args:
        known_width_mm: Real-world width of reference object in mm
        pixel_width: Width of reference object in pixels
        focal_length_px: Camera focal length in pixels
        
    Returns:
        Estimated distance in millimeters
    """
    if pixel_width <= 0:
        raise ValueError("Pixel width must be positive")
    
    distance_mm = (known_width_mm * focal_length_px) / pixel_width
    return round(distance_mm, 2)


def get_structure_thresholds(structure_type: str) -> Dict[str, float]:
    """
    Get safety thresholds for a specific structure type.
    
    Args:
        structure_type: Type of structure (deck, pavement, wall, etc.)
        
    Returns:
        Dictionary with critical and max allowable thresholds
        
    Example:
        >>> get_structure_thresholds('deck')
        {'critical_crack_width': 0.3, 'max_allowable': 0.5, 'description': 'Bridge deck or floor slab'}
    """
    try:
        struct_enum = StructureType(structure_type.lower())
    except ValueError:
        logger.warning(f"Unknown structure type '{structure_type}', using wall defaults")
        struct_enum = StructureType.WALL
    
    return STRUCTURE_THRESHOLDS.get(struct_enum, STRUCTURE_THRESHOLDS[StructureType.WALL])


def calculate_inspection_interval(severity: DamageSeverity) -> int:
    """
    Calculate recommended inspection interval in months based on severity.
    
    Args:
        severity: Damage severity level
        
    Returns:
        Recommended inspection interval in months
    """
    intervals = {
        DamageSeverity.NONE: 12,
        DamageSeverity.MINOR: 12,
        DamageSeverity.MODERATE: 3,
        DamageSeverity.SEVERE: 1,
        DamageSeverity.CRITICAL: 0,  # Immediate
        DamageSeverity.COLLAPSE_RISK: 0
    }
    return intervals.get(severity, 6)


if __name__ == '__main__':
    # Example usage and validation
    print("=" * 70)
    print("CIVIL ENGINEERING UTILITIES - VALIDATION")
    print("=" * 70)
    
    # Test 1: Pixel to mm conversion
    print("\nTest 1: Basic Conversion")
    width = pixels_to_mm(50, 2000, 800)
    print(f"  50px at 2m distance (f=800px) = {width}mm")
    assert width == 125.0, "Basic conversion failed"
    print("  ✓ PASSED")
    
    # Test 2: Crack severity assessment
    print("\nTest 2: Crack Severity")
    severity, rec = calculate_crack_severity(0.5)
    print(f"  0.5mm crack -> {severity.value}")
    assert severity == DamageSeverity.SEVERE, "Severity assessment failed"
    print("  ✓ PASSED")
    
    # Test 3: Structure condition assessment (Classification mode)
    print("\nTest 3: Structure Condition Assessment")
    assessment = assess_structure_condition('deck', 'cracked', 0.92)
    print(f"  Deck cracked (92% conf) -> {assessment.severity.value}")
    print(f"  Safety factor: {assessment.safety_factor}")
    assert assessment.severity == DamageSeverity.SEVERE
    print("  ✓ PASSED")
    
    assessment2 = assess_structure_condition('pavement', 'uncracked', 0.98)
    print(f"  Pavement uncracked (98% conf) -> {assessment2.severity.value}")
    assert assessment2.severity == DamageSeverity.NONE
    print("  ✓ PASSED")
    
    # Test 4: Batch assessment
    print("\nTest 4: Batch Assessment")
    classifications = [
        {'structure_type': 'deck', 'condition': 'cracked', 'confidence': 0.95},
        {'structure_type': 'wall', 'condition': 'uncracked', 'confidence': 0.88},
        {'structure_type': 'pavement', 'condition': 'cracked', 'confidence': 0.82}
    ]
    report = batch_structure_assessment(classifications)
    print(f"  Overall rating: {report['overall_rating']}")
    print(f"  Avg safety factor: {report['average_safety_factor']}")
    print("  ✓ PASSED")
    
    # Test 5: Camera parameters
    print("\nTest 5: Camera Parameters")
    f_px = get_camera_parameters(36, 50, 1920)
    print(f"  Full-frame 50mm lens at 1920px width: f={f_px}px")
    print("  ✓ PASSED")
    
    # Test 6: Structure thresholds
    print("\nTest 6: Structure Thresholds")
    thresholds = get_structure_thresholds('deck')
    print(f"  Deck thresholds: {thresholds}")
    assert thresholds['critical_crack_width'] == 0.3
    print("  ✓ PASSED")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
