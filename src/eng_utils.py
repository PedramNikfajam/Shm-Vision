#!/usr/bin/env python3
"""
Civil Engineering Utilities for Structural Health Monitoring
============================================================

Pure-python engineering helpers: photogrammetric unit conversion, severity
assessment, camera-parameter calculation, and fleet-level condition rollups.
All domain constants live in :mod:`config` (single source of truth).

Both classification mode (structure type + condition) and measurement mode
(physical widths) are supported via :func:`assess_structure_condition`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from config import (  # noqa: F401  (re-exported for backward compatibility)
    DamageSeverity,
    STRUCTURE_THRESHOLDS,
    get_safety_factor,
    get_recommendation,
    assess_severity_from_width,
    assess_severity_from_confidence,
    get_structure_thresholds,
)

# Module logger (no basicConfig here: the entrypoint owns logging setup).
logger = logging.getLogger(__name__)


class DamageType(Enum):
    """Types of structural damage."""

    CRACK = "crack"
    SPALLING = "spalling"
    CORROSION = "corrosion"
    DELAMINATION = "delamination"


class StructureType(Enum):
    """Types of civil infrastructure elements."""

    DECK = "deck"
    PAVEMENT = "pavement"
    WALL = "wall"
    BRIDGE = "bridge"
    COLUMN = "column"
    BEAM = "beam"


@dataclass(frozen=True)
class DamageAssessment:
    """Result of assessing one defect instance (measurement mode)."""

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


@dataclass(frozen=True)
class StructureAssessment:
    """Result of assessing one element from a classification result."""

    structure_type: StructureType
    condition: str  # 'cracked' | 'uncracked'
    confidence: float
    severity: Optional[DamageSeverity]
    recommendation: str
    safety_factor: float
    critical_threshold_mm: float
    max_allowable_mm: float


def pixels_to_mm(
    pixel_width: float,
    distance_mm: float,
    focal_length_px: float,
) -> float:
    """
    Convert a pixel measurement to millimeters using the pinhole model.

    Formula:
        W_mm = W_px * (D_mm / f_px)

    Args:
        pixel_width: Measured length in pixels (>= 0).
        distance_mm: Camera-to-surface distance in millimeters (> 0).
        focal_length_px: Effective focal length in pixels (> 0).

    Returns:
        Real-world width in millimeters, rounded to 2 decimals.

    Raises:
        TypeError: If any argument is not numeric.
        ValueError: If focal length / distance are not strictly positive or
            the pixel width is negative.

    Example:
        >>> pixels_to_mm(pixel_width=50, distance_mm=2000, focal_length_px=800)
        125.0
    """
    if not all(
        isinstance(x, (int, float)) and not isinstance(x, bool)
        for x in (pixel_width, distance_mm, focal_length_px)
    ):
        raise TypeError("All inputs must be numeric (int or float)")

    if focal_length_px <= 0:
        raise ValueError(f"Focal length must be strictly positive. Got: {focal_length_px}")
    if distance_mm <= 0:
        raise ValueError(f"Distance must be strictly positive. Got: {distance_mm}")
    if pixel_width < 0:
        raise ValueError(f"Pixel width cannot be negative. Got: {pixel_width}")

    real_width_mm = pixel_width * (distance_mm / focal_length_px)
    logger.debug(
        "Converted %.2fpx @ %.0fmm (f=%.1fpx) -> %.3fmm",
        pixel_width, distance_mm, focal_length_px, real_width_mm,
    )
    return round(real_width_mm, 2)


def pixels_to_mm_batch(measurements: List[Dict[str, float]]) -> List[Optional[float]]:
    """
    Batch-convert pixel measurements; per-item errors become ``None``.

    Args:
        measurements: Dicts with keys ``pixel_width``, ``distance_mm``,
            ``focal_length_px``.

    Returns:
        Millimeter values aligned with the input (None where invalid).
    """
    results: List[Optional[float]] = []
    for i, meas in enumerate(measurements):
        try:
            results.append(
                pixels_to_mm(
                    meas["pixel_width"], meas["distance_mm"], meas["focal_length_px"]
                )
            )
        except (ValueError, TypeError, KeyError) as exc:
            logger.error("Error processing measurement %d: %s", i, exc)
            results.append(None)
    return results


def calculate_crack_severity(
    width_mm: float,
    length_mm: Optional[float] = None,
    structure_type: Optional[str] = None,
) -> Tuple[DamageSeverity, str]:
    """
    Assess crack severity (optionally structure-aware) and return a verdict.

    Args:
        width_mm: Crack width in millimeters.
        length_mm: Optional length used to add a monitoring note for long
            (>= 500 mm) minor/moderate cracks.
        structure_type: Optional structure key (deck, pavement, ...). When
            given, element-specific limits apply.

    Returns:
        ``(severity, recommendation)`` tuple.
    """
    severity = assess_severity_from_width(width_mm, structure_type)
    recommendation = get_recommendation(severity)

    if (
        length_mm is not None
        and length_mm > 500
        and severity in (DamageSeverity.MINOR, DamageSeverity.MODERATE)
    ):
        recommendation += (
            f" Note: Crack length ({length_mm:.1f}mm) requires monitoring."
        )
    return severity, recommendation


def estimate_spalling_area(
    pixel_width: float,
    pixel_height: float,
    distance_mm: float,
    focal_length_px: float,
) -> Tuple[float, DamageSeverity, str]:
    """
    Convert a spalling patch from pixels to an area and grade severity.

    Args:
        pixel_width: Patch width in pixels.
        pixel_height: Patch height in pixels.
        distance_mm: Camera distance in millimeters.
        focal_length_px: Focal length in pixels.

    Returns:
        ``(area_mm2, severity, recommendation)`` tuple.

    Raises:
        ValueError: If any pixel dimension is negative.
    """
    if pixel_width < 0 or pixel_height < 0:
        raise ValueError("Pixel dimensions cannot be negative")

    real_width = pixels_to_mm(pixel_width, distance_mm, focal_length_px)
    real_height = pixels_to_mm(pixel_height, distance_mm, focal_length_px)
    area_mm2 = real_width * real_height

    # Spalling ladder (mm^2), ACI 201.1R-guided indicative limits.
    ladder = [
        (DamageSeverity.MINOR, 100.0),
        (DamageSeverity.MODERATE, 500.0),
        (DamageSeverity.SEVERE, 2000.0),
        (DamageSeverity.CRITICAL, 5000.0),
    ]
    severity = DamageSeverity.COLLAPSE_RISK
    for sev, limit in ladder:
        if area_mm2 < limit:
            severity = sev
            break

    return area_mm2, severity, get_recommendation(severity)


def assess_structure_condition(
    structure_type: str,
    condition: str,
    confidence: float = 1.0,
    physical_width_mm: Optional[float] = None,
) -> StructureAssessment:
    """
    Assess one element from a classification outcome, optionally enriched
    with a physical crack-width measurement.

    Priority: physical width (mm) drives severity when available; otherwise a
    conservative confidence-based fallback is used.

    Args:
        structure_type: Element key (deck, pavement, wall, bridge, ...).
        condition: ``'cracked'`` or ``'uncracked'``.
        confidence: Classification confidence in [0, 1].
        physical_width_mm: Optional measured crack width in millimeters.

    Returns:
        :class:`StructureAssessment` with severity, safety factor, thresholds
        and a human-readable recommendation.
    """
    try:
        struct_enum = StructureType((structure_type or "").lower())
    except ValueError:
        struct_enum = StructureType.WALL
        logger.warning(
            "Unknown structure type '%s', using wall defaults", structure_type
        )

    thresholds = get_structure_thresholds(structure_type)

    if (condition or "").lower() == "uncracked":
        severity: Optional[DamageSeverity] = DamageSeverity.NONE
        safety_factor = 1.0
        recommendation = (
            f"{structure_type.upper()}: No damage detected (confidence: {confidence:.1%}). "
            f"Continue routine inspection schedule. "
            f"Next inspection recommended in 12 months."
        )
    else:
        if physical_width_mm is not None and physical_width_mm > 0:
            severity = assess_severity_from_width(physical_width_mm, structure_type)
            measurement_msg = f"Physical measurement: {physical_width_mm:.3f}mm. "
        else:
            severity = assess_severity_from_confidence(confidence, is_damaged=True)
            measurement_msg = "No physical measurement available. "

        safety_factor = get_safety_factor(severity)
        recommendation = (
            f"{structure_type.upper()} CRACK DETECTED (confidence: {confidence:.1%}). "
            f"{get_recommendation(severity)} "
            f"Critical threshold: {thresholds.critical_width}mm. "
            f"Max allowable: {thresholds.max_allowable}mm. "
            f"{measurement_msg}"
        )

    return StructureAssessment(
        structure_type=struct_enum,
        condition=(condition or "").lower(),
        confidence=confidence,
        severity=severity,
        recommendation=recommendation,
        safety_factor=safety_factor,
        critical_threshold_mm=thresholds.critical_width,
        max_allowable_mm=thresholds.max_allowable,
    )


#: Severity -> health score (0-100) used by fleet rollups.
SEVERITY_SCORES: Dict[DamageSeverity, float] = {
    DamageSeverity.NONE: 100,
    DamageSeverity.MINOR: 90,
    DamageSeverity.MODERATE: 70,
    DamageSeverity.SEVERE: 40,
    DamageSeverity.CRITICAL: 10,
    DamageSeverity.COLLAPSE_RISK: 0,
    DamageSeverity.UNCERTAIN: 80,
}


def structural_safety_rating(
    assessments: List[DamageAssessment],
) -> Dict:
    """
    Aggregate per-defect assessments into a fleet safety report.

    Scoring is mean-of-scores over defect severities.

    Args:
        assessments: List of :class:`DamageAssessment`.

    Returns:
        Dict with ``overall_rating`` (SAFE/CAUTION/UNSAFE/DANGER),
        ``safety_score``, per-severity counts, and prioritized actions.
    """
    if not assessments:
        return {
            "overall_rating": "SAFE",
            "safety_score": 100.0,
            "critical_count": 0,
            "severe_count": 0,
            "total_defects": 0,
            "recommendation": get_recommendation(DamageSeverity.NONE),
            "priority_actions": [],
        }

    total_score = 0.0
    critical_count = 0
    severe_count = 0
    priority_actions: List[str] = []

    for assessment in assessments:
        total_score += SEVERITY_SCORES.get(assessment.severity, 50)
        if assessment.severity == DamageSeverity.CRITICAL:
            critical_count += 1
            priority_actions.append(
                f"CRITICAL: {assessment.damage_type.value} - {assessment.recommendation}"
            )
        elif assessment.severity == DamageSeverity.SEVERE:
            severe_count += 1
            priority_actions.append(
                f"SEVERE: {assessment.damage_type.value} - {assessment.recommendation}"
            )

    avg_score = total_score / len(assessments)

    if critical_count:
        overall = "DANGER"
    elif avg_score >= 90:
        overall = "SAFE"
    elif avg_score >= 70:
        overall = "CAUTION"
    elif avg_score >= 40:
        overall = "UNSAFE"
    else:
        overall = "DANGER"

    worst = (
        DamageSeverity.CRITICAL if critical_count
        else DamageSeverity.SEVERE if severe_count
        else DamageSeverity.MODERATE
    )
    return {
        "overall_rating": overall,
        "safety_score": round(avg_score, 1),
        "critical_count": critical_count,
        "severe_count": severe_count,
        "total_defects": len(assessments),
        "recommendation": get_recommendation(worst),
        "priority_actions": priority_actions,
    }


def batch_structure_assessment(classifications: List[Dict]) -> Dict:
    """
    Roll up multiple classification outcomes into one condition report.

    Args:
        classifications: Dicts with ``structure_type``, ``condition``,
            optional ``confidence`` and optional ``physical_width_mm``.

    Returns:
        Dict with overall rating, average safety factor, damage percentage
        and a summary recommendation.
    """
    if not classifications:
        return {
            "overall_rating": "SAFE",
            "average_safety_factor": 1.0,
            "total_structures": 0,
            "damaged_structures": 0,
            "healthy_structures": 0,
            "damage_percentage": 0.0,
            "recommendation": get_recommendation(DamageSeverity.NONE),
        }

    assessments = [
        assess_structure_condition(
            c["structure_type"],
            c["condition"],
            c.get("confidence", 1.0),
            c.get("physical_width_mm"),
        )
        for c in classifications
    ]
    n = len(assessments)
    damaged = sum(1 for a in assessments if a.condition == "cracked")
    avg_safety = sum(a.safety_factor for a in assessments) / n

    if avg_safety >= 0.9:
        rating = "SAFE"
    elif avg_safety >= 0.7:
        rating = "CAUTION"
    elif avg_safety >= 0.4:
        rating = "UNSAFE"
    else:
        rating = "DANGER"

    return {
        "overall_rating": rating,
        "average_safety_factor": round(avg_safety, 2),
        "total_structures": n,
        "damaged_structures": damaged,
        "healthy_structures": n - damaged,
        "damage_percentage": round(damaged / n * 100, 1),
        "recommendation": {
            "DANGER": (
                "EMERGENCY: Multiple critical damages detected. "
                "Immediate action required."
            ),
            "UNSAFE": "Multiple damages detected. Schedule repairs within 30 days.",
            "CAUTION": "Some damages detected. Monitor and schedule inspections.",
            "SAFE": "Structures in good condition. Continue routine monitoring.",
        }[rating],
    }


def get_camera_parameters(
    sensor_width_mm: float,
    focal_length_mm: float,
    image_width_px: int,
) -> float:
    """
    Convert physical lens specs to an effective pixel focal length.

    Args:
        sensor_width_mm: Sensor width in millimeters (> 0).
        focal_length_mm: Lens focal length in millimeters (> 0).
        image_width_px: Image width in pixels (> 0).

    Returns:
        Focal length in pixels (rounded to 2 decimals).

    Raises:
        ValueError: If any parameter is not strictly positive.

    Example:
        >>> get_camera_parameters(sensor_width_mm=36, focal_length_mm=50, image_width_px=1920)
        2666.67
    """
    if sensor_width_mm <= 0 or focal_length_mm <= 0 or image_width_px <= 0:
        raise ValueError("All camera parameters must be positive")

    return round((image_width_px * focal_length_mm) / sensor_width_mm, 2)


def estimate_distance_from_known_object(
    known_width_mm: float,
    pixel_width: float,
    focal_length_px: float,
) -> float:
    """
    Estimate camera distance from a reference object of known real width.

    Args:
        known_width_mm: Real-world width of the reference object (mm).
        pixel_width: Its projected width in pixels (> 0).
        focal_length_px: Focal length in pixels (> 0).

    Returns:
        Estimated distance in millimeters.

    Raises:
        ValueError: If pixel width / focal length are not strictly positive.
    """
    if known_width_mm <= 0:
        raise ValueError("known_width_mm must be positive")
    if pixel_width <= 0:
        raise ValueError("Pixel width must be positive")
    if focal_length_px <= 0:
        raise ValueError("Focal length must be strictly positive")

    return round((known_width_mm * focal_length_px) / pixel_width, 2)


def calculate_inspection_interval(severity: DamageSeverity) -> int:
    """
    Recommended re-inspection interval in months for a severity level.

    Args:
        severity: Damage severity level.

    Returns:
        Interval in months (0 == immediate action required).
    """
    intervals = {
        DamageSeverity.NONE: 12,
        DamageSeverity.MINOR: 12,
        DamageSeverity.MODERATE: 3,
        DamageSeverity.SEVERE: 1,
        DamageSeverity.CRITICAL: 0,
        DamageSeverity.COLLAPSE_RISK: 0,
        DamageSeverity.UNCERTAIN: 1,
    }
    return intervals.get(severity, 6)


if __name__ == "__main__":
    # Lightweight self-check; kept in-repo so `python src/eng_utils.py`
    # verifies the engineering core without any test framework.
    assert abs(pixels_to_mm(50, 2000, 800) - 125.0) < 1e-9
    assert abs(get_camera_parameters(36, 50, 1920) - 2666.67) < 0.01
    assert abs(
        estimate_distance_from_known_object(200.0, 80.0, 800.0) - 2000.0
    ) < 0.01

    # Structure-aware severity: same width grades differently per element.
    assert assess_severity_from_width(0.2, "pavement") == DamageSeverity.MINOR
    assert assess_severity_from_width(0.2, "deck") == DamageSeverity.MODERATE
    assert assess_severity_from_width(2.0, "deck") == DamageSeverity.CRITICAL
    assert assess_severity_from_width(4.0, "deck") == DamageSeverity.COLLAPSE_RISK

    sev, rec = calculate_crack_severity(0.5, structure_type="deck")
    assert sev == DamageSeverity.SEVERE
    assert "Repair required" in rec

    deck = assess_structure_condition("deck", "cracked", 0.92)
    assert deck.severity == DamageSeverity.MODERATE  # conservative fallback
    assert deck.safety_factor == get_safety_factor(DamageSeverity.MODERATE)

    uncracked = assess_structure_condition("pavement", "uncracked", 0.98)
    assert uncracked.severity == DamageSeverity.NONE

    report = batch_structure_assessment(
        [
            {"structure_type": "deck", "condition": "cracked", "confidence": 0.95},
            {"structure_type": "wall", "condition": "uncracked", "confidence": 0.88},
            {"structure_type": "pavement", "condition": "cracked", "confidence": 0.82},
        ]
    )
    assert report["total_structures"] == 3
    assert 0.0 <= report["average_safety_factor"] <= 1.0

    print("eng_utils self-check passed")
