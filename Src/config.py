#!/usr/bin/env python3
"""
Centralized Configuration for SHM Vision
========================================
All thresholds, safety factors, and engineering constants in one place
to ensure consistency across app.py, inference.py, and eng_utils.py.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional


class DamageSeverity(str, Enum):
    """Structural damage severity levels."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
    COLLAPSE_RISK = "collapse_risk"
    UNCERTAIN = "uncertain"


# ------------------------------------------------------------------
# Classification Confidence Thresholds
# ------------------------------------------------------------------
CONFIDENCE_THRESHOLDS = {
    "high": 0.85,
    "medium": 0.75,
    "low": 0.60,
}

# When confidence is below this, flag as uncertain
UNCERTAINTY_THRESHOLD = 0.75

# ------------------------------------------------------------------
# Safety Factors (consistent across all modules)
# ------------------------------------------------------------------
SAFETY_FACTORS = {
    DamageSeverity.NONE: 1.00,
    DamageSeverity.MINOR: 0.85,
    DamageSeverity.MODERATE: 0.65,
    DamageSeverity.SEVERE: 0.40,
    DamageSeverity.CRITICAL: 0.20,
    DamageSeverity.COLLAPSE_RISK: 0.00,
    DamageSeverity.UNCERTAIN: 0.90,  # Slightly reduced due to model uncertainty
}

# ------------------------------------------------------------------
# Crack Width Thresholds (mm) per ACI 224R-01 & Eurocode 2
# ------------------------------------------------------------------
# These are used when physical measurement IS available
PHYSICAL_SEVERITY_THRESHOLDS_MM = {
    DamageSeverity.MINOR: 0.10,
    DamageSeverity.MODERATE: 0.30,
    DamageSeverity.SEVERE: 1.00,
    DamageSeverity.CRITICAL: 3.00,
    DamageSeverity.COLLAPSE_RISK: float("inf"),
}

# ------------------------------------------------------------------
# Structure-Specific Safety Thresholds
# ------------------------------------------------------------------
STRUCTURE_THRESHOLDS = {
    "deck": {
        "critical_width": 0.3,
        "max_allowable": 0.5,
        "description": "Bridge deck or floor slab",
    },
    "pavement": {
        "critical_width": 6.0,
        "max_allowable": 12.0,
        "description": "Road or sidewalk pavement",
    },
    "wall": {
        "critical_width": 0.3,
        "max_allowable": 1.0,
        "description": "Concrete or masonry wall",
    },
}

# ------------------------------------------------------------------
# Recommendations per Severity
# ------------------------------------------------------------------
RECOMMENDATIONS = {
    DamageSeverity.NONE: (
        "No damage detected. Continue routine inspection schedule. "
        "Next inspection recommended in 12 months."
    ),
    DamageSeverity.MINOR: (
        "Document damage. No immediate action required. Re-inspect in 12 months. "
        "Monitor for any progression."
    ),
    DamageSeverity.MODERATE: (
        "Schedule detailed inspection within 3 months. Monitor for progression. "
        "Consider sealing cracks to prevent moisture ingress."
    ),
    DamageSeverity.SEVERE: (
        "Repair required within 30 days. Restrict load if near critical members. "
        "Engineer review recommended."
    ),
    DamageSeverity.CRITICAL: (
        "Immediate repair mandatory. Restrict access to area. "
        "Licensed engineer review within 48 hours."
    ),
    DamageSeverity.COLLAPSE_RISK: (
        "EMERGENCY: Evacuate area immediately. Close structure to traffic. "
        "Emergency structural assessment required."
    ),
    DamageSeverity.UNCERTAIN: (
        "Model confidence is low. Manual inspection and physical measurement required. "
        "Do not rely solely on AI assessment."
    ),
}

# ------------------------------------------------------------------
# Class Mappings
# ------------------------------------------------------------------
CLASS_NAMES = {
    0: "deck_cracked",
    1: "deck_uncracked",
    2: "pavement_cracked",
    3: "pavement_uncracked",
    4: "wall_cracked",
    5: "wall_uncracked",
}

STRUCTURE_MAP = {
    "deck_cracked": "deck",
    "deck_uncracked": "deck",
    "pavement_cracked": "pavement",
    "pavement_uncracked": "pavement",
    "wall_cracked": "wall",
    "wall_uncracked": "wall",
}

CONDITION_MAP = {
    "deck_cracked": "cracked",
    "deck_uncracked": "uncracked",
    "pavement_cracked": "cracked",
    "pavement_uncracked": "uncracked",
    "wall_cracked": "cracked",
    "wall_uncracked": "uncracked",
}

# ------------------------------------------------------------------
# Default Camera Parameters
# ------------------------------------------------------------------
DEFAULT_CAMERA_DISTANCE_MM = 2000
DEFAULT_FOCAL_LENGTH_PX = 800

# ------------------------------------------------------------------
# Crack Measurement Defaults
# ------------------------------------------------------------------
CRACK_MEASUREMENT = {
    "canny_low": 50,
    "canny_high": 150,
    "morph_kernel_size": 3,
    "min_crack_length_px": 30,
    "gaussian_blur_kernel": 5,
}


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def get_safety_factor(severity: DamageSeverity) -> float:
    """Get safety factor for a given severity level."""
    return SAFETY_FACTORS.get(severity, 0.50)


def get_recommendation(severity: DamageSeverity) -> str:
    """Get recommendation for a given severity level."""
    return RECOMMENDATIONS.get(severity, "Inspect structure.")


def assess_severity_from_width(width_mm: float) -> DamageSeverity:
    """
    Assess severity based on physical crack width (mm).
    Uses ACI 224R-01 thresholds.
    """
    if width_mm < PHYSICAL_SEVERITY_THRESHOLDS_MM[DamageSeverity.MINOR]:
        return DamageSeverity.MINOR
    elif width_mm < PHYSICAL_SEVERITY_THRESHOLDS_MM[DamageSeverity.MODERATE]:
        return DamageSeverity.MODERATE
    elif width_mm < PHYSICAL_SEVERITY_THRESHOLDS_MM[DamageSeverity.SEVERE]:
        return DamageSeverity.SEVERE
    elif width_mm < PHYSICAL_SEVERITY_THRESHOLDS_MM[DamageSeverity.CRITICAL]:
        return DamageSeverity.CRITICAL
    else:
        return DamageSeverity.COLLAPSE_RISK


def assess_severity_from_confidence(
    confidence: float, is_damaged: bool
) -> DamageSeverity:
    """
    Fallback severity assessment when physical measurement is NOT available.
    Uses confidence as a proxy for certainty, NOT severity.
    """
    if not is_damaged:
        return DamageSeverity.NONE

    if confidence < UNCERTAINTY_THRESHOLD:
        return DamageSeverity.UNCERTAIN

    # When we are confident damage exists but have no physical measurement,
    # we conservatively assume at least MODERATE until measured
    if confidence > 0.95:
        return DamageSeverity.MODERATE  # Was SEVERE - changed to be conservative
    elif confidence > 0.85:
        return DamageSeverity.MODERATE
    else:
        return DamageSeverity.MINOR


def get_structure_thresholds(structure_type: str) -> Dict[str, float]:
    """Get safety thresholds for a specific structure type."""
    return STRUCTURE_THRESHOLDS.get(structure_type, STRUCTURE_THRESHOLDS["wall"])
