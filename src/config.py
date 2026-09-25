#!/usr/bin/env python3
"""
Centralized Configuration for SHM Vision
========================================

Single source of truth for every domain constant used across the project
(`app.py`, `src/inference.py`, `src/crack_measurement.py`, `src/eng_utils.py`,
`scripts/evaluate.py`). All other modules MUST import from here instead of
redefining class maps, thresholds, or recommendations locally.

Engineering references:
    - ACI 224R-01 "Control of Cracking in Concrete Structures"
    - EN 1992-1-1 (Eurocode 2), Table 7.1N crack width limits
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

# ------------------------------------------------------------------
# Severity taxonomy
# ------------------------------------------------------------------


class DamageSeverity(str, Enum):
    """Structural damage severity levels (ordered by increasing risk)."""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
    COLLAPSE_RISK = "collapse_risk"
    UNCERTAIN = "uncertain"


#: Monotonic risk order used for "worst-of" aggregations.
SEVERITY_ORDER: Tuple[DamageSeverity, ...] = (
    DamageSeverity.NONE,
    DamageSeverity.MINOR,
    DamageSeverity.MODERATE,
    DamageSeverity.SEVERE,
    DamageSeverity.CRITICAL,
    DamageSeverity.COLLAPSE_RISK,
)


def worst_severity(a: DamageSeverity, b: DamageSeverity) -> DamageSeverity:
    """Return the higher-risk of two severities (UNCERTAIN compares lowest)."""
    if a == DamageSeverity.UNCERTAIN:
        return b
    if b == DamageSeverity.UNCERTAIN:
        return a
    return a if SEVERITY_ORDER.index(a) >= SEVERITY_ORDER.index(b) else b


# ------------------------------------------------------------------
# Classification confidence thresholds
# ------------------------------------------------------------------

#: Below this confidence the prediction is flagged UNCERTAIN and must not be
#: used for load-restriction decisions on its own.
UNCERTAINTY_THRESHOLD = 0.75

#: Confidence tiers, ordered high -> low. The lowest tier's boundary is
#: ``UNCERTAINTY_THRESHOLD`` itself, so "low confidence" and "flagged
#: UNCERTAIN" are the *same* set of predictions by construction. The previous
#: table declared a ``low`` tier at 0.60 while the UNCERTAIN gate sat at 0.75,
#: leaving 0.60-0.75 labelled "low" yet graded as a confident MINOR crack.
CONFIDENCE_THRESHOLDS = {
    "high": 0.85,
    "medium": UNCERTAINTY_THRESHOLD,
}

#: A surface the classifier confidently calls cracked is at least this severe.
#: Measurement may escalate severity above it, but a thin measured crack must
#: never be reported as healthier than an unmeasured one -- previously a
#: measured 0.05 mm wall crack graded MINOR (SF 0.85) while the *same* crack
#: left unmeasured graded MODERATE (SF 0.65), so measuring improved the
#: safety factor.
MIN_DETECTED_CRACK_SEVERITY = DamageSeverity.MINOR


def confidence_tier(confidence: float) -> str:
    """
    Bucket a confidence score into a display tier.

    Args:
        confidence: Top-1 softmax probability in [0, 1].

    Returns:
        ``'high'``, ``'medium'``, or ``'low'``. ``'low'`` is returned exactly
        when :func:`assess_severity_from_confidence` would return UNCERTAIN,
        so the label and the severity can never disagree.
    """
    if confidence >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    if confidence >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"

# ------------------------------------------------------------------
# Safety factors per severity (dimensionless residual-capacity proxy)
# ------------------------------------------------------------------

SAFETY_FACTORS: Dict[DamageSeverity, float] = {
    DamageSeverity.NONE: 1.00,
    DamageSeverity.MINOR: 0.85,
    DamageSeverity.MODERATE: 0.65,
    DamageSeverity.SEVERE: 0.40,
    DamageSeverity.CRITICAL: 0.20,
    DamageSeverity.COLLAPSE_RISK: 0.00,
    DamageSeverity.UNCERTAIN: 0.90,  # slightly reduced due to model uncertainty
}

# ------------------------------------------------------------------
# Generic (structure-agnostic) crack-width severity thresholds in mm,
# per ACI 224R-01 reasonable design crack widths.
# Used only when the structure type is unknown.
# ------------------------------------------------------------------

PHYSICAL_SEVERITY_THRESHOLDS_MM: Dict[DamageSeverity, float] = {
    DamageSeverity.MINOR: 0.10,
    DamageSeverity.MODERATE: 0.30,
    DamageSeverity.SEVERE: 1.00,
    DamageSeverity.CRITICAL: 3.00,
    DamageSeverity.COLLAPSE_RISK: float("inf"),
}

# ------------------------------------------------------------------
# Structure-specific safety thresholds (mm).
# `critical_width`: width above which serviceability is compromised.
# `max_allowable`:  absolute design limit per ACI 224R-01 Table 4.
# ------------------------------------------------------------------


@dataclass(frozen=True)
class StructureThresholds:
    """Per-structure crack-width limits (mm) and metadata."""

    critical_width: float
    max_allowable: float
    description: str


STRUCTURE_THRESHOLDS: Dict[str, StructureThresholds] = {
    "deck": StructureThresholds(0.3, 0.5, "Bridge deck or floor slab"),
    "pavement": StructureThresholds(6.0, 12.0, "Road or sidewalk pavement"),
    "wall": StructureThresholds(0.3, 1.0, "Concrete or masonry wall"),
    "bridge": StructureThresholds(0.2, 0.4, "Bridge structural element"),
    "column": StructureThresholds(0.3, 0.5, "Structural column"),
    "beam": StructureThresholds(0.3, 0.5, "Structural beam"),
}

# ------------------------------------------------------------------
# Recommendations per severity (actionability-ordered)
# ------------------------------------------------------------------

RECOMMENDATIONS: Dict[DamageSeverity, str] = {
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
        "EMERGENCY: Restrict access to the area immediately. "
        "Close the structure to use. Emergency structural assessment required."
    ),
    DamageSeverity.UNCERTAIN: (
        "Model confidence is low. Manual inspection and physical measurement required. "
        "Do not rely solely on AI assessment."
    ),
}

# ------------------------------------------------------------------
# Classification class mappings (single definition -- do not duplicate)
# ------------------------------------------------------------------

CLASS_NAMES: Dict[int, str] = {
    0: "deck_cracked",
    1: "deck_uncracked",
    2: "pavement_cracked",
    3: "pavement_uncracked",
    4: "wall_cracked",
    5: "wall_uncracked",
    # Optional 7th "abstain" class: non-concrete / out-of-scope surfaces.
    # The "z_" prefix keeps it sorting AFTER wall_* so YOLO's alphabetical
    # folder->id mapping preserves ids 0-5 whether the class is trained or
    # not (6-class models simply never predict id 6).
    6: "z_other",
}

#: Out-of-scope abstain class name (see CLASS_NAMES note).
OTHER_CLASS = "z_other"

STRUCTURE_MAP: Dict[str, str] = {
    name: name.split("_", 1)[0] for name in CLASS_NAMES.values()
}

CONDITION_MAP: Dict[str, str] = {
    name: ("cracked" if name.endswith("_cracked") else "uncracked")
    for name in CLASS_NAMES.values()
}

#: BGR annotation colors per class (used by inference.py HUD).
CLASS_COLORS: Dict[str, tuple] = {
    **{name: (0, 0, 255) for name in CLASS_NAMES.values() if name.endswith("_cracked")},
    **{name: (0, 255, 0) for name in CLASS_NAMES.values() if name.endswith("_uncracked")},
    OTHER_CLASS: (160, 160, 160),
}

# ------------------------------------------------------------------
# Camera / photogrammetry defaults
# ------------------------------------------------------------------

DEFAULT_CAMERA_DISTANCE_MM = 2000
DEFAULT_FOCAL_LENGTH_PX = 800

# ------------------------------------------------------------------
# Crack-measurement pipeline defaults
# ------------------------------------------------------------------

CRACK_MEASUREMENT = {
    "canny_low": 50,
    "canny_high": 150,
    "morph_kernel_size": 3,
    "min_crack_length_px": 30,
    "gaussian_blur_kernel": 5,
    #: Discard connected components whose area is below this fraction of the
    #: image -- they are specks of dirt, not cracks.
    "min_component_area_frac": 1e-4,
    #: A component is crack-like if its circularity 4*pi*A/P^2 stays below
    #: this. Thin/long structures score << 1; blobs approach 1. Replaces
    #: bounding-box elongation, which rejected diagonal/zigzag cracks.
    "max_circularity": 0.4,
    #: Whole-image gate: skeleton length / mean width must reach this.
    #: Cracks are long-and-thin; leftover texture clusters are chunky.
    "min_length_width_ratio": 6.0,
    #: Rejected when the surviving mask covers more than this fraction of
    #: the image: real cracks are sparse (<10%), sprawling noise/texture
    #: masks (random-noise test: ~55%) exceed it trivially.
    "max_mask_area_frac": 0.15,
    #: Scale-invariant width ceiling, expressed as a fraction of the image's
    #: shorter side. Replaces the old absolute ``max_mean_width_px = 6.0``,
    #: which silently rejected every crack wider than ~6 px -- catastrophic at
    #: close range, where photogrammetry is most accurate (a 12 mm pavement
    #: crack is 4.8 px at the default 2.5 mm/px but 120 px at 0.1 mm/px, and
    #: was rejected as "no crack detected"). A genuine surface crack is never
    #: a large fraction of the frame; a texture blob usually is.
    "max_mean_width_frac": 0.10,
    #: Minimum crack width, in pixels, that this pipeline can resolve. Tied to
    #: ``gaussian_blur_kernel`` (5x5): a structure narrower than roughly half
    #: the kernel is reshaped by the blur before it is ever thresholded, so its
    #: apparent width is set by the kernel rather than by the crack. The
    #: integrated-intensity estimator still returns a *plausible* number for
    #: such a feature (a 1 px bar came back as 7.01 mm at the default
    #: 2.5 mm/px), which is exactly why the floor must reject it explicitly
    #: instead of trusting the estimate. The floor is in pixels, so it is
    #: independent of the calibration: 1 px is 1 px however mm scale applies.
    "min_resolvable_width_px": 3.0,
    #: Reject the whole measurement when the mean width exceeds this many mm
    #: (physically implausible for a surface crack => texture false positive).
    "max_plausible_avg_width_mm": 50.0,
}

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


def get_safety_factor(severity: DamageSeverity) -> float:
    """Return residual-capacity proxy for a severity level."""
    return SAFETY_FACTORS.get(severity, 0.50)


def get_recommendation(severity: DamageSeverity) -> str:
    """Return the engineering action string for a severity level."""
    return RECOMMENDATIONS.get(severity, "Inspect structure.")


def get_structure_thresholds(structure_type: str) -> StructureThresholds:
    """
    Return crack-width limits for a structure type.

    Unknown types fall back to ``wall`` (conservative for superstructure
    elements: 0.3 mm critical) and are logged by the caller if needed.
    """
    key = (structure_type or "").lower()
    return STRUCTURE_THRESHOLDS.get(key, STRUCTURE_THRESHOLDS["wall"])


# ------------------------------------------------------------------
# Measurement-trust gate: photogrammetry needs enough image resolution
# per mm. With DEFAULT_FOCAL_LENGTH_PX=800 @ 2 m, 1 px = 2.5 mm -- a wall
# crack of 0.3 mm is 0.12 px, so any "measurement" there is texture.
# Only trust a measured width when the calibration can resolve it AND the
# classifier is confident the surface is cracked at all.
# ------------------------------------------------------------------


def max_trusted_mm_per_px(structure_type: Optional[str] = None) -> float:
    """
    Coarsest calibration (mm per pixel) at which a measured width may
    drive severity, per element type.

    Rule of thumb: one pixel must span at most 1/3 of the structure's
    ``max_allowable`` width, so a limit-scale crack covers several pixels.
    Unknown structure falls back to a strict 1 mm/px.

    Args:
        structure_type: Optional element key (deck/pavement/wall/...).

    Returns:
        Trusted mm/px ceiling (e.g. pavement: 12/3 = 4.0, wall: 1/3 = 0.33).
    """
    if structure_type:
        return get_structure_thresholds(structure_type).max_allowable / 3.0
    return 1.0


def can_escalate_severity(
    confidence: float, mm_per_px: float, structure_type: Optional[str] = None
) -> bool:
    """
    Decide whether a photogrammetric measurement may override severity.

    Measurement is geometry; classification is evidence. With a
    low-confidence "cracked" verdict or a calibration too coarse to
    resolve serviceability-level widths, the measured width is unreliable
    and must not drive SEVERE/CRITICAL/COLLAPSE_RISK verdicts on its own.

    Args:
        confidence: Top-1 classification confidence of the cracked label.
        mm_per_px: Physical size of one pixel in mm at the calibration.
        structure_type: Optional element key for the resolution ceiling.

    Returns:
        True when measurement-based severity escalation is allowed.
    """
    return (
        confidence >= UNCERTAINTY_THRESHOLD
        and mm_per_px <= max_trusted_mm_per_px(structure_type)
    )


def assess_severity_from_width(
    width_mm: float, structure_type: Optional[str] = None
) -> DamageSeverity:
    """
    Assess crack severity from a physical width measurement.

    When ``structure_type`` is provided, the structure-specific limits are
    used (e.g. 2 mm is routine for pavement but SEVERE for a deck); the
    generic ACI 224R-01 ladder is used otherwise.

    Args:
        width_mm: Measured crack width in millimeters (must be >= 0).
        structure_type: Optional structure key from ``STRUCTURE_THRESHOLDS``.

    Returns:
        The mapped :class:`DamageSeverity` (never ``UNCERTAIN``).

    Raises:
        ValueError: If ``width_mm`` is negative.
    """
    if width_mm < 0:
        raise ValueError(f"width_mm must be non-negative, got {width_mm}")

    if structure_type is None:
        ladder = PHYSICAL_SEVERITY_THRESHOLDS_MM
        upper = [
            (DamageSeverity.MINOR, ladder[DamageSeverity.MINOR]),
            (DamageSeverity.MODERATE, ladder[DamageSeverity.MODERATE]),
            (DamageSeverity.SEVERE, ladder[DamageSeverity.SEVERE]),
            (DamageSeverity.CRITICAL, ladder[DamageSeverity.CRITICAL]),
        ]
    else:
        th = get_structure_thresholds(structure_type)
        # Structure ladder anchored on the element-specific critical width,
        # scaled by the same 1x / 3.3x / 10x / 30x progression as the ACI
        # generic ladder relative to its 0.3 mm anchor.
        scale = th.critical_width / 0.3
        upper = [
            (DamageSeverity.MINOR, 0.10 * scale),
            (DamageSeverity.MODERATE, th.critical_width),
            (DamageSeverity.SEVERE, 1.00 * scale),
            (DamageSeverity.CRITICAL, 3.00 * scale),
        ]

    for severity, limit in upper:
        if width_mm < limit:
            return severity
    return DamageSeverity.COLLAPSE_RISK


def assess_severity_from_confidence(
    confidence: float, is_damaged: bool
) -> DamageSeverity:
    """
    Fallback severity when no physical measurement exists.

    Confidence expresses *certainty of the label*, not damage magnitude, so it
    must not grade severity: the previous implementation returned MINOR below
    0.85 and MODERATE at/above it, which meant a *more* confident detection
    produced a *worse* safety factor (0.85 -> SF 0.85, 0.86 -> SF 0.65) and
    put a cliff in the middle of the confidence range.

    A detected crack is therefore graded at a single conservative baseline
    (MINOR) whenever the classifier is confident, and never better. A
    trustworthy physical measurement may escalate from here, but measurement
    can never grade a detected crack as healthier than the baseline -- see
    :func:`worst_severity`, which callers use to keep the worse of the two.

    Args:
        confidence: Top-1 softmax probability in [0, 1].
        is_damaged: Whether the predicted class is a cracked class.

    Returns:
        NONE if undamaged; UNCERTAIN below ``UNCERTAINTY_THRESHOLD``;
        otherwise MINOR as the conservative baseline.
    """
    if not is_damaged:
        return DamageSeverity.NONE

    if confidence < UNCERTAINTY_THRESHOLD:
        return DamageSeverity.UNCERTAIN

    return DamageSeverity.MINOR
