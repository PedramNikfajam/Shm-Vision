"""
SHM Vision - Structural Health Monitoring with Computer Vision
==============================================================

AI-driven civil infrastructure inspection: YOLOv8 classification, physical
crack measurement via photogrammetry, and ACI 224R-01 / Eurocode 2 based
safety assessment.

Submodules:
    config: Domain constants and severity logic (single source of truth).
    crack_measurement: CV pipeline estimating crack geometry in mm.
    eng_utils: Engineering assessment helpers and fleet rollups.
    inference: CLI classification pipeline (image/batch/video/stream).
    train: YOLOv8 training/validation/export CLI.

Note:
    Modules use flat intra-package imports (``from config import ...``) so
    each file is also runnable as a script. Import the package with ``src/``
    on ``sys.path`` (as ``app.py`` does), or run files directly.
"""

__version__ = "2.0.0"
__author__ = "SHM Vision Team"
__description__ = "AI-Driven Structural Health Monitoring System"

from .config import (  # noqa: F401
    DamageSeverity,
    assess_severity_from_confidence,
    assess_severity_from_width,
    get_recommendation,
    get_safety_factor,
    get_structure_thresholds,
)

__all__ = [
    "DamageSeverity",
    "assess_severity_from_width",
    "assess_severity_from_confidence",
    "get_safety_factor",
    "get_recommendation",
    "get_structure_thresholds",
]
