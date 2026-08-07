"""
SHM Vision - Structural Health Monitoring with Computer Vision
===============================================================

AI-driven civil infrastructure inspection system for detecting and 
measuring structural damage using deep learning.

Modules:
    eng_utils: Civil engineering calculations and safety assessments
    train: YOLOv8 model training pipeline
    inference: Damage detection and structural analysis
"""

__version__ = "1.0.0"
__author__ = "SHM Vision Team"
__description__ = "AI-Driven Structural Health Monitoring System"

from .eng_utils import (
    pixels_to_mm,
    pixels_to_mm_batch,
    calculate_crack_severity,
    estimate_spalling_area,
    structural_safety_rating,
    get_camera_parameters,
    estimate_distance_from_known_object,
    DamageAssessment,
    DamageSeverity,
    DamageType
)

__all__ = [
    'pixels_to_mm',
    'pixels_to_mm_batch',
    'calculate_crack_severity',
    'estimate_spalling_area',
    'structural_safety_rating',
    'get_camera_parameters',
    'estimate_distance_from_known_object',
    'DamageAssessment',
    'DamageSeverity',
    'DamageType'
]
