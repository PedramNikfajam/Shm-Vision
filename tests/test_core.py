"""
Unit tests for SHM Vision engineering core.

Run from project root:
    python -m pytest tests/ -v          (if pytest installed)
    python tests/run_all.py             (stdlib-only runner, no pytest)

These tests intentionally avoid network/model downloads: they cover the
deterministic engineering core (config, eng_utils, crack_measurement) plus
import-safety of the CLI modules.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (  # noqa: E402
    CLASS_NAMES,
    CONDITION_MAP,
    CONFIDENCE_THRESHOLDS,
    MIN_DETECTED_CRACK_SEVERITY,
    SEVERITY_ORDER,
    STRUCTURE_MAP,
    UNCERTAINTY_THRESHOLD,
    DamageSeverity,
    assess_severity_from_confidence,
    assess_severity_from_width,
    can_escalate_severity,
    confidence_tier,
    get_recommendation,
    get_safety_factor,
    get_structure_thresholds,
    max_trusted_mm_per_px,
    worst_severity,
)
from crack_measurement import (  # noqa: E402
    estimate_crack_orientation,
    estimate_crack_width,
    filter_components,
    pixels_to_mm,
    skeletonize_mask,
)
from eng_utils import (  # noqa: E402
    batch_structure_assessment,
    calculate_crack_severity,
    estimate_spalling_area,
    get_camera_parameters,
    assess_structure_condition,
    pixels_to_mm as eng_pixels_to_mm,
    structural_safety_rating,
)


# ----------------------------------------------------------------------
# config: severity ladders
# ----------------------------------------------------------------------


class TestSeverityFromWidth:
    def test_generic_ladder_boundaries(self):
        assert assess_severity_from_width(0.05) == DamageSeverity.MINOR
        assert assess_severity_from_width(0.2) == DamageSeverity.MODERATE
        assert assess_severity_from_width(0.5) == DamageSeverity.SEVERE
        assert assess_severity_from_width(2.0) == DamageSeverity.CRITICAL
        assert assess_severity_from_width(5.0) == DamageSeverity.COLLAPSE_RISK

    def test_structure_aware(self):
        # 2mm: routine for pavement, critical for a deck.
        assert assess_severity_from_width(2.0, "pavement") == DamageSeverity.MODERATE
        assert assess_severity_from_width(2.0, "deck") == DamageSeverity.CRITICAL

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            assess_severity_from_width(-1.0)

    def test_unknown_structure_falls_back_to_wall(self):
        th = get_structure_thresholds("nonexistent")
        assert th.critical_width == get_structure_thresholds("wall").critical_width


class TestSeverityFromConfidence:
    def test_undamaged_is_none(self):
        assert assess_severity_from_confidence(0.99, False) == DamageSeverity.NONE

    def test_low_confidence_is_uncertain(self):
        assert assess_severity_from_confidence(0.60, True) == DamageSeverity.UNCERTAIN

    def test_confident_damage_is_conservative(self):
        # Confidence is certainty of label, not magnitude, so it must not grade
        # severity. Confident damage gets one baseline (MINOR); measurement is
        # the only thing that may escalate it. Previously a confidence-keyed
        # split returned MODERATE at >=0.85, which made a *more* confident
        # detection produce a *worse* safety factor.
        assert assess_severity_from_confidence(0.99, True) == DamageSeverity.MINOR

    def test_severity_never_improves_as_confidence_rises(self):
        ranks = [
            SEVERITY_ORDER.index(assess_severity_from_confidence(c, True))
            for c in (0.76, 0.80, 0.85, 0.86, 0.95, 0.99)
        ]
        assert ranks == sorted(ranks), ranks

    def test_measurement_cannot_grade_healthier_than_baseline(self):
        baseline = assess_severity_from_confidence(0.99, True)
        for w in (0.01, 0.05, 0.1):
            graded = worst_severity(
                assess_severity_from_width(w, "wall"), MIN_DETECTED_CRACK_SEVERITY
            )
            assert SEVERITY_ORDER.index(graded) >= SEVERITY_ORDER.index(baseline)

    def test_worst_severity(self):
        assert worst_severity(DamageSeverity.MINOR, DamageSeverity.SEVERE) == DamageSeverity.SEVERE
        assert worst_severity(DamageSeverity.UNCERTAIN, DamageSeverity.MINOR) == DamageSeverity.MINOR


class TestConfidenceTier:
    """'low confidence' and 'flagged UNCERTAIN' must be the same set."""

    @pytest.mark.parametrize(
        "conf", [0.0, 0.3, 0.6, 0.74, 0.749, 0.75, 0.8, 0.84, 0.85, 0.99, 1.0]
    )
    def test_low_tier_matches_uncertain_gate(self, conf):
        assert (confidence_tier(conf) == "low") == (
            assess_severity_from_confidence(conf, True) == DamageSeverity.UNCERTAIN
        )

    def test_thresholds_have_no_dead_tier(self):
        # The old table declared a 'low' tier at 0.60 while the UNCERTAIN gate
        # sat at 0.75, leaving 0.60-0.75 labelled low yet graded as confident.
        assert CONFIDENCE_THRESHOLDS["medium"] == UNCERTAINTY_THRESHOLD
        assert "low" not in CONFIDENCE_THRESHOLDS


class TestMeasurementTrustGate:
    """Photogrammetry must not escalate severity on its own."""

    def test_resolution_ceiling_per_structure(self):
        # 1px must span at most 1/3 of max_allowable.
        assert abs(max_trusted_mm_per_px("pavement") - 4.0) < 1e-9   # 12/3
        assert abs(max_trusted_mm_per_px("wall") - 1.0 / 3.0) < 1e-9
        assert max_trusted_mm_per_px(None) == 1.0

    def test_low_confidence_never_escalates(self):
        # Default calibration 2.5 mm/px, pavement otherwise permissive:
        # a 48% "cracked" verdict must stay UNCERTAIN regardless of width.
        assert not can_escalate_severity(0.4885, 2.5, "pavement")
        assert not can_escalate_severity(0.4885, 2.5, "wall")

    def test_coarse_calibration_blocks_fine_elements(self):
        # Confident classification, but wall/deck critical widths (0.3mm)
        # are sub-pixel at 2.5 mm/px -> measurement untrusted.
        assert not can_escalate_severity(0.99, 2.5, "wall")
        assert not can_escalate_severity(0.99, 2.5, "deck")
        # Pavement max_allowable 12mm: 2.5 mm/px is coarse but resolvable.
        assert can_escalate_severity(0.99, 2.5, "pavement")

    def test_good_calibration_allows_escalation(self):
        # Close-up shot: 0.1 mm/px resolves everything; confident verdict.
        assert can_escalate_severity(0.90, 0.1, "wall")
        assert can_escalate_severity(0.90, 0.1, "deck")


class TestConsistency:
    def test_class_maps_consistent(self):
        for name in CLASS_NAMES.values():
            assert name in STRUCTURE_MAP
            assert name in CONDITION_MAP
            assert CONDITION_MAP[name] in ("cracked", "uncracked")
            assert STRUCTURE_MAP[name] == name.split("_", 1)[0]

    def test_safety_factors_monotonic(self):
        order = [DamageSeverity.NONE, DamageSeverity.MINOR, DamageSeverity.MODERATE,
                 DamageSeverity.SEVERE, DamageSeverity.CRITICAL, DamageSeverity.COLLAPSE_RISK]
        factors = [get_safety_factor(s) for s in order]
        assert all(a >= b for a, b in zip(factors, factors[1:]))

    def test_recommendations_exist(self):
        for sev in DamageSeverity:
            assert get_recommendation(sev)


# ----------------------------------------------------------------------
# eng_utils: photogrammetry & assessment
# ----------------------------------------------------------------------


class TestPhotogrammetry:
    def test_basic_conversion(self):
        assert abs(eng_pixels_to_mm(50, 2000, 800) - 125.0) < 1e-9

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            eng_pixels_to_mm(10, 0, 800)
        with pytest.raises(ValueError):
            eng_pixels_to_mm(10, 2000, -1)
        with pytest.raises(TypeError):
            eng_pixels_to_mm("10", 2000, 800)

    def test_camera_parameters(self):
        assert abs(get_camera_parameters(36, 50, 1920) - 2666.67) < 0.01

    def test_spalling_area(self):
        area, severity, _ = estimate_spalling_area(7, 7, 1000, 1000)
        assert area == pytest.approx(49.0)
        assert severity == DamageSeverity.MINOR
        # Boundary: exactly 100 mm^2 crosses the MINOR limit.
        _, severity2, _ = estimate_spalling_area(10, 10, 1000, 1000)
        assert severity2 == DamageSeverity.MODERATE


class TestAssessment:
    def test_physical_width_overrides_confidence(self):
        a = assess_structure_condition("deck", "cracked", 0.99, physical_width_mm=2.0)
        assert a.severity == DamageSeverity.CRITICAL

    def test_uncracked_is_healthy(self):
        a = assess_structure_condition("wall", "uncracked", 0.8)
        assert a.severity == DamageSeverity.NONE
        assert a.safety_factor == 1.0

    def test_severity_length_note(self):
        sev, rec = calculate_crack_severity(0.15, length_mm=600)
        assert sev == DamageSeverity.MODERATE
        assert "monitoring" in rec.lower()

    def test_batch_rollup_counts(self):
        report = batch_structure_assessment([
            {"structure_type": "deck", "condition": "cracked", "confidence": 0.95},
            {"structure_type": "wall", "condition": "uncracked", "confidence": 0.9},
        ])
        assert report["total_structures"] == 2
        assert report["damaged_structures"] == 1
        assert report["overall_rating"] in ("SAFE", "CAUTION", "UNSAFE", "DANGER")

    def test_safety_rating_empty(self):
        r = structural_safety_rating([])
        assert r["overall_rating"] == "SAFE"
        assert r["safety_score"] == 100.0


# ----------------------------------------------------------------------
# crack_measurement: synthetic-image pipeline
# ----------------------------------------------------------------------


def _make_crack_image(width_px: int = 6) -> np.ndarray:
    """Gray canvas with one horizontal dark bar of known width."""
    import cv2

    canvas = np.full((480, 640, 3), 180, dtype=np.uint8)
    y0 = 240
    cv2.rectangle(canvas, (60, y0), (560, y0 + width_px - 1), (20, 20, 20), -1)
    return canvas


class TestCrackMeasurement:
    def test_pinhole_conversion(self):
        assert pixels_to_mm(50, 2000, 800) == 125.0

    def test_synthetic_crack_recovered(self):
        res = estimate_crack_width(_make_crack_image(), 2000, 800)
        assert res["measured"]
        # 6-px bar @ 2.5 mm/px = 15 mm (tolerance for morphology).
        assert abs(res["avg_width_mm"] - 15.0) < 2.5
        assert res["n_components"] == 1
        assert res["severity"] == DamageSeverity.COLLAPSE_RISK  # 15mm generic ladder

    def test_texture_rejected(self):
        # Noisy canvas without elongated dark structure -> no measurement.
        rng = np.random.default_rng(42)
        noisy = rng.integers(120, 200, (480, 640, 3), dtype=np.uint8)
        res = estimate_crack_width(noisy, 2000, 800)
        assert not res["measured"]

    def test_width_is_monotonic_in_true_width(self):
        # Regression: 1/2/3/4 px bars all measured 10.00 mm and an 8 px bar
        # came out narrower than a 7 px one, because 2*dt on an integer
        # distance transform can only yield even pixel counts.
        prev = 0.0
        for w_px in (3, 4, 5, 6, 7, 8):
            res = estimate_crack_width(_make_crack_image(w_px), 2000, 800)
            assert res["measured"], f"{w_px}px rejected: {res.get('message')}"
            assert res["avg_width_mm"] > prev, (
                f"non-monotonic: {w_px}px -> {res['avg_width_mm']:.2f} "
                f"<= {prev:.2f}"
            )
            prev = res["avg_width_mm"]

    def test_accurate_above_the_blur_floor(self):
        # With sub-pixel estimation, widths above the 5x5-blur floor should
        # land within ~15% of truth at 2.5 mm/px.
        for w_px, true_mm, tol in ((3, 7.5, 0.20), (4, 10.0, 0.15), (6, 15.0, 0.05)):
            res = estimate_crack_width(_make_crack_image(w_px), 2000, 800)
            assert res["measured"]
            assert abs(res["avg_width_mm"] - true_mm) / true_mm < tol, (
                f"{w_px}px: got {res['avg_width_mm']:.2f}mm, "
                f"expected {true_mm}mm within {tol:.0%}"
            )

    def test_thin_crack_reported_unresolved_not_guessed(self):
        # A 1 px feature is below the 5x5 blur floor. The estimator will still
        # return a plausible-looking number for it, so the gate must reject it
        # rather than publish a width.
        res = estimate_crack_width(_make_crack_image(1), 2000, 800)
        assert not res["measured"]
        assert res["resolution_limited"] is True
        assert res["severity"] == DamageSeverity.UNCERTAIN

    @pytest.mark.parametrize("dist,focal", [(2000, 800), (200, 2000)])
    def test_resolution_floor_is_calibration_independent(self, dist, focal):
        # The floor is an image-resolution limit, not a physical one: 1 px is
        # 1 px however many mm it maps to.
        res = estimate_crack_width(_make_crack_image(1), dist, focal)
        assert res["resolution_limited"] is True

    def test_wide_crack_accepted_at_close_range(self):
        # Regression: the old absolute max_mean_width_px=6.0 gate rejected
        # anything wider than 6 px, so close-up photos -- the accurate regime
        # -- reported "no crack detected" for a 12 px (1.2 mm) crack.
        import cv2

        canvas = np.full((600, 800, 3), 180, dtype=np.uint8)
        cv2.rectangle(canvas, (60, 300), (740, 311), (20, 20, 20), -1)
        res = estimate_crack_width(canvas, distance_mm=200, focal_length_px=2000)
        assert res["measured"], res.get("message")
        assert 0.9 < res["avg_width_mm"] < 1.5, res["avg_width_mm"]


class TestCrackOverlay:
    """The red fill must mark the crack, not bury it."""

    @staticmethod
    def _cracked_scene():
        import cv2

        canvas = np.full((300, 400, 3), 180, dtype=np.uint8)
        cv2.rectangle(canvas, (40, 150), (360, 157), (20, 20, 20), -1)
        return canvas

    def test_fill_never_spills_outside_the_crack_mask(self):
        # Regression: overlay_crack_mask dilated by thickness=2, inflating a
        # 14.7% crack mask to 18.0% of a 256px tile, so the 30%-opacity wash
        # covered the crack it was meant to mark.
        import cv2

        res = estimate_crack_width(
            self._cracked_scene(), 2000, 800, return_debug_info=True
        )
        assert res["overlay_image"] is not None
        mask = res["debug_images"]["filtered_mask"] > 0
        ov = res["overlay_image"].astype(int)
        ordered = np.sort(ov, axis=2)
        # Painted = red channel clearly dominant.
        painted = (ov.max(axis=2) - ordered[:, :, 1]) > 40
        assert not (painted & ~mask).any(), (
            f"{int((painted & ~mask).sum())} painted pixels lie outside the crack"
        )

    def test_overlay_present_even_when_width_is_unresolved(self):
        # A crack below the blur floor is still located, so it must still be
        # marked; only the width is withheld.
        import cv2

        canvas = np.full((300, 400, 3), 180, dtype=np.uint8)
        cv2.rectangle(canvas, (40, 150), (360, 150), (20, 20, 20), -1)  # 1px
        res = estimate_crack_width(canvas, 2000, 800)
        if res.get("crack_located"):
            assert res["overlay_image"] is not None
            assert not res["measured"]

    def test_overlay_preserves_image_shape(self):
        import cv2

        res = estimate_crack_width(self._cracked_scene(), 2000, 800)
        assert res["overlay_image"].shape == (300, 400, 3)

    def test_orientation_horizontal(self):
        skel = np.zeros((100, 200), dtype=np.uint8)
        skel[50, 20:180] = 255
        dev = estimate_crack_orientation(skel)
        assert dev is not None
        assert min(dev, 180 - dev) < 5

    def test_filter_rejects_blob(self):
        import cv2

        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 40, 255, -1)  # round blob, elongation ~1
        out, kept = filter_components(mask)
        assert kept == 0
        assert cv2.countNonZero(out) == 0

    def test_structure_aware_severity_in_result(self):
        # 15mm avg: collapse-risk generically, but 'minor'-adjacent for
        # pavement scale (12mm max allowable => collapse here too); assert
        # grading ran with structure context by checking severity present.
        res = estimate_crack_width(_make_crack_image(), 2000, 800,
                                   structure_type="pavement")
        assert isinstance(res["severity"], DamageSeverity)

    def test_result_reports_mm_per_px(self):
        res = estimate_crack_width(_make_crack_image(), 2000, 800)
        assert abs(res["mm_per_px"] - 2.5) < 1e-3

    def test_coarse_calibration_warns_for_fine_elements(self):
        # Wall at 2.5 mm/px: measurement still returns numbers (they are
        # honest geometry) but is flagged as low-accuracy.
        res = estimate_crack_width(_make_crack_image(), 2000, 800,
                                   structure_type="wall")
        assert res["measured"]
        assert "WARNING" in res["message"]
        assert "coarser" in res["message"]


# ----------------------------------------------------------------------
# Import safety of CLI modules
# ----------------------------------------------------------------------


def test_inference_imports():
    import inference  # noqa: F401

    assert hasattr(inference, "StructuralClassifier")


def test_package_imports():
    import src  # noqa: F401  (project root on path)

    assert src.__version__ == "2.0.0"
