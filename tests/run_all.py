#!/usr/bin/env python3
"""
Stdlib-only test runner for SHM Vision (fallback when pytest is absent).

Mirrors tests/test_core.py: executes every check and prints a pass/fail
summary with a non-zero exit code on failure.

Run from project root:
    python tests/run_all.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import (
    CLASS_NAMES,
    CONDITION_MAP,
    MIN_DETECTED_CRACK_SEVERITY,
    SEVERITY_ORDER,
    STRUCTURE_MAP,
    DamageSeverity,
    assess_severity_from_confidence,
    assess_severity_from_width,
    confidence_tier,
    get_recommendation,
    get_safety_factor,
    get_structure_thresholds,
    worst_severity,
)
from crack_measurement import (
    estimate_crack_orientation,
    estimate_crack_width,
    filter_components,
    pixels_to_mm,
)
from eng_utils import (
    batch_structure_assessment,
    calculate_crack_severity,
    assess_structure_condition,
    estimate_spalling_area,
    get_camera_parameters,
    pixels_to_mm as eng_pixels_to_mm,
)

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# config ---------------------------------------------------------------

@check("generic severity ladder boundaries")
def _():
    assert assess_severity_from_width(0.05) == DamageSeverity.MINOR
    assert assess_severity_from_width(0.2) == DamageSeverity.MODERATE
    assert assess_severity_from_width(0.5) == DamageSeverity.SEVERE
    assert assess_severity_from_width(2.0) == DamageSeverity.CRITICAL
    assert assess_severity_from_width(5.0) == DamageSeverity.COLLAPSE_RISK


@check("structure-aware severity (2mm: pavement moderate / deck critical)")
def _():
    assert assess_severity_from_width(2.0, "pavement") == DamageSeverity.MODERATE
    assert assess_severity_from_width(2.0, "deck") == DamageSeverity.CRITICAL


@check("negative width raises ValueError")
def _():
    try:
        assess_severity_from_width(-1.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


@check("low-confidence damaged -> UNCERTAIN; confident -> MINOR baseline")
def _():
    assert assess_severity_from_confidence(0.60, True) == DamageSeverity.UNCERTAIN
    # Corrected semantics: confidence grades certainty of the LABEL, not
    # damage magnitude, so a confident crack gets one conservative baseline
    # instead of a confidence-keyed cliff at 0.85 (previously 0.85 -> MINOR
    # but 0.86 -> MODERATE, i.e. more confidence produced a worse grade).
    assert assess_severity_from_confidence(0.99, True) == DamageSeverity.MINOR
    assert assess_severity_from_confidence(0.99, False) == DamageSeverity.NONE


@check("severity is monotonic in confidence (no cliff at 0.85)")
def _():
    prev = None
    for c in (0.76, 0.80, 0.84, 0.85, 0.86, 0.90, 0.95, 0.99, 1.00):
        sev = assess_severity_from_confidence(c, True)
        rank = SEVERITY_ORDER.index(sev)
        if prev is not None:
            assert rank >= prev, f"severity improved at conf={c}: {sev}"
        prev = rank


@check("measurement can never grade a crack healthier than the baseline")
def _():
    baseline = assess_severity_from_confidence(0.99, True)
    for w in (0.01, 0.05, 0.1, 0.15):
        graded = worst_severity(
            assess_severity_from_width(w, "wall"), MIN_DETECTED_CRACK_SEVERITY
        )
        assert SEVERITY_ORDER.index(graded) >= SEVERITY_ORDER.index(baseline), (
            f"measured {w}mm graded {graded} -- better than unmeasured {baseline}"
        )


@check("confidence_tier 'low' coincides exactly with the UNCERTAIN gate")
def _():
    for c in (0.0, 0.3, 0.6, 0.74, 0.749, 0.75, 0.8, 0.84, 0.85, 0.99):
        low = confidence_tier(c) == "low"
        uncertain = (
            assess_severity_from_confidence(c, True) == DamageSeverity.UNCERTAIN
        )
        assert low == uncertain, f"conf={c}: tier_low={low} uncertain={uncertain}"


@check("class maps mutually consistent")
def _():
    for name in CLASS_NAMES.values():
        assert name in STRUCTURE_MAP and name in CONDITION_MAP
        assert CONDITION_MAP[name] in ("cracked", "uncracked")
        assert STRUCTURE_MAP[name] == name.split("_", 1)[0]


@check("safety factors monotonic non-increasing with severity")
def _():
    order = [DamageSeverity.NONE, DamageSeverity.MINOR, DamageSeverity.MODERATE,
             DamageSeverity.SEVERE, DamageSeverity.CRITICAL, DamageSeverity.COLLAPSE_RISK]
    factors = [get_safety_factor(s) for s in order]
    assert all(a >= b for a, b in zip(factors, factors[1:])), factors


@check("recommendations defined for all severities")
def _():
    for sev in DamageSeverity:
        assert get_recommendation(sev)


# eng_utils ------------------------------------------------------------

@check("photogrammetry basic conversion 50px@2m f=800 -> 125mm")
def _():
    assert abs(eng_pixels_to_mm(50, 2000, 800) - 125.0) < 1e-9


@check("photogrammetry rejects invalid inputs")
def _():
    for fn in (
        lambda: eng_pixels_to_mm(10, 0, 800),
        lambda: eng_pixels_to_mm(10, 2000, -1),
        lambda: eng_pixels_to_mm("10", 2000, 800),
    ):
        try:
            fn()
            raise AssertionError("expected exception")
        except (ValueError, TypeError):
            pass


@check("camera parameter conversion f=50mm sensor=36mm img=1920 -> 2666.67px")
def _():
    assert abs(get_camera_parameters(36, 50, 1920) - 2666.67) < 0.01


@check("spalling area severity")
def _():
    area, severity, _ = estimate_spalling_area(7, 7, 1000, 1000)
    assert abs(area - 49.0) < 1e-6 and severity == DamageSeverity.MINOR
    # Boundary: exactly 100 mm^2 crosses the MINOR limit.
    _, severity2, _ = estimate_spalling_area(10, 10, 1000, 1000)
    assert severity2 == DamageSeverity.MODERATE


@check("physical width overrides confidence-based severity")
def _():
    a = assess_structure_condition("deck", "cracked", 0.99, physical_width_mm=2.0)
    assert a.severity == DamageSeverity.CRITICAL


@check("crack severity with long-crack monitoring note")
def _():
    sev, rec = calculate_crack_severity(0.15, length_mm=600)
    assert sev == DamageSeverity.MODERATE and "monitoring" in rec.lower()


@check("batch rollup counts and rating")
def _():
    report = batch_structure_assessment([
        {"structure_type": "deck", "condition": "cracked", "confidence": 0.95},
        {"structure_type": "wall", "condition": "uncracked", "confidence": 0.9},
    ])
    assert report["total_structures"] == 2
    assert report["damaged_structures"] == 1
    assert report["overall_rating"] in ("SAFE", "CAUTION", "UNSAFE", "DANGER")


# crack_measurement ----------------------------------------------------

def _make_crack_image(width_px=6):
    canvas = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(canvas, (60, 240), (560, 240 + width_px - 1), (20, 20, 20), -1)
    return canvas


@check("synthetic 6px crack recovered at ~15mm (2.5mm/px)")
def _():
    res = estimate_crack_width(_make_crack_image(), 2000, 800)
    assert res["measured"], res.get("message")
    assert abs(res["avg_width_mm"] - 15.0) < 2.5, res["avg_width_mm"]
    assert res["n_components"] == 1
    assert res["severity"] == DamageSeverity.COLLAPSE_RISK


@check("width is monotonic in true crack width (was quantized + non-monotonic)")
def _():
    # Regression: 1/2/3/4px bars all measured 10.00mm, and an 8px bar came out
    # NARROWER than a 7px one, because 2*dt on an integer distance transform
    # can only yield even pixel counts.
    prev = 0.0
    for w_px in (3, 4, 5, 6, 7, 8):
        res = estimate_crack_width(_make_crack_image(w_px), 2000, 800)
        assert res["measured"], f"{w_px}px rejected: {res.get('message')}"
        got = res["avg_width_mm"]
        assert got > prev, f"non-monotonic: {w_px}px -> {got:.2f}mm <= {prev:.2f}mm"
        prev = got


@check("thin crack reported as UNRESOLVED, not as a fake width")
def _():
    res = estimate_crack_width(_make_crack_image(1), 2000, 800)
    assert not res["measured"], "1px crack must not be published as a measurement"
    assert res.get("resolution_limited") is True
    assert res["severity"] == DamageSeverity.UNCERTAIN
    assert "floor" in res["message"].lower()


@check("resolution floor is calibration-independent (it is an image limit)")
def _():
    # The floor is expressed in pixels, so a 1px feature is unresolvable no
    # matter how many mm a pixel maps to. 0.1mm/px makes it 100x smaller
    # physically, but not more measurable.
    for dist, focal in ((2000, 800), (200, 2000)):
        res = estimate_crack_width(_make_crack_image(1), dist, focal)
        assert res.get("resolution_limited") is True, (
            f"1px feature reported as measured at {dist}mm/{focal}px: "
            f"{res.get('message')}"
        )


@check("close-up calibration resolves a crack the default one cannot")
def _():
    # Same 4px feature: unresolvable-as-width at 2.5mm/px is still 4px, but
    # at 0.1mm/px it is a 0.4mm hairline that must still be gated out. What
    # matters is that a genuinely wide feature IS accepted close up.
    canvas = np.full((600, 800, 3), 180, dtype=np.uint8)
    cv2.rectangle(canvas, (60, 300), (740, 311), (20, 20, 20), -1)  # 12px
    res = estimate_crack_width(canvas, distance_mm=200, focal_length_px=2000)
    assert res["measured"], res.get("message")
    assert 0.9 < res["avg_width_mm"] < 1.5, res["avg_width_mm"]


@check("wide cracks are accepted at close range (old 6px cap rejected them)")
def _():
    # 12mm pavement crack at 0.1 mm/px = 120px wide. The old absolute
    # max_mean_width_px=6.0 gate rejected anything wider than 6px, so a
    # close-up photo -- the accurate regime -- reported "no crack detected".
    canvas = np.full((600, 800, 3), 180, dtype=np.uint8)
    cv2.rectangle(canvas, (60, 300), (740, 300 + 12), (20, 20, 20), -1)
    res = estimate_crack_width(canvas, distance_mm=200, focal_length_px=2000)
    assert res["measured"], res.get("message")
    assert res["avg_width_mm"] > 0.5, res["avg_width_mm"]


@check("red fill marks the crack without spilling outside the mask")
def _():
    canvas = np.full((300, 400, 3), 180, dtype=np.uint8)
    cv2.rectangle(canvas, (40, 150), (360, 157), (20, 20, 20), -1)
    res = estimate_crack_width(canvas, 2000, 800, return_debug_info=True)
    mask = res["debug_images"]["filtered_mask"] > 0
    ov = res["overlay_image"].astype(int)
    ordered = np.sort(ov, axis=2)
    painted = (ov.max(axis=2) - ordered[:, :, 1]) > 40
    spill = int((painted & ~mask).sum())
    assert spill == 0, f"{spill} painted pixels outside the crack mask"


@check("texture-only image rejected (no false measurement)")
def _():
    rng = np.random.default_rng(42)
    noisy = rng.integers(120, 200, (480, 640, 3), dtype=np.uint8)
    res = estimate_crack_width(noisy, 2000, 800)
    assert not res["measured"], res.get("message")


@check("orientation: horizontal line ~0/180 deg")
def _():
    skel = np.zeros((100, 200), dtype=np.uint8)
    skel[50, 20:180] = 255
    dev = estimate_crack_orientation(skel)
    assert dev is not None and min(dev, 180 - dev) < 5


@check("component filter rejects round blob")
def _():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 40, 255, -1)
    out, kept = filter_components(mask)
    assert kept == 0 and cv2.countNonZero(out) == 0


@check("CLI modules import cleanly")
def _():
    import inference  # noqa: F401
    import src  # noqa: F401


def main() -> int:
    passed = failed = 0
    print("=" * 70)
    print("SHM VISION - ENGINEERING CORE TEST SUITE")
    print("=" * 70)
    for name, fn in CHECKS:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print("-" * 70)
    print(f"{passed} passed, {failed} failed")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
