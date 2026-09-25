#!/usr/bin/env python3
"""
Crack Measurement Module for SHM Vision
=======================================

Estimates physical crack geometry (width, length, orientation) from images
using classical computer vision + photogrammetry.

Pipeline:
    1. Preprocess: grayscale -> CLAHE -> Gaussian blur.
    2. Segment: Otsu threshold on inverted intensity (cracks are dark),
       then open/close morphology.
    3. Filter: connected-component analysis rejects specks (area) and
       texture blobs (elongation).
    4. Skeletonize: 1-px centerline via ximgproc thinning, with a
       Guo-Hall-free fallback to morphological opening-based thinning.
    5. Measure: distance transform sampled along the skeleton gives
       per-point half-widths; robust statistics summarize them.
    6. Convert: pixels -> mm via the pinhole relation W = px * D / f.

Usage:
    from crack_measurement import estimate_crack_width
    result = estimate_crack_width(image, distance_mm=2000,
                                  focal_length_px=800, structure_type="deck")
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config import (
    CRACK_MEASUREMENT,
    DEFAULT_CAMERA_DISTANCE_MM,
    DEFAULT_FOCAL_LENGTH_PX,
    MIN_DETECTED_CRACK_SEVERITY,
    DamageSeverity,
    assess_severity_from_width,
    get_safety_factor,
    get_structure_thresholds,
    max_trusted_mm_per_px,
    worst_severity,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Preprocessing & segmentation
# ----------------------------------------------------------------------


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Prepare an image for crack segmentation.

    Grayscale conversion (if needed) -> CLAHE contrast normalization
    (handles uneven outdoor lighting) -> Gaussian blur (suppresses sensor
    noise without destroying hairline cracks).

    Args:
        image: Input image, BGR (HxWx3) or grayscale (HxW).

    Returns:
        Preprocessed single-channel uint8 image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    ksize = CRACK_MEASUREMENT["gaussian_blur_kernel"]
    return cv2.GaussianBlur(enhanced, (ksize, ksize), 0)


def detect_crack_mask(preprocessed: np.ndarray) -> np.ndarray:
    """
    Threshold a preprocessed image into a binary crack mask (tiered).

    Strategy: cracks are thin dark structures, highlighted by a
    morphological **black-hat** transform (closing - image), immune to
    lighting gradients. The response is binarized with Otsu, then:

    * **strict tier**: intersect with the top ~1% of responses. Kills
      texture (used when the strong response alone forms a plausible
      crack).
    * **loose tier**: raw Otsu response. Catches faint cracks the strict
      tier misses; false positives are acceptable because upstream callers
      only run measurement after the classifier flags the image as cracked.

    A light 3x3 closing bridges 1-px gaps; no opening anywhere (opening
    erases hairline cracks).

    Args:
        preprocessed: Output of :func:`preprocess_image`.

    Returns:
        Binary uint8 mask (255 = crack candidate).
    """
    k = max(9, (min(preprocessed.shape[:2]) // 16) | 1)  # odd, ~1/16 of side
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(preprocessed, cv2.MORPH_BLACKHAT, kernel)

    _, otsu = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(otsu) == 0:
        return otsu

    # Strict: keep only the strongest ~1% of the response.
    strong = np.percentile(blackhat, 99.0)
    strict = np.where((blackhat >= strong) & (otsu > 0), 255, 0).astype(np.uint8)
    return strict if cv2.countNonZero(strict) else otsu


def detect_crack_mask_tiered(preprocessed: np.ndarray):
    """
    Yield candidate masks from strict to loose.

    Callers pass each candidate through filtering + gates and take the
    first that yields a plausible crack measurement. Returns an iterator
    of binary masks.
    """
    k = max(9, (min(preprocessed.shape[:2]) // 16) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(preprocessed, cv2.MORPH_BLACKHAT, kernel)

    _, otsu = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(otsu) == 0:
        return

    strong = np.percentile(blackhat, 99.0)
    strict = np.where((blackhat >= strong) & (otsu > 0), 255, 0).astype(np.uint8)
    bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if cv2.countNonZero(strict):
        yield cv2.morphologyEx(strict, cv2.MORPH_CLOSE, bridge, iterations=1)
    yield cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, bridge, iterations=1)


def filter_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Reject non-crack connected components from a binary mask.

    Three criteria (all from :data:`config.CRACK_MEASUREMENT`):
        * ``min_component_area_frac``: drops dirt specks / salt noise.
        * ``max_circularity``: cracks are thin & long, so their area is
          tiny relative to their perimeter (4*pi*A/P^2 << 1). Solid blobs
          (shadows, stains, aggregate) approach 1 and are rejected.
        * ``min_length_width_ratio``: total skeleton length over mean
          width must reach this. Remaining texture survivors (short,
          chunky clusters) fail it. Bounding-box elongation was abandoned:
          it rejected diagonal and zigzag cracks whose bbox is square-ish.

    Args:
        mask: Binary uint8 mask.

    Returns:
        (filtered_mask, kept_component_count).
    """
    cfg = CRACK_MEASUREMENT
    h, w = mask.shape[:2]
    img_area = float(h * w)
    min_area = img_area * cfg["min_component_area_frac"]

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    kept = 0

    for i in range(1, num):  # label 0 is background
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        # Perimeter-based thinness test (robust to orientation/shape).
        comp = (labels == i).astype(np.uint8)
        contours = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True)
        circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1.0)
        if circularity > cfg["max_circularity"]:
            continue
        out[labels == i] = 255
        kept += 1

    return out, kept


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Reduce a binary mask to its 1-pixel-wide centerline.

    Prefers OpenCV-contrib ``ximgproc.thinning`` (Zhang-Suen); falls back to
    the classic morphological "erode-open-subtract" skeleton if the contrib
    package is unavailable.

    Args:
        mask: Binary uint8 mask.

    Returns:
        Binary uint8 skeleton (255 on centerline pixels).
    """
    try:
        from cv2 import ximgproc  # type: ignore

        return ximgproc.thinning(mask)
    except (ImportError, AttributeError):
        pass

    skeleton = np.zeros(mask.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = mask.copy()
    while cv2.countNonZero(temp) > 0:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(temp, opened))
        temp = eroded.copy()
    return skeleton


# ----------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------


def measure_crack_width_pixels(
    mask: np.ndarray, skeleton: np.ndarray, intensity: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Measure crack width statistics via distance transform along skeleton.

    At every skeleton pixel, the L2 distance transform value equals the
    half-width of the crack at that point (distance to nearest edge), so the
    local full width is ``2 * dt``.

    ``cv2.distanceTransform`` returns integer-valued distances, so a plain
    ``2 * dt`` can only ever produce *even* pixel widths. Measured against
    synthetic bars this quantized 1, 2, 3 and 4 px cracks all to 4 px, and an
    8 px crack came out *narrower* than a 7 px one -- the single largest
    source of bogus "10.00 mm" readings at the default calibration.

    The fix is a sub-pixel edge estimate: the local width is refined against
    the underlying grayscale intensity profile across the crack. A dark
    structure on a brighter background has its true edges at the half-maximum
    intensity crossings, which interpolate smoothly between pixels and so
    resolve widths far below one pixel of quantization.

    Args:
        mask: Binary crack mask.
        skeleton: 1-px centerline of the mask.
        intensity: Optional single-channel image used for sub-pixel edge
            localization. When omitted (or mismatched in shape) the integer
            distance transform is used, i.e. the old quantized behaviour.

    Returns:
        Dict of robust width statistics in pixels:
        ``max``, ``mean``, ``median``, ``p90``, ``std``, ``samples``.

    Raises:
        ValueError: If the skeleton is empty.
    """
    coords = np.argwhere(skeleton > 0)
    if coords.size == 0:
        raise ValueError("Empty skeleton: no crack centerline found")

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    widths = 2.0 * dist[coords[:, 0], coords[:, 1]]

    if intensity is not None and intensity.shape == mask.shape:
        refined = _subpixel_widths(intensity, mask, coords, widths)
        if refined is not None:
            widths = refined

    q25, q75 = np.percentile(widths, [25, 75])
    iqr = q75 - q25
    lo, hi = q25 - 1.5 * iqr, q75 + 1.5 * iqr
    inliers = widths[(widths >= lo) & (widths <= hi)]
    if inliers.size:
        widths = inliers

    return {
        "max": float(np.max(widths)),
        "mean": float(np.mean(widths)),
        "median": float(np.median(widths)),
        "p90": float(np.percentile(widths, 90)),
        "std": float(np.std(widths)),
        "samples": int(widths.size),
    }


def _subpixel_edge_offset(
    profile: np.ndarray, center_idx: int, half_width: float
) -> Optional[float]:
    """
    Estimate crack width along one scanline by integrated intensity.

    The half-maximum edge criterion needs a trustworthy background estimate,
    which is unavailable here: the mask is built from a 5x5-blurred image, so
    the transition spans several pixels and a scanline segment short enough to
    bracket the crossing still lies *inside* the crack. Measuring against that
    corrupted background made a 12 px bar read as 4 px.

    The integrated-intensity ("area") method sidesteps edges entirely. For a
    dark structure of width ``w`` on a background of level ``b`` with core
    level ``c``, the profile satisfies
    ``sum(max(0, b - I)) = w * (b - c)``, so
    ``w = sum(max(0, b - I)) / (b - c)``. Blur redistributes intensity but
    conserves the integral, which makes the estimate blur-invariant and
    genuinely sub-pixel.

    Args:
        profile: 1-D intensity samples along a scanline crossing the crack.
        center_idx: Index in ``profile`` of the skeleton pixel.
        half_width: Integer distance-transform half-width, in samples; used
            only to size the search and to validate the segment.

    Returns:
        Sub-pixel width in samples, or ``None`` when the segment does not
        reach background on both sides (caller keeps the integer estimate).
    """
    n = profile.size
    if n < 7 or center_idx < 2 or center_idx > n - 3:
        return None

    # Background: high percentile of the segment -- cracks are dark, so the
    # bright tail represents the surrounding surface.
    b = float(np.percentile(profile, 90.0))
    core = float(profile[center_idx - 1 : center_idx + 2].min())
    denom = b - core
    if denom <= 1e-6:
        # Not a dark structure on a bright background (e.g. a bright defect).
        return None

    # The segment must actually reach background on BOTH sides, otherwise the
    # integral is truncated and the width comes out too small.
    edge_samples = np.concatenate((profile[:2], profile[-2:]))
    if float(edge_samples.mean()) < 0.75 * b:
        return None

    excess = float(np.clip(b - profile, 0.0, None).sum())
    width = excess / denom
    if not np.isfinite(width) or width <= 0:
        return None
    return width


def _subpixel_widths(
    intensity: np.ndarray,
    mask: np.ndarray,
    coords: np.ndarray,
    fallback: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Refine distance-transform widths using the grayscale profile.

    For each skeleton pixel the row and column intensity profiles through it
    are measured; their widths are averaged. Samples whose profile cannot be
    resolved keep the integer distance transform, so the result is never worse
    than the unrefined estimate.

    Args:
        intensity: Single-channel image, same shape as ``mask``.
        mask: Binary crack mask.
        coords: ``(N, 2)`` array of skeleton ``(row, col)`` coordinates.
        fallback: Integer distance-transform widths for each coordinate.

    Returns:
        Array of refined widths, or ``None`` if no sample could be refined.
    """
    if intensity.ndim == 3:
        intensity = cv2.cvtColor(intensity, cv2.COLOR_BGR2GRAY)
    intensity = intensity.astype(np.float32)

    refined = np.array(fallback, dtype=np.float64)
    n_refined = 0
    rows, cols = intensity.shape

    for i, ((r, c), base) in enumerate(zip(coords, fallback)):
        half = max(1, int(round(base / 2.0)))
        # The scanline must span the crack plus background on both sides, so
        # reach scales with the estimated half-width (plus slack for blur).
        reach = max(3 * half, 10)
        r, c = int(r), int(c)
        if r - reach < 0 or c - reach < 0 or r + reach >= rows or c + reach >= cols:
            reach = min(r, c, rows - r - 1, cols - c - 1)
        if reach < 6:
            continue

        cands = []
        for arr, idx in ((intensity[r, :], c), (intensity[:, c], r)):
            seg = arr[idx - reach : idx + reach + 1].astype(np.float64)
            w = _subpixel_edge_offset(seg, reach, half)
            if w is not None:
                cands.append(w)
        if cands:
            refined[i] = float(np.mean(cands))
            n_refined += 1

    if n_refined == 0:
        return None
    # Physically sane envelope: at least 1 px, and never more than the
    # integer estimate inflated by the blur halo.
    refined = np.clip(refined, 1.0, fallback * 2.5 + 1.0)
    return refined


def measure_crack_length_pixels(skeleton: np.ndarray) -> float:
    """
    Estimate crack length in pixels as the skeleton's pixel count
    (centerline length in px ~= number of skeleton pixels for 8-connected
    thin lines, slightly underestimating diagonal segments).

    Args:
        skeleton: 1-px centerline mask.

    Returns:
        Skeleton length in pixels (0 if empty).
    """
    return float(cv2.countNonZero(skeleton))


def estimate_crack_orientation(skeleton: np.ndarray) -> Optional[float]:
    """
    Dominant crack orientation via PCA on skeleton pixels.

    Args:
        skeleton: 1-px centerline mask.

    Returns:
        Orientation in degrees within [0, 180): 0 = horizontal,
        90 = vertical. ``None`` if fewer than 2 skeleton pixels.
    """
    coords = np.argwhere(skeleton > 0)
    if len(coords) < 2:
        return None

    # coords rows are (row, col) = (y, x): PCA dim 0 = y, dim 1 = x.
    _, eigvec, _ = cv2.PCACompute2(coords.astype(np.float64), mean=None)
    dx, dy = float(eigvec[0, 1]), float(eigvec[0, 0])
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    return float(angle)


def overlay_crack_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 0, 255),
    alpha: float = 0.45,
    thickness: int = 1,
    skeleton: Optional[np.ndarray] = None,
    centerline_color: tuple = (0, 255, 255),
) -> np.ndarray:
    """
    Blend a semi-transparent colored overlay of the crack mask onto an image.

    The fill covers *exactly* the detected crack pixels. It used to dilate the
    mask by ``thickness=2`` first, which on a 256 px inspection tile inflated a
    14.7% crack mask to 18.0% of the frame -- the 30%-opacity red wash then
    covered the very crack it was meant to mark, and the surrounding aggregate
    texture with it. Dilating is now opt-in (``thickness > 1``) and off by
    default.

    When ``skeleton`` is supplied its centerline is drawn on top in a
    contrasting colour, so the detected crack path stays unambiguous even where
    the fill is semi-transparent.

    Args:
        image: Original BGR image.
        mask: Binary mask to highlight.
        color: BGR overlay color (default red).
        alpha: Overlay opacity in [0, 1].
        thickness: Mask dilation radius in pixels; 1 (default) paints the mask
            as-is, >1 grows it for extra visibility.
        skeleton: Optional 1-px centerline mask drawn over the fill.
        centerline_color: BGR color for the centerline (default yellow).

    Returns:
        BGR image with the mask highlighted.
    """
    result = image.copy()
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    paint = mask
    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        paint = cv2.dilate(mask, kernel, iterations=1)

    overlay = result.copy()
    overlay[paint > 0] = color
    out = cv2.addWeighted(result, 1.0 - alpha, overlay, alpha, 0)

    if skeleton is not None and cv2.countNonZero(skeleton):
        # Draw the centerline at full opacity: it is 1 px wide, so it marks the
        # detected path without hiding surface detail.
        out[skeleton > 0] = centerline_color

    return out


def pixels_to_mm(
    pixel_width: float, distance_mm: float, focal_length_px: float
) -> float:
    """
    Convert a pixel length to millimeters: ``W = px * D / f`` (pinhole).

    Args:
        pixel_width: Length in pixels (>= 0).
        distance_mm: Camera distance in mm (> 0).
        focal_length_px: Focal length in pixels (> 0).

    Returns:
        Length in millimeters, rounded to 3 decimals.

    Raises:
        ValueError: On non-positive distance/focal length or negative width.
    """
    if focal_length_px <= 0 or distance_mm <= 0:
        raise ValueError("focal_length and distance must be positive")
    if pixel_width < 0:
        raise ValueError("pixel_width cannot be negative")
    return round(pixel_width * (distance_mm / focal_length_px), 3)


def estimate_crack_width(
    image: np.ndarray,
    distance_mm: float = DEFAULT_CAMERA_DISTANCE_MM,
    focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX,
    structure_type: Optional[str] = None,
    return_debug_info: bool = False,
) -> Dict:
    """
    Full crack-measurement pipeline: segment -> filter -> skeletonize ->
    measure -> convert to mm -> grade severity.

    Args:
        image: Input image (BGR HxWx3 or grayscale HxW).
        distance_mm: Camera-to-surface distance in mm.
        focal_length_px: Focal length in pixels.
        structure_type: Optional element key (deck/pavement/wall/...) making
            severity grading structure-aware.
        return_debug_info: Include intermediate images + per-component
            records in the result under ``debug_images``/``components``.

    Returns:
        Dict with keys:
            ``measured`` (bool): True when a plausible crack was quantified.
            ``avg_width_mm`` / ``max_width_mm``: headline widths.
            ``median_width_mm`` / ``p90_width_mm`` / ``width_std_mm``:
                granular distribution statistics.
            ``length_mm``: estimated crack length.
            ``orientation_deg``: dominant orientation (0 = horizontal).
            ``n_components``: crack-like components retained.
            ``severity`` / ``safety_factor``: engineering assessment.
            ``overlay_image``: BGR visualization (None on failure).
            ``message``: human-readable summary of what happened.
            ``distance_mm`` / ``focal_length_px``: echo of inputs.
    """
    result: Dict = {
        "max_width_mm": 0.0,
        "avg_width_mm": 0.0,
        "measured": False,
        "overlay_image": None,
        "message": "",
    }
    try:
        preprocessed = preprocess_image(image)

        # Tiered segmentation: strict mask first (texture-proof), loose
        # Otsu fallback (catches faint cracks). The first tier whose
        # filtered mask passes all gates wins. FP risk of the loose tier
        # is acceptable: the app only calls measurement after the
        # classifier flags the image as cracked.
        skeleton = None
        for cand_mask in detect_crack_mask_tiered(preprocessed):
            cand, n_kept = filter_components(cand_mask)
            if n_kept == 0 or cv2.countNonZero(cand) < CRACK_MEASUREMENT[
                "min_crack_length_px"
            ]:
                continue
            # Area gate on the FILTERED mask: crack-shaped components only.
            # Raw tier1 masks on dense alligator cracking legitimately cover
            # 20-40% of the frame; the circularity filter strips the texture
            # halo, so a sparse residue here means genuine crack structure.
            # A sprawling filtered mask (>=15%) is noise/texture.
            if cv2.countNonZero(cand) > CRACK_MEASUREMENT[
                "max_mask_area_frac"
            ] * cand.size:
                continue
            try:
                cand_skel = skeletonize_mask(cand)
                cand_stats = measure_crack_width_pixels(
                    cand, cand_skel, intensity=preprocessed
                )
            except ValueError:
                continue

            # Length/width + width-plausibility gates.
            length_px = measure_crack_length_pixels(cand_skel)
            lw_ratio = length_px / max(cand_stats["mean"], 0.5)
            if lw_ratio < CRACK_MEASUREMENT["min_length_width_ratio"]:
                continue
            # Scale-invariant width ceiling (fraction of the shorter side):
            # catches texture blobs without rejecting genuinely wide cracks
            # at close range, which the old absolute 6 px cap did.
            if cand_stats["mean"] > CRACK_MEASUREMENT["max_mean_width_frac"] * min(
                cand.shape[:2]
            ):
                continue

            mask, n_components, skeleton, stats_px = (
                cand, n_kept, cand_skel, cand_stats
            )
            break

        if skeleton is None:
            result.update(
                n_components=0,
                severity=DamageSeverity.NONE,
                safety_factor=1.0,
                message="No crack-like component detected in image for measurement.",
            )
            if return_debug_info:
                result["debug_images"] = {"preprocessed": preprocessed}
            return result

        result["n_components"] = n_components
        orientation = estimate_crack_orientation(skeleton)

        # Paint the detected crack as soon as it is located, independently of
        # whether the *width* survives validation below. Previously the
        # overlay was only attached on the fully-accepted path, so a crack that
        # was detected but rejected as resolution-limited or implausible was
        # shown to the user with no red marking at all.
        result["overlay_image"] = overlay_crack_mask(
            image, mask, skeleton=skeleton
        )
        result["crack_located"] = True

        mm = lambda px_val: pixels_to_mm(px_val, distance_mm, focal_length_px)
        mm_per_px = distance_mm / focal_length_px
        avg_w_mm = mm(stats_px["mean"])
        result.update(
            max_width_mm=mm(stats_px["max"]),
            avg_width_mm=avg_w_mm,
            median_width_mm=mm(stats_px["median"]),
            p90_width_mm=mm(stats_px["p90"]),
            width_std_mm=mm(stats_px["std"]),
            length_mm=mm(length_px),
            orientation_deg=orientation,
            measured_points=stats_px["samples"],
            mm_per_px=round(mm_per_px, 4),
            distance_mm=distance_mm,
            focal_length_px=focal_length_px,
        )

        # Physically implausible width => texture false positive, reject.
        max_plausible = CRACK_MEASUREMENT["max_plausible_avg_width_mm"]
        if avg_w_mm > max_plausible:
            result.update(
                severity=DamageSeverity.UNCERTAIN,
                safety_factor=get_safety_factor(DamageSeverity.UNCERTAIN),
                message=(
                    f"Measurement rejected: avg={avg_w_mm:.1f}mm exceeds plausible "
                    f"crack width ({max_plausible:.0f}mm). Likely texture/shadow."
                ),
            )
            return result

        # Resolvability floor: below this the segmentation halo and anti-aliasing
        # dominate, so any number we print is noise. Report the limit honestly
        # instead of a confident figure -- a 1 px crack previously came back as
        # a clean "10.00mm" at the default 2.5 mm/px calibration.
        min_resolvable_mm = (
            CRACK_MEASUREMENT["min_resolvable_width_px"] * mm_per_px
        )
        if stats_px["mean"] < CRACK_MEASUREMENT["min_resolvable_width_px"]:
            result.update(
                severity=DamageSeverity.UNCERTAIN,
                safety_factor=get_safety_factor(DamageSeverity.UNCERTAIN),
                measured=False,
                resolution_limited=True,
                message=(
                    f"Crack detected but width is below the measurement floor: "
                    f"~{stats_px['mean']:.1f}px < "
                    f"{CRACK_MEASUREMENT['min_resolvable_width_px']:.1f}px resolvable "
                    f"(~{min_resolvable_mm:.2f}mm at 1px={mm_per_px:.2f}mm). "
                    f"Get closer / use full resolution to measure it; treat the "
                    f"width as unresolved, not as {avg_w_mm:.2f}mm."
                ),
            )
            logger.info(result["message"])
            return result

        severity = worst_severity(
            assess_severity_from_width(avg_w_mm, structure_type),
            MIN_DETECTED_CRACK_SEVERITY,
        )
        result.update(
            severity=severity,
            safety_factor=get_safety_factor(severity),
            measured=True,
            resolution_limited=False,
            message=(
                f"Crack measured: avg={avg_w_mm:.3f}mm, max={result['max_width_mm']:.3f}mm, "
                f"length={result['length_mm']:.1f}mm ({n_components} component(s), "
                f"{stats_px['samples']} sample points; 1px={mm_per_px:.2f}mm)"
            ),
        )
        if structure_type and mm_per_px > max_trusted_mm_per_px(structure_type):
            result["message"] += (
                f" WARNING: calibration resolution 1px={mm_per_px:.2f}mm is coarser "
                f"than {max_trusted_mm_per_px(structure_type):.2f}mm/px trusted for "
                f"{structure_type}; width estimate is low-accuracy."
            )
        logger.info(result["message"])

        if return_debug_info:
            vis = overlay_crack_mask(image, mask)
            vis[skeleton > 0] = (0, 255, 0)
            result["debug_images"] = {
                "preprocessed": preprocessed,
                "filtered_mask": mask,
                "skeleton": skeleton,
                "visualization": vis,
            }
        return result

    except Exception as exc:  # defensive: callers treat result dict as total
        logger.error("Crack measurement failed: %s", exc, exc_info=True)
        result.update(
            severity=DamageSeverity.UNCERTAIN,
            safety_factor=get_safety_factor(DamageSeverity.UNCERTAIN),
            measured=False,
            message=f"Measurement error: {exc}",
        )
        return result


def annotate_measurement(
    image: np.ndarray,
    measurement: Dict,
    x_offset: int = 20,
    y_offset: int = 140,
) -> np.ndarray:
    """
    Render measurement results as a HUD panel on a copy of the image.

    Args:
        image: BGR image.
        measurement: Dict as returned by :func:`estimate_crack_width`.
        x_offset: Panel text x origin.
        y_offset: Panel text y origin.

    Returns:
        Annotated BGR image.
    """
    img = image.copy()
    h, w = img.shape[:2]

    overlay = img.copy()
    panel_h = 120
    cv2.rectangle(overlay, (0, y_offset - 20), (w, y_offset + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    severity = measurement.get("severity")
    color = (
        (0, 255, 0)
        if severity in (None, DamageSeverity.NONE)
        else (0, 165, 255)
        if severity in (DamageSeverity.MINOR, DamageSeverity.MODERATE)
        else (0, 0, 255)
    )

    cv2.putText(
        img, "PHYSICAL MEASUREMENT", (x_offset, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    if measurement.get("measured"):
        line1 = f"Avg: {measurement['avg_width_mm']:.3f} mm  Max: {measurement['max_width_mm']:.3f} mm"
        line2 = f"Len: {measurement.get('length_mm', 0.0):.1f} mm | Orient: "
        orient = measurement.get("orientation_deg")
        line2 += f"{orient:.0f} deg" if orient is not None else "n/a"
        sev = measurement["severity"]
        line3 = f"Severity: {sev.value.upper()} | SF: {measurement['safety_factor']:.2f}"
    else:
        line1 = measurement.get("message", "No measurement available")
        line2 = "Using classification-based estimate only."
        line3 = ""

    cv2.putText(img, line1, (x_offset, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(img, line2, (x_offset, y_offset + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, line3, (x_offset, y_offset + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


if __name__ == "__main__":
    # Synthetic-image self-check: draw a known black bar on gray, verify the
    # pipeline recovers it at D=2000mm, f=800px (1 px = 2.5 mm). A 6-row bar
    # measures 6 px wide -> 15.0 mm.
    canvas = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(canvas, (60, 240), (560, 245), (20, 20, 20), -1)
    res = estimate_crack_width(
        canvas, distance_mm=2000, focal_length_px=800, return_debug_info=True
    )
    print(res["message"])
    assert res["measured"], "expected the synthetic crack to be measured"
    assert abs(res["avg_width_mm"] - 15.0) < 2.5, (
        f"avg width {res['avg_width_mm']}mm, expected ~15.0mm (6px * 2.5mm/px)"
    )
    assert res["severity"] == DamageSeverity.COLLAPSE_RISK
    assert res["orientation_deg"] is not None
    deviation_from_horizontal = min(res["orientation_deg"], 180.0 - res["orientation_deg"])
    assert deviation_from_horizontal < 10, (
        "horizontal bar should report ~0/180 deg orientation"
    )
    assert res["debug_images"]["skeleton"] is not None
    print("crack_measurement self-check passed")
