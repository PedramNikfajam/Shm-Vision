#!/usr/bin/env python3
"""
SHM Vision - Structural Health Monitoring Dashboard
====================================================

Streamlit application: single-image analysis, batch inspection, live
stream classification, and system configuration. All domain constants and
engineering logic are imported from ``src.config`` (single source of truth).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Make src/ importable (project may be launched from any working directory).
SRC_PATH = str(Path(__file__).resolve().parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from config import (  # noqa: E402
    CLASS_NAMES,
    CONDITION_MAP,
    DEFAULT_CAMERA_DISTANCE_MM,
    DEFAULT_FOCAL_LENGTH_PX,
    MIN_DETECTED_CRACK_SEVERITY,
    OTHER_CLASS,
    DamageSeverity,
    STRUCTURE_MAP,
    UNCERTAINTY_THRESHOLD,
    assess_severity_from_confidence,
    can_escalate_severity,
    confidence_tier,
    get_safety_factor,
    get_recommendation,
    get_structure_thresholds,
    worst_severity,
)
from crack_measurement import estimate_crack_width  # noqa: E402
from eng_utils import batch_structure_assessment  # noqa: E402

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM DARK THEME CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SHM Vision | Concrete Damage Classifier",
    page_icon="👷‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e2e8f0;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarExpandButton"],
    [data-testid="stHeaderActionElements"] {
        visibility: visible !important;
        color: #00d4ff !important;
    }
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff 0%, #7b2cbf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .glass-card {
        background: rgba(30, 30, 55, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 212, 255, 0.18);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .stButton>button {
        background: linear-gradient(90deg, #7b2cbf 0%, #00d4ff 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        width: 100%;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        opacity: 0.92;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
    }
    .badge-healthy {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
    }
    .badge-moderate {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
    }
    .badge-severe {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(15, 15, 30, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. MODEL BACKENDS (PyTorch / ONNX / Simulation)
# -----------------------------------------------------------------------------


@st.cache_resource
def load_shm_model(weights_path: str, use_onnx: bool = False) -> dict:
    """
    Load the best available inference backend, with caching.

    Preference order: ONNX session (if requested and available) -> PyTorch
    YOLO -> simulation stub. Each failure degrades gracefully and is
    surfaced to the user via the sidebar status.

    Args:
        weights_path: Path to ``best.pt`` classification weights.
        use_onnx: Prefer a sibling ``.onnx`` file when present.

    Returns:
        Dict describing the backend: ``type`` (onnx|yolo|mock), the loaded
        object, and the resolved path.
    """
    onnx_path = weights_path.replace(".pt", ".onnx")

    if use_onnx and Path(onnx_path).exists():
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            inp = session.get_inputs()[0]
            return {
                "type": "onnx",
                "session": session,
                "input_name": inp.name,
                "input_shape": [
                    d if isinstance(d, int) else 224 for d in inp.shape[2:]
                ],
                "path": onnx_path,
            }
        except Exception as exc:  # fall through to PyTorch
            st.warning(f"ONNX load failed ({exc}); falling back to PyTorch.")

    if Path(weights_path).exists():
        try:
            from ultralytics import YOLO

            return {
                "type": "yolo",
                "model": YOLO(weights_path),
                "path": weights_path,
            }
        except Exception as exc:
            st.warning(f"PyTorch load failed ({exc}).")

    return {"type": "mock", "model": None, "path": "Simulation Mode"}


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


@st.cache_data
def _imagenet_norm() -> tuple:
    """ImageNet channel mean/std used by Ultralytics classification models."""
    return (
        np.array([0.485, 0.456, 0.406], dtype=np.float32),
        np.array([0.229, 0.224, 0.225], dtype=np.float32),
    )


def classify_structure(image_pil: Image.Image, model_dict: dict) -> tuple:
    """
    Classify one image and return ``(class_name, top1_conf, prob_dict)``.

    All three backends return the same interface. The ONNX backend runs the
    same preprocessing as Ultralytics (square resize, /255, ImageNet norm)
    plus a softmax over logits. The simulation backend returns a fixed
    distribution and is clearly labeled in the UI.
    """
    m_type = model_dict["type"]

    if m_type == "yolo":
        results = model_dict["model"].predict(image_pil, verbose=False)
        probs = results[0].probs
        top1_id = int(probs.top1)
        prob_dict = {
            CLASS_NAMES[i]: float(probs.data[i])
            for i in CLASS_NAMES
            if i < len(probs.data)
        }
        return CLASS_NAMES.get(top1_id, f"class_{top1_id}"), float(probs.top1conf), prob_dict

    if m_type == "onnx":
        h, w = model_dict["input_shape"][0], model_dict["input_shape"][1]
        arr = np.array(image_pil.convert("RGB").resize((w, h), Image.BILINEAR))
        arr = arr.astype(np.float32) / 255.0
        mean, std = _imagenet_norm()
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)[None]  # HWC -> NCHW
        logits = model_dict["session"].run(
            None, {model_dict["input_name"]: arr}
        )[0][0]
        probs = _softmax(logits)
        top1_id = int(np.argmax(probs))
        prob_dict = {
            CLASS_NAMES.get(i, f"class_{i}"): float(p)
            for i, p in enumerate(probs)
        }
        return (
            CLASS_NAMES.get(top1_id, f"class_{top1_id}"),
            float(probs[top1_id]),
            prob_dict,
        )

    # Simulation Mode: deterministic demo distribution (clearly labeled).
    prob_dict = {
        "deck_cracked": 0.935,
        "deck_uncracked": 0.025,
        "pavement_cracked": 0.020,
        "pavement_uncracked": 0.010,
        "wall_cracked": 0.005,
        "wall_uncracked": 0.005,
    }
    return "deck_cracked", 0.935, prob_dict


# -----------------------------------------------------------------------------
# 3. ENGINEERING EVALUATION (delegates to src.config -- no local duplicates)
# -----------------------------------------------------------------------------


def evaluate_metrics(
    class_name: str,
    confidence: float,
    image_pil: Image.Image | None = None,
    camera_params: dict | None = None,
) -> dict:
    """
    Convert a classification outcome into an engineering assessment.

    Severity resolution order:
        1. Physical crack measurement (photogrammetry), structure-aware.
        2. Confidence-based conservative fallback (UNCERTAIN below threshold).

    Returns:
        Dict with severity, safety factor, badge metadata, thresholds,
        recommendation text and measurement diagnostics.
    """
    is_cracked = CONDITION_MAP.get(class_name, "uncracked") == "cracked"
    struct_type = STRUCTURE_MAP.get(class_name, "deck")
    thresholds = get_structure_thresholds(struct_type)

    if class_name == OTHER_CLASS:
        status_text = "UNKNOWN"
        badge_class, badge_label = "badge-moderate", "OUT OF SCOPE - NOT A RECOGNIZED SURFACE"
        severity = DamageSeverity.UNCERTAIN
        safety_factor = get_safety_factor(severity)
        rec_full = (
            "**OUT OF SCOPE**: the image does not look like one of the trained "
            "structure surfaces (deck / pavement / wall). This is NOT a healthy "
            "verdict -- inspect manually or extend the training data."
        )
        return {
            "is_cracked": False,
            "struct_type": "other",
            "severity": severity.value,
            "safety_factor": safety_factor,
            "status_text": status_text,
            "badge_class": badge_class,
            "badge_label": badge_label,
            "thresholds": thresholds,
            "recommendation": rec_full,
            "physical_width_mm": None,
            "measurement_detail": None,
        }

    physical_width = None
    measurement_detail = None
    measurement_msg = ""

    if not is_cracked:
        severity = DamageSeverity.NONE
        safety_factor = 1.0
        status_text = "HEALTHY"
    else:
        status_text = "DAMAGED"
        if image_pil is not None and camera_params is not None:
            try:
                img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                measurement = estimate_crack_width(
                    img_bgr,
                    distance_mm=camera_params.get("dist", DEFAULT_CAMERA_DISTANCE_MM),
                    focal_length_px=camera_params.get("focal", DEFAULT_FOCAL_LENGTH_PX),
                    structure_type=struct_type,
                )
                measurement_detail = measurement
                if measurement.get("resolution_limited"):
                    # The crack is real but too thin for this calibration to
                    # quantify. Say so; do not print a number.
                    severity = assess_severity_from_confidence(confidence, True)
                    safety_factor = get_safety_factor(severity)
                    measurement_msg = (
                        f" | Width UNRESOLVED at this calibration "
                        f"(1px={measurement.get('mm_per_px', 0):.2f}mm): "
                        f"{measurement.get('message', '')}"
                    )
                elif measurement["measured"]:
                    physical_width = float(measurement["avg_width_mm"])
                    # Trust gate: geometry alone never escalates severity.
                    # When classification confidence is low or calibration is
                    # too coarse for this element, keep the conservative
                    # confidence-based fallback and show the width as info.
                    mm_per_px = float(
                        measurement.get(
                            "mm_per_px",
                            camera_params.get("dist", DEFAULT_CAMERA_DISTANCE_MM)
                            / camera_params.get("focal", DEFAULT_FOCAL_LENGTH_PX),
                        )
                    )
                    if can_escalate_severity(confidence, mm_per_px, struct_type):
                        # Never grade a detected crack healthier than the
                        # MINOR baseline: measuring must not improve the
                        # safety factor relative to leaving it unmeasured.
                        severity = worst_severity(
                            measurement["severity"],
                            MIN_DETECTED_CRACK_SEVERITY,
                        )
                        safety_factor = get_safety_factor(severity)
                        measurement_msg = f" | Measured: {physical_width:.2f}mm"
                    else:
                        severity = assess_severity_from_confidence(confidence, True)
                        safety_factor = get_safety_factor(severity)
                        measurement_msg = (
                            f" | Measured width {physical_width:.2f}mm shown for "
                            f"reference; severity not escalated: insufficient "
                            f"confidence/calibration resolution."
                        )
                else:
                    severity = assess_severity_from_confidence(confidence, True)
                    safety_factor = get_safety_factor(severity)
                    measurement_msg = (
                        f" | Measurement rejected: {measurement.get('message', '')}"
                    )
            except Exception as exc:
                severity = assess_severity_from_confidence(confidence, True)
                safety_factor = get_safety_factor(severity)
                measurement_msg = f" | Measurement error: {exc}"
        else:
            severity = assess_severity_from_confidence(confidence, True)
            safety_factor = get_safety_factor(severity)

    if severity == DamageSeverity.NONE:
        badge_class, badge_label = "badge-healthy", "HEALTHY (NO DAMAGE)"
        status_text = "HEALTHY"
    elif severity == DamageSeverity.UNCERTAIN:
        badge_class, badge_label = "badge-moderate", "UNCERTAIN - MANUAL CHECK REQUIRED"
        # status_text previously stayed "DAMAGED" for an UNCERTAIN verdict, so
        # the summary line contradicted the badge directly above it.
        status_text = "UNCERTAIN"
    elif severity in (DamageSeverity.SEVERE, DamageSeverity.CRITICAL, DamageSeverity.COLLAPSE_RISK):
        badge_class, badge_label = "badge-severe", "SEVERE / CRITICAL DAMAGE"
        status_text = "DAMAGED"
    elif severity == DamageSeverity.MODERATE:
        badge_class, badge_label = "badge-moderate", "MODERATE DAMAGE"
        status_text = "DAMAGED"
    else:
        badge_class, badge_label = "badge-moderate", "MINOR DAMAGE"
        status_text = "DAMAGED"

    recommendation = get_recommendation(severity)
    rec_full = (
        f"**{struct_type.upper()} EVALUATION**: {recommendation} "
        f"(Critical Limit: {thresholds.critical_width}mm | "
        f"Allowable Max: {thresholds.max_allowable}mm)"
        f"{measurement_msg}"
    )

    return {
        "is_cracked": is_cracked,
        "struct_type": struct_type,
        "severity": severity.value,
        "safety_factor": safety_factor,
        "status_text": status_text,
        "badge_class": badge_class,
        "badge_label": badge_label,
        "thresholds": thresholds,
        "recommendation": rec_full,
        "physical_width_mm": physical_width,
        "measurement_detail": measurement_detail,
        # Honest-reporting fields: lets the UI and the PDF/CSV/JSON exports
        # state the evidence quality instead of implying false certainty.
        "confidence_tier": confidence_tier(confidence),
        "measurement_status": _measurement_status(measurement_detail),
        "resolution_limited": bool(
            measurement_detail and measurement_detail.get("resolution_limited")
        ),
    }


def _measurement_status(detail: dict | None) -> str:
    """
    Summarise measurement evidence quality for display and export.

    Args:
        detail: The ``measurement_detail`` dict, or ``None`` when no
            measurement was attempted (e.g. the surface was not cracked).

    Returns:
        One of ``not_attempted``, ``unresolved``, ``rejected``, or
        ``measured``.
    """
    if not detail:
        return "not_attempted"
    if detail.get("resolution_limited"):
        return "unresolved"
    if detail.get("measured"):
        return "measured"
    return "rejected"


def annotate_image(image_input, class_name: str, confidence: float, metrics: dict) -> np.ndarray:
    """Return the raw image as an ndarray (overlay is generated by the
    measurement module and applied upstream)."""
    if isinstance(image_input, Image.Image):
        return np.array(image_input)
    return np.asarray(image_input).copy()


# -----------------------------------------------------------------------------
# 4. REPORT GENERATION (JSON / CSV / PDF)
# -----------------------------------------------------------------------------


def build_json_report(
    class_name: str, confidence: float, metrics: dict, camera_params: dict
) -> str:
    """Serialize one analysis to a JSON string (enums -> plain strings)."""
    detail = metrics.get("measurement_detail") or {}
    payload = {
        "schema_version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "classification": {
            "class": class_name,
            "confidence": round(confidence, 4),
            "confidence_tier": metrics.get("confidence_tier"),
        },
        "assessment": {
            "structure_type": metrics["struct_type"],
            "condition": metrics["status_text"],
            "severity": metrics["severity"],
            "safety_factor": metrics["safety_factor"],
            "critical_width_mm": metrics["thresholds"].critical_width,
            "max_allowable_mm": metrics["thresholds"].max_allowable,
            "recommendation": metrics["recommendation"].replace("**", ""),
        },
        "measurement": {
            "performed": bool(detail.get("measured")),
            "status": metrics.get("measurement_status"),
            "resolution_limited": bool(detail.get("resolution_limited")),
            "mm_per_px": detail.get("mm_per_px"),
            "avg_width_mm": detail.get("avg_width_mm"),
            "max_width_mm": detail.get("max_width_mm"),
            "median_width_mm": detail.get("median_width_mm"),
            "p90_width_mm": detail.get("p90_width_mm"),
            "length_mm": detail.get("length_mm"),
            "orientation_deg": detail.get("orientation_deg"),
            "n_components": detail.get("n_components"),
            "message": detail.get("message"),
        },
        "camera_params": camera_params,
    }
    return json.dumps(payload, indent=2)


def build_csv_report(
    class_name: str, confidence: float, metrics: dict
) -> str:
    """Serialize one analysis to a single-row CSV string."""
    df = pd.DataFrame([{
        "Class": class_name,
        "Confidence": round(confidence, 4),
        "ConfidenceTier": metrics.get("confidence_tier", ""),
        "Status": metrics["status_text"],
        "Severity": metrics["severity"],
        "SafetyFactor": metrics["safety_factor"],
        "Structure": metrics["struct_type"],
        "MeasurementStatus": metrics.get("measurement_status", ""),
        "MeasuredWidth_mm": (
            round(metrics["physical_width_mm"], 3)
            if metrics["physical_width_mm"] is not None else ""
        ),
    }])
    return df.to_csv(index=False)


def generate_pdf_report(
    image_pil: Image.Image,
    class_name: str,
    confidence: float,
    metrics: dict,
    camera_params: dict,
) -> bytes | None:
    """
    Generate an executive PDF report (raw image, metrics table,
    recommendation). Returns None on failure; the UI degrades to JSON/CSV.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import (
            Image as RLImage,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=letter,
            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36,
        )
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DocTitle", parent=styles["Heading1"],
            fontSize=18, textColor=colors.HexColor("#0f172a"), spaceAfter=4,
        )
        sub_style = ParagraphStyle(
            "DocSub", parent=styles["Normal"], fontSize=9,
            textColor=colors.HexColor("#64748b"), spaceAfter=14,
        )
        rec_title_style = ParagraphStyle(
            "RecTitle", parent=styles["Heading3"], fontSize=11,
            textColor=colors.HexColor("#1e293b"), spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "DocBody", parent=styles["Normal"], fontSize=9.5,
            textColor=colors.HexColor("#334155"), leading=14,
        )

        story.append(Paragraph("<b>STRUCTURAL HEALTH MONITORING REPORT</b>", title_style))
        story.append(Paragraph(
            f"<b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"<b>Method:</b> Deep Learning Classification",
            sub_style,
        ))
        story.append(Spacer(1, 4))

        # Aspect-correct image placement (max width 420pt).
        max_w = 420.0
        ratio = image_pil.height / max(image_pil.width, 1)
        img_buffer = io.BytesIO()
        image_pil.save(img_buffer, format="JPEG", quality=95)
        img_buffer.seek(0)
        story.append(RLImage(img_buffer, width=max_w, height=max_w * ratio))
        story.append(Spacer(1, 14))

        if metrics["status_text"] == "HEALTHY":
            status_color = colors.HexColor("#10b981")
        elif metrics["severity"] in ("minor", "moderate", "uncertain"):
            status_color = colors.HexColor("#f59e0b")
        else:
            status_color = colors.HexColor("#ef4444")

        detail = metrics.get("measurement_detail") or {}
        data = [
            ["Metric Parameter", "Inspection Value"],
            ["Class Diagnosis", class_name.upper()],
            ["Structure Element", metrics["struct_type"].capitalize()],
            ["Condition Status", metrics["status_text"]],
            ["Confidence Score", f"{confidence * 100:.2f}%"],
            ["Severity Assessment", metrics["severity"].capitalize()],
            ["Safety Factor", f"{metrics['safety_factor']:.2f}"],
            ["Camera Distance", f"{camera_params['dist']} mm"],
            ["Focal Length", f"{camera_params['focal']} px"],
        ]
        if detail.get("measured"):
            data += [
                ["Measured Avg Width", f"{detail['avg_width_mm']:.3f} mm"],
                ["Measured Max Width", f"{detail['max_width_mm']:.3f} mm"],
                ["Crack Length", f"{detail.get('length_mm', 0.0):.1f} mm"],
                ["Crack Orientation", f"{(detail.get('orientation_deg') or 0):.0f} deg"],
            ]

        t = Table(data, colWidths=[200, 240])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TEXTCOLOR", (1, 5), (1, 5), status_color),
            ("FONTNAME", (1, 5), (1, 5), "Helvetica-Bold"),
        ]))
        story.append(t)
        story.append(Spacer(1, 14))

        story.append(Paragraph("<b>Engineering Recommendation:</b>", rec_title_style))
        story.append(Paragraph(
            metrics["recommendation"].replace("**", ""), body_style
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as exc:
        st.error(f"PDF generation error: {exc}")
        return None


# -----------------------------------------------------------------------------
# 5. SIDEBAR & NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<h2 style='color:#00d4ff; font-size: 1.5rem;'>👷‍♂️ SHM AI Portal</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#94a3b8; font-size: 0.85rem;'>Civil Engineering Damage Classifier</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigation",
        ["Single Image Analysis", "Batch Processing", "Live Stream", "Settings"],
        index=0,
    )

    st.divider()
    st.markdown("### ⚙️ Model Selection")

    runs_dir = Path("runs")
    found_weights = sorted(str(p) for p in runs_dir.rglob("best.pt")) if runs_dir.exists() else []
    selected_weight = st.selectbox(
        "Weights File", found_weights or ["No weights found - Simulation Mode"]
    )
    use_onnx = st.toggle("Accelerate via ONNX", value=False)

    model_dict = load_shm_model(selected_weight, use_onnx)

    if model_dict["type"] == "yolo":
        st.success("✅ PyTorch YOLO loaded")
    elif model_dict["type"] == "onnx":
        st.success("⚡ ONNX session active")
    else:
        st.warning(
            "⚠️ Weights file missing.\nRunning in **Simulation Mode** — "
            "results are NOT model predictions."
        )

# Main Header Banner
st.markdown(
    "<div class='main-header'>Structural Health Monitoring Dashboard</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sub-header'>AI Vision Inspection Pipeline & Automated Safety Assessment</div>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# PAGE 1: SINGLE IMAGE ANALYSIS
# -----------------------------------------------------------------------------
if page == "Single Image Analysis":
    col_left, col_right = st.columns([1, 1.2], gap="large")

    if "single_analysis" not in st.session_state:
        st.session_state.single_analysis = None

    with col_left:
        st.subheader("📷 Image Source & Parameters")

        input_mode = st.radio(
            "Select Input Source:", ["Upload File", "Camera Input"], horizontal=True
        )
        uploaded_file = (
            st.file_uploader(
                "Upload concrete surface image", type=["jpg", "jpeg", "png", "bmp"]
            )
            if input_mode == "Upload File"
            else st.camera_input("Capture inspection frame")
        )

        with st.expander("⚙️ Camera Calibration", expanded=False):
            st.caption(
                "Physical measurement uses the pinhole model: "
                "W_mm = W_px × D / f. Calibrate f via Settings → Camera Helper."
            )
            dist_mm = st.number_input(
                "Distance to Structure (mm)",
                value=DEFAULT_CAMERA_DISTANCE_MM, step=100,
            )
            focal_px = st.number_input(
                "Camera Focal Length (px)",
                value=DEFAULT_FOCAL_LENGTH_PX, step=50,
            )

        analyze_btn = st.button("🔍 ANALYZE STRUCTURE", use_container_width=True)

    if analyze_btn:
        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file).convert("RGB")
            cname, conf, probs = classify_structure(image_pil, model_dict)
            camera_params = {"dist": dist_mm, "focal": focal_px}
            metrics = evaluate_metrics(cname, conf, image_pil, camera_params)

            # Crack overlay comes from the measurement module itself. It is
            # shown whenever a crack was LOCATED, even if the width failed
            # validation -- an inspector still needs to see where the crack
            # is, and the caption states that the width is unresolved.
            annotated_arr = np.array(image_pil)
            detail = metrics.get("measurement_detail") or {}
            if metrics["is_cracked"] and detail.get("overlay_image") is not None:
                annotated_arr = cv2.cvtColor(
                    detail["overlay_image"], cv2.COLOR_BGR2RGB
                )
            annotated_pil = Image.fromarray(annotated_arr)

            st.session_state.single_analysis = {
                "annotated_arr": annotated_arr,
                "annotated_pil": annotated_pil,
                "class_name": cname,
                "confidence": conf,
                "probs": probs,
                "metrics": metrics,
                "camera_params": camera_params,
                "pdf_bytes": generate_pdf_report(
                    annotated_pil, cname, conf, metrics, camera_params
                ),
                "json_str": build_json_report(cname, conf, metrics, camera_params),
                "csv_str": build_csv_report(cname, conf, metrics),
            }
        else:
            st.warning("Please upload or capture an image first.")

    with col_right:
        if st.session_state.single_analysis:
            data = st.session_state.single_analysis
            metrics = data["metrics"]

            st.image(
                data["annotated_arr"],
                caption="Annotated Structural Inspection Result",
                use_container_width=True,
            )
            st.markdown(
                f"<div style='margin-bottom: 15px;'>"
                f"<span class='{metrics['badge_class']}'>{metrics['badge_label']}</span></div>",
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Structure Type", metrics["struct_type"].capitalize())
            m2.metric(
                "Confidence",
                f"{data['confidence'] * 100:.1f}%",
                delta=metrics.get("confidence_tier", ""),
                delta_color="off",
            )
            m3.metric("Safety Factor", f"{metrics['safety_factor']:.2f}")

            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=metrics["safety_factor"],
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Structural Safety Factor",
                           "font": {"color": "#00d4ff", "size": 15}},
                    number={"font": {"color": "#ffffff", "size": 30},
                            "valueformat": ".2f"},
                    gauge={
                        "axis": {"range": [0, 1], "tickwidth": 1,
                                 "tickcolor": "#ffffff"},
                        "bar": {"color": "#7b2cbf"},
                        "bgcolor": "#1a1a2e",
                        "steps": [
                            {"range": [0, 0.4], "color": "rgba(239, 68, 68, 0.75)"},
                            {"range": [0.4, 0.7], "color": "rgba(245, 158, 11, 0.75)"},
                            {"range": [0.7, 1.0], "color": "rgba(16, 185, 129, 0.75)"},
                        ],
                    },
                )
            )
            gauge_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=210, margin=dict(l=20, r=20, t=30, b=10),
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

            st.info(metrics["recommendation"])

            detail = metrics.get("measurement_detail") or {}
            if metrics.get("resolution_limited"):
                st.warning(
                    "📏 **CRACK WIDTH UNRESOLVED**: a crack was located but it is "
                    "thinner than this calibration can measure. Get closer to the "
                    "surface and use the full-resolution image, then re-run — the "
                    "width is reported as unresolved rather than guessed."
                )
                st.caption(detail.get("message", ""))
            elif metrics["severity"] == "uncertain":
                st.warning(
                    "⚠️ **LOW CONFIDENCE / MEASUREMENT REJECTED**: Perform manual "
                    "inspection and physical measurement before engineering decisions."
                )
            elif metrics["physical_width_mm"] is not None:
                with st.expander("📐 Physical Measurement Details", expanded=True):
                    md1, md2, md3 = st.columns(3)
                    md1.metric("Avg Width", f"{detail['avg_width_mm']:.3f} mm")
                    md2.metric("Max Width", f"{detail['max_width_mm']:.3f} mm")
                    md3.metric("Length", f"{detail.get('length_mm', 0.0):.1f} mm")
                    md4, md5, md6 = st.columns(3)
                    md4.metric("P90 Width", f"{detail.get('p90_width_mm', 0.0):.3f} mm")
                    md5.metric("Components", detail.get("n_components", 0))
                    orient = detail.get("orientation_deg")
                    md6.metric("Orientation", f"{orient:.0f}°" if orient is not None else "n/a")

            sorted_probs = sorted(data["probs"].items(), key=lambda x: x[1], reverse=True)
            bar_fig = px.bar(
                x=[x[1] * 100 for x in sorted_probs],
                y=[x[0] for x in sorted_probs],
                orientation="h",
                text=[f"{x[1] * 100:.1f}%" for x in sorted_probs],
                title="Class Probability Distribution",
                labels={"x": "Probability (%)", "y": "Class"},
            )
            bar_fig.update_traces(marker_color="#00d4ff", textposition="outside")
            bar_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(range=[0, 115], gridcolor="#2a2a40"),
                yaxis=dict(autorange="reversed"),
                height=260, margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            st.markdown("#### 📤 Export Results")
            e1, e2, e3 = st.columns(3)
            e1.download_button("JSON Report", data["json_str"],
                               file_name="shm_report.json",
                               mime="application/json",
                               key="btn_download_json_single")
            e2.download_button("CSV Report", data["csv_str"],
                               file_name="shm_report.csv",
                               mime="text/csv",
                               key="btn_download_csv_single")
            if data["pdf_bytes"]:
                e3.download_button("PDF Document", data["pdf_bytes"],
                                   file_name="shm_report.pdf",
                                   mime="application/pdf",
                                   key="btn_download_pdf_single")

# -----------------------------------------------------------------------------
# PAGE 2: BATCH PROCESSING
# -----------------------------------------------------------------------------
elif page == "Batch Processing":
    st.subheader("📦 Multi-Image Inspection Batch Pipeline")

    batch_files = st.file_uploader(
        "Upload multiple structural images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    with st.expander("⚙️ Batch Camera Calibration (for measurement)", expanded=False):
        b_dist = st.number_input("Batch Distance (mm)", value=DEFAULT_CAMERA_DISTANCE_MM, step=100)
        b_focal = st.number_input("Batch Focal Length (px)", value=DEFAULT_FOCAL_LENGTH_PX, step=50)
        b_measure = st.checkbox(
            "Run physical crack measurement on damaged predictions", value=True
        )

    if batch_files and st.button("RUN BATCH ANALYSIS"):
        results_list = []
        progress_bar = st.progress(0)

        for idx, file in enumerate(batch_files):
            img_pil = Image.open(file).convert("RGB")
            cname, conf, _ = classify_structure(img_pil, model_dict)
            met = evaluate_metrics(
                cname, conf, img_pil,
                {"dist": b_dist, "focal": b_focal} if b_measure else None,
            )
            detail = met.get("measurement_detail") or {}
            results_list.append({
                "Filename": file.name,
                "Class": cname,
                "Structure": met["struct_type"],
                "Condition": met["status_text"],
                "Severity": met["severity"],
                "Confidence (%)": round(conf * 100, 2),
                "Safety Factor": met["safety_factor"],
                "Measured Width (mm)": (
                    round(met["physical_width_mm"], 3)
                    if met["physical_width_mm"] is not None else None
                ),
                "Crack Length (mm)": (
                    round(detail["length_mm"], 1)
                    if detail.get("measured") else None
                ),
            })
            progress_bar.progress((idx + 1) / len(batch_files))

        df_batch = pd.DataFrame(results_list)

        # Fleet-level rollup from eng_utils (rating + recommendation).
        fleet = batch_structure_assessment([
            {
                "structure_type": r["Structure"],
                "condition": "cracked" if r["Condition"] == "DAMAGED" else "uncracked",
                "confidence": r["Confidence (%)"] / 100.0,
                "physical_width_mm": r["Measured Width (mm)"],
            }
            for r in results_list
        ])

        st.markdown("### 📊 Batch Inspection Summary")
        st.markdown(
            f"**Fleet Rating: {fleet['overall_rating']}** — "
            f"{fleet['recommendation']}"
        )
        b1, b2, b3, b4 = st.columns(4)
        total = len(df_batch)
        damaged = int(fleet["damaged_structures"])
        b1.metric("Total Images", total)
        b2.metric("Damaged Structures", damaged)
        b3.metric("Healthy Structures", total - damaged)
        b4.metric("Avg Safety Factor", f"{fleet['average_safety_factor']:.2f}")

        c1, c2 = st.columns(2)
        with c1:
            pie_fig = px.pie(
                df_batch, names="Condition", title="Damage Ratio",
                color="Condition",
                color_discrete_map={"HEALTHY": "#10b981", "DAMAGED": "#ef4444"},
            )
            pie_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white")
            )
            st.plotly_chart(pie_fig, use_container_width=True)
        with c2:
            sev_fig = px.histogram(
                df_batch, x="Severity", title="Severity Distribution",
                color="Severity",
                category_orders={
                    "Severity": ["none", "minor", "moderate", "uncertain",
                                 "severe", "critical", "collapse_risk"]
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            sev_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(sev_fig, use_container_width=True)

        st.markdown("### 📋 Inspection Logs")
        st.dataframe(df_batch, use_container_width=True)
        st.download_button(
            "📥 Download Batch Report CSV",
            df_batch.to_csv(index=False),
            file_name="batch_shm_summary.csv",
            mime="text/csv",
            key="btn_batch_csv",
        )

# -----------------------------------------------------------------------------
# PAGE 3: LIVE STREAM (real capture, measured FPS -- no simulated telemetry)
# -----------------------------------------------------------------------------
elif page == "Live Stream":
    st.subheader("📹 Real-Time Inspection Feed")

    col_ctrl, col_view = st.columns([1, 2])

    with col_ctrl:
        source_type = st.selectbox(
            "Video Feed Source", ["Webcam", "RTSP / IP Stream", "Video File"]
        )
        stream_url = None
        video_tmp = None
        if source_type == "RTSP / IP Stream":
            stream_url = st.text_input("RTSP Stream URL", "rtsp://user:pass@host:554/live")
        elif source_type == "Video File":
            vid_file = st.file_uploader("Upload Inspection Video", type=["mp4", "avi"])
            if vid_file is not None:
                video_tmp = vid_file

        st.divider()
        st.markdown("#### Live Telemetry")
        fps_slot = st.empty()
        verdict_slot = st.empty()

    with col_view:
        st.markdown("##### Camera Canvas")
        frame_slot = st.empty()

    start_stream = st.session_state.get("streaming", False)

    b1, b2 = st.columns(2)
    if b1.button("▶ START STREAM", use_container_width=True):
        st.session_state.streaming = True
        start_stream = True
    if b2.button("⏹ STOP STREAM", use_container_width=True):
        st.session_state.streaming = False
        start_stream = False

    if start_stream:
        source: int | str
        tmp_path = None
        if source_type == "Webcam":
            source = 0
        elif source_type == "RTSP / IP Stream":
            source = stream_url or ""
        else:
            if video_tmp is None:
                st.warning("Upload a video file first.")
                source = None
            else:
                # cv2 requires a real file path on Windows.
                suffix = Path(video_tmp.name).suffix or ".mp4"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(video_tmp.getvalue())
                tmp.close()
                tmp_path = tmp.name
                source = tmp_path

        if source is not None:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                st.error(f"Could not open source: {source}")
            else:
                st.info("Streaming... press ⏹ STOP STREAM to end.")
                fps_ema = 0.0
                frame_idx = 0
                while st.session_state.get("streaming", False):
                    t0 = time.perf_counter()
                    ret, frame = cap.read()
                    if not ret:
                        if source_type == "Video File":
                            break  # video finished
                        st.warning("Frame capture failed; retrying...")
                        time.sleep(0.1)
                        continue

                    frame_idx += 1
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cname, conf, _ = classify_structure(
                        Image.fromarray(frame_rgb), model_dict
                    )
                    met = evaluate_metrics(cname, conf)

                    # Live verdict HUD drawn on the actual frame.
                    color = (16, 185, 129) if not met["is_cracked"] else (239, 68, 68)
                    cv2.putText(frame_rgb, f"{cname.upper()}  {conf:.0%}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    cv2.putText(frame_rgb,
                                f"SF {met['safety_factor']:.2f} | {met['severity'].upper()}",
                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    dt = time.perf_counter() - t0
                    inst_fps = 1.0 / dt if dt > 0 else 0.0
                    fps_ema = inst_fps if frame_idx == 1 else (
                        0.9 * fps_ema + 0.1 * inst_fps
                    )

                    frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)
                    fps_slot.metric(
                        "Measured Inference FPS (EMA)", f"{fps_ema:.1f} FPS"
                    )
                    verdict_slot.metric(
                        "Latest Verdict",
                        f"{cname} ({met['severity']})",
                    )

                cap.release()
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)
                if source_type == "Video File":
                    st.session_state.streaming = False
    else:
        frame_slot.image(
            np.zeros((400, 600, 3), dtype=np.uint8),
            caption="Stream Standby",
            use_container_width=True,
        )

# -----------------------------------------------------------------------------
# PAGE 4: SETTINGS
# -----------------------------------------------------------------------------
elif page == "Settings":
    st.subheader("⚙️ System Configuration & Reference Standards")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Safety Thresholds", "Class Taxonomy", "Camera Helper", "Model Export"]
    )

    with tab1:
        st.markdown("##### ACI 224R-01 / Eurocode 2 Reference Limits")
        from config import STRUCTURE_THRESHOLDS

        st.table(pd.DataFrame([
            {
                "Structure": k.capitalize(),
                "Description": v.description,
                "Critical (mm)": v.critical_width,
                "Max Limit (mm)": v.max_allowable,
            }
            for k, v in STRUCTURE_THRESHOLDS.items()
        ]))

    with tab2:
        st.markdown("##### Classification Mapping Taxonomy")
        st.table(pd.DataFrame([
            {"ID": k, "Label": v,
             "Structure": STRUCTURE_MAP[v].capitalize(),
             "Condition": CONDITION_MAP[v].upper()}
            for k, v in CLASS_NAMES.items()
        ]))

    with tab3:
        st.markdown("##### Camera Focal Length Calculator")
        st.caption(
            "Computes focal length in pixels from physical lens specs "
            "(f_px = image_width_px × f_mm / sensor_width_mm)."
        )
        c1, c2, c3 = st.columns(3)
        sensor_w = c1.number_input("Sensor Width (mm)", value=36.0, step=0.5)
        focal_mm = c2.number_input("Lens Focal (mm)", value=50.0, step=1.0)
        img_w = c3.number_input("Image Width (px)", value=1920, step=100)
        if sensor_w > 0 and img_w > 0:
            st.metric("Effective Focal Length", f"{img_w * focal_mm / sensor_w:.1f} px")

    with tab4:
        st.markdown("##### Model Compilation & Export")
        st.caption(
            "Use the CLI for real exports: "
            "`python src/train.py --export --weights best.pt --format onnx`"
        )
        fmt = st.selectbox(
            "Target Platform Format", ["ONNX", "TensorRT", "TorchScript", "OpenVINO"]
        )
        st.code(
            f"python src/train.py --export --weights {selected_weight} "
            f"--format {fmt.lower().replace('tensorrt', 'engine')}",
            language="bash",
        )
