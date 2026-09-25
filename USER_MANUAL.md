# SHM Vision — User Manual / دفترچه راهنمای کاربری
**Version 2.0 | 2026-09-26**

> **v2.0 changed how results are reported.** Confidence no longer grades severity,
> sub-resolution cracks are reported as `UNRESOLVED` instead of a number, and crack
> widths are sub-pixel. See section 10 (*Understanding Results*) and the
> [v2.0 release notes](https://github.com/PedramNikfajam/Shm-Vision/releases/tag/v2.0).

---

## Table of Contents / فهرست مطالب

1. [Overview / معرفی](#1-overview--معرفی)
2. [Installation / نصب](#2-installation--نصب)
3. [Quick Start / راه‌اندازی سریع](#3-quick-start--راه‌اندازی-سریع)
4. [User Interface / رابط کاربری](#4-user-interface--رابط-کاربری)
5. [Single Image Analysis / تحلیل تک‌عکس](#5-single-image-analysis--تحلیل-تکعکس)
6. [Batch Processing / پردازش دسته‌ای](#6-batch-processing--پردازش-دستهای)
7. [Live Stream / پخش زنده](#7-live-stream--پخش-زنده)
8. [Settings / تنظیمات](#8-settings--تنظیمات)
9. [Camera Calibration / کالیبراسیون دوربین](#9-camera-calibration--کالیبراسیون-دوربین)
10. [Understanding Results / درک نتایج](#10-understanding-results--درک-نتایج)
11. [Export Reports / خروجی گزارش](#11-export-reports--خروجی-گزارش)
12. [Troubleshooting / رفع اشکال](#12-troubleshooting--رفع-اشکال)

---

## 1. Overview / معرفی

**SHM Vision** is an AI-powered Structural Health Monitoring system designed for civil engineers and inspection teams. It uses deep learning (YOLOv8) to classify structural damage in concrete surfaces and computer vision to measure crack widths physically.

**SHM Vision** یک سیستم پایش سلامت سازه مبتنی بر هوش مصنوعی است که برای مهندسان عمران و تیم‌های بازرسی طراحی شده. از یادگیری عمیق (YOLOv8) برای طبقه‌بندی آسیب‌های سازه‌ای در سطوح بتنی و بینایی کامپیوتر برای اندازه‌گیری فیزیکی عرض ترک‌ها استفاده می‌کند.

### Supported Structures / سازه‌های پشتیبانی‌شده:
- **Deck** — Bridge decks, floor slabs / دال‌های پل و کف
- **Pavement** — Roads, sidewalks / جاده و پیاده‌رو
- **Wall** — Concrete or masonry walls / دیوارهای بتنی و آجری

### Conditions / وضعیت‌ها:
- **Cracked** — Structural damage detected / آسیب سازه‌ای شناسایی شد
- **Uncracked** — Healthy condition / وضعیت سالم

---

## 2. Installation / نصب

### Prerequisites / پیش‌نیازها:
- Python 3.10 or higher
- pip package manager

### Step 1: Clone Repository / کلون کردن ریپازیتوری
```bash
git clone https://github.com/PedramNikfajam/Shm-Vision.git
cd Shm-Vision
```

### Step 2: Install Dependencies / نصب وابستگی‌ها
```bash
pip install -r requirements.txt
```

### Step 3: Download Model Weights / دانلود وزن‌های مدل
Download `yolov8n-cls.pt` (or your trained `best.pt`) and place it in the project root directory.

وزن‌های مدل را دانلود کرده و در ریشه پروژه قرار دهید.

---

## 3. Quick Start / راه‌اندازی سریع

Launch the application with Streamlit:

```bash
streamlit run app.py
```

The dashboard will automatically open in your browser at:
```
http://localhost:8501
```

---

## 4. User Interface / رابط کاربری

The sidebar (left) contains navigation and model settings:

| Section | Description |
|---------|-------------|
| **Navigation** | Switch between Single Image, Batch, Live Stream, Settings |
| **Model Selection** | Choose weights file (best.pt) and ONNX toggle |

---

## 5. Single Image Analysis / تحلیل تک‌عکس

### Steps / مراحل:
1. **Select Input Source** — Choose "Upload File" or "Camera Input"
2. **Upload Image** — Select JPG/PNG/BMP file
3. **Camera Calibration** (optional) — Set distance and focal length
4. **Click ANALYZE STRUCTURE** — Run AI analysis

### Results Displayed / نتایج نمایش‌داده‌شده:
- **Annotated Image** — Original image with red crack overlay
- **Condition Badge** — HEALTHY / MODERATE DAMAGE / SEVERE DAMAGE
- **Metrics** — Structure type, confidence %, safety factor
- **Safety Gauge** — Visual indicator (0 to 1)
- **Class Probabilities** — Horizontal bar chart of all 6 classes
- **Engineering Recommendation** — Actionable advice per ACI/Eurocode

### Camera Calibration Parameters / پارامترهای کالیبراسیون:
| Parameter | Default | Description |
|-----------|---------|-------------|
| Distance to Structure | 2000 mm | فاصله دوربین تا سطح سازه |
| Focal Length | 800 px | فاصله کانونی دوربین به پیکسل |

---

## 6. Batch Processing / پردازش دسته‌ای

1. Go to **Batch Processing** page
2. Upload multiple images (JPG/PNG)
3. Click **RUN BATCH ANALYSIS**
4. View summary cards and charts
5. Download consolidated CSV report

---

## 7. Live Stream / پخش زنده

1. Go to **Live Stream** page
2. Select source:
   - **Webcam** — Local camera
   - **RTSP / Drone** — IP camera stream URL
   - **Video File** — Pre-recorded video
3. Click **START STREAM**
4. Real-time classification with HUD overlay

---

## 8. Settings / تنظیمات

Three tabs available / سه تب در دسترس:

### Tab 1: Safety Thresholds / آستانه‌های ایمنی
View ACI/Eurocode reference limits for each structure type.

### Tab 2: Class Taxonomy / طبقه‌بندی کلاس‌ها
View the 6-class mapping table with IDs, labels, structures, and conditions.

### Tab 3: Model Export / خروجی مدل
Export trained model to ONNX, TensorRT, TorchScript, or OpenVINO format.

---

## 9. Camera Calibration / کالیبراسیون دوربین

For accurate physical crack width measurement, calibrate your camera:

### Formula:
```
focal_length_px = (image_width_px * focal_length_mm) / sensor_width_mm
```

### Example:
Full-frame camera (36mm sensor), 50mm lens, 1920px width:
```
focal_px = (1920 * 50) / 36 = 2667 px
```

### Default Values / مقادیر پیش‌فرض:
- Distance: 2000 mm (2 meters)
- Focal Length: 800 px

---

## 10. Understanding Results / درک نتایج

### Severity Levels / سطوح شدت:
| Severity | Safety Factor | Action Required |
|----------|---------------|-----------------|
| NONE | 1.00 | No action — routine inspection |
| MINOR | 0.85 | Document — re-inspect in 12 months |
| MODERATE | 0.65 | NDT testing within 30 days |
| SEVERE | 0.40 | Repair within 30 days |
| CRITICAL | 0.20 | Immediate repair — restrict access |
| COLLAPSE_RISK | 0.00 | Emergency — evacuate area |
| UNCERTAIN | 0.90 | Manual inspection required |

Severity grades **damage magnitude, not classifier confidence**. A detected crack is
graded `MINOR` as a conservative baseline; only a trustworthy physical measurement
can escalate it (`MINOR → MODERATE → SEVERE → CRITICAL → COLLAPSE_RISK`). A
measurement can never lower the grade below `MINOR`, so measuring a crack will not
improve its safety factor.

سقف شدت بر اساس **شدت آسیب** تعیین می‌شود، نه میزان اطمینان مدل. ترک شناسایی‌شده در
ابتدا با سطح `MINOR` ارزیابی می‌شود و فقط اندازه‌گیری فیزیکی معتبر می‌تواند آن را
تشدید کند. اندازه‌گیری هرگز شدت را کاهش نمی‌دهد.

### Confidence Score / نمره اطمینان:
| Tier | Range | Meaning |
|------|-------|---------|
| **High** | ≥ 85% | Verdict may drive a decision / اطمینان بالا |
| **Medium** | 75–85% | Verdict usable, verify on site / اطمینان متوسط |
| **Low** | < 75% | Flagged `UNCERTAIN` — manual check required / اطمینان پایین |

The **Low** tier is exactly the set of predictions that are flagged `UNCERTAIN`: a
prediction below 75% confidence never receives a clean verdict or a safety factor of
1.00. The tier is shown next to the confidence figure in the dashboard and included
in the JSON and CSV exports as `confidence_tier`.

رد آستانه ۷۵٪ دقیقاً همان دسته‌ای است که `UNCERTAIN` اعلام می‌شود.

### Crack width resolution / تفکیک‌پذیری عرض ترک:

If the detected crack is thinner than the imaging blur kernel, the width **cannot**
be measured at that distance. The dashboard reports this as `UNRESOLVED` rather than
printing a number, and never uses it for severity. To obtain a real measurement, move
closer and use the full-resolution image.

اگر ترک از کرنل محو‌شدگی نازک‌تر باشد، عرض آن قابل اندازه‌گیری نیست. سامانه به‌جای
عدد، وضعیت `UNRESOLVED` را گزارش می‌کند. برای اندازه‌گیری واقعی، نزدیک‌تر شوید و از
تصویر با بالاترین تفکیک استفاده کنید.

---

## 11. Export Reports / خروجی گزارش

Three export formats available / سه فرمت خروجی:

### JSON Report
Structured data for integration with other systems.

### CSV Report
Tabular data for Excel/spreadsheet analysis.

### PDF Document
Professional engineering report with:
- Clean annotated image
- High-resolution metrics table
- Engineering recommendation
- Timestamp and method

---

## 12. Troubleshooting / رفع اشکال

### Issue: "Weights file missing. Running in Simulation Mode"
**Solution:** Simulation Mode returns a **placeholder distribution, not a prediction**.
Download the trained checkpoint from the
[v2.0 release](https://github.com/PedramNikfajam/Shm-Vision/releases/tag/v2.0) and place
it in the project root or under `runs/classify/`:

```bash
curl -L -O https://github.com/PedramNikfajam/Shm-Vision/releases/download/v2.0/best.pt
sha256sum best.pt   # 0fec9340c033b6889b3a66485bcc1ed1bc1765399763c0d125e6051e29b712bb
```

Weights are excluded from git (`.gitignore` excludes `*.pt`), so a fresh clone has none
until you download it.

### Issue: Width reported as `UNRESOLVED`
**Solution:** The crack is real but thinner than the imaging blur kernel at the current
distance, so no width can be trusted. Get closer to the surface (≈0.5 m), use the
full-resolution image, and increase the focal-length figure to match your optics. The
system deliberately withholds a number here rather than inventing one.

### Issue: Measurement shows an implausible width
**Solution:** Check the calibration first — `W = px × D / f`. A width far above
`CRACK_MEASUREMENT["max_plausible_avg_width_mm"]` (50 mm) is rejected as texture or
shadow. Very wide reported values usually mean the calibration is far too coarse: at
the default 2 m / 800 px, 1 px = 2.5 mm, which cannot resolve a 0.3 mm wall crack.

### Issue: Severity says `MINOR` even at 99% confidence
**Solution:** This is intended. Confidence expresses certainty of the *label*, not
damage magnitude, so a confident detection is graded at the conservative `MINOR`
baseline and only a valid physical measurement escalates it. See section 10,
*Understanding Results*.

### Issue: A cracked surface is reported `HEALTHY`
**Solution:** Be aware of the known model limitation — `deck_cracked` recall is 0.70,
so roughly 3 in 10 cracked decks are classified as uncracked. Treat a `HEALTHY`
verdict on a deck as unconfirmed and verify manually. This is a model-capacity issue
that v2 does not fix.

### Issue: Crack overlay shows scattered red dots
**Solution:** Small texture fragments can survive filtering. The red fill marks exactly
the detected crack pixels and the yellow line is the measured centreline; the main
crack remains the dominant feature.

### Issue: Low confidence score
**Solution:** Ensure the image is well-lit and in focus, and that it resembles the
training domain (close-up concrete tiles, per SDNET2018). Below 75% the result is
flagged `UNCERTAIN` by design.

### Issue: `RuntimeError: Invalid device string: '0'`
**Solution:** You are on a CPU-only PyTorch install. `--device` now defaults to `auto`,
which selects CUDA when available and CPU otherwise. You can also pass `--device cpu`
explicitly.

### Issue: Stream not connecting
**Solution:** For RTSP, verify the URL format: `rtsp://username:password@ip:port/stream`

---

## Contact / تماس

For questions or contributions, please open an issue on GitHub:
**https://github.com/PedramNikfajam/Shm-Vision**

---

**SHM Vision — Pedram Nikfarjam | v2.0 | 2026**
