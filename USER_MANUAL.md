# SHM Vision — User Manual / دفترچه راهنمای کاربری
**Version 1.0 | 2026-08-07**

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

### Confidence Score / نمره اطمینان:
- **> 95%** — High confidence / اطمینان بالا
- **75-95%** — Medium confidence / اطمینان متوسط
- **< 75%** — Low confidence — manual check recommended / اطمینان پایین

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
**Solution:** Place your `best.pt` or `yolov8n-cls.pt` in the project root or `runs/classify/` directory.

### Issue: Measurement shows unrealistic width (>20mm)
**Solution:** Check camera calibration parameters. Ensure distance and focal length are correct.

### Issue: Crack overlay shows scattered red dots
**Solution:** This is normal noise from edge detection. The system filters noise but some texture may be highlighted. The main crack will still be visible.

### Issue: Low confidence score
**Solution:** Ensure image is well-lit and focused. Poor lighting or blurry images reduce accuracy.

### Issue: Stream not connecting
**Solution:** For RTSP, verify URL format: `rtsp://username:password@ip:port/stream`

---

## Contact / تماس

For questions or contributions, please open an issue on GitHub.

---

**SHM Vision Team | 2026**
