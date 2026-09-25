#!/usr/bin/env python3
"""Generator for kaggle/shm_vision_kaggle.ipynb — self-contained SDNET edition.

Embeds project code (train.py, prepare_data.py, evaluate.py, hyperparams.yaml)
as base64 so the notebook needs NO code upload: attach the SDNET2018 dataset,
enable GPU, run.
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EMBED = {
    "src/train.py": ROOT / "src/train.py",
    "scripts/prepare_data.py": ROOT / "scripts/prepare_data.py",
    "scripts/evaluate.py": ROOT / "scripts/evaluate.py",
    "config/hyperparams.yaml": ROOT / "config/hyperparams.yaml",
}

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}

embedded_cell = f"""\
# 2. Materialize project code (embedded in this notebook - no upload needed)
import base64
from pathlib import Path

PROJECT = Path("/kaggle/working/shm-vision")
PROJECT.mkdir(parents=True, exist_ok=True)

FILES = {json.dumps({rel: base64.b64encode(path.read_bytes()).decode() for rel, path in EMBED.items()})}

for rel, b64 in FILES.items():  # ponytail: base64 blob for zero-quoting-bug embedding; view sources in the repo
    dest = PROJECT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(b64))
    print("wrote", dest)
"""

cells = [
    md("""\
# SHM Vision — Kaggle GPU Training (SDNET2018)

Trains the YOLOv8-cls 6-class concrete damage classifier on the **SDNET2018** Kaggle dataset, evaluates it, and produces downloadable artifacts. No code upload needed — project sources are embedded in this notebook.

**Setup (one-time):**
1. New Notebook → **Settings → Accelerator → GPU** (T4 x2 or P100).
2. **Add Input** → attach the SDNET2018 dataset (`structural-defects-network-concrete-crack-images`).
3. Run cells top-to-bottom.

Flow: locate SDNET → normalize folder names → optional out-of-scope class → balanced 80/10/10 split (3200/class, oversampled) → train (yolov8s-cls @ 256px) → evaluate on test → package `best.pt` for download.

**Optional — out-of-scope abstain class:** attach ANY second dataset with diverse
everyday photos (people, sky, grass, cars, wood...) and cell 3b automatically
samples up to 3200 of them into the `z_other` class (oversampled to match the
structural classes) so the model can say "not a structural surface" instead of
forcing a concrete verdict. Skip cell 3b to train the plain 6-class model."""),
    md("""\
## Step-by-step

1. **Settings** (right panel): Accelerator → **GPU T4 x2** (or P100), Internet → **ON**.
2. **Add Input** → attach SDNET2018 (`structural-defects-network-concrete-crack-images`).
3. *(Optional, for `z_other`)* Add Input → attach any everyday-photos dataset (e.g. Intel Image Classification).
4. **Run all** (or run cells top-to-bottom). Cell 6 (training) takes ~2–4 h on a T4.
5. Watch `training_console.log` / Output tab. After cell 9, download `shm_vision_artifacts/`.
6. Drop `best.pt` into `runs/classify/shm_classification/weights/` at home and `streamlit run app.py`.

See `kaggle/KAGGLE_SETUP.md` for the full guide + resume/troubleshooting."""),
    code("""\
# 1. Locate the SDNET2018 dataset under /kaggle/input (auto-discover, slug-agnostic)
from pathlib import Path

CANDIDATES = [
    d for d in Path("/kaggle/input").rglob("*")
    if d.is_dir() and {c.name.lower() for c in d.iterdir() if c.is_dir()} >= {"decks", "pavements", "walls"}
]
assert CANDIDATES, "SDNET2018 not found - add the dataset via Add Input, then re-run."
RAW = CANDIDATES[0]
print("SDNET root:", RAW)
for s in sorted(RAW.iterdir()):
    if s.is_dir():
        print(" ", s.name, "->", [c.name for c in sorted(s.iterdir())])"""),
    code(embedded_cell),
    code('''\
# 2b. Environment check: fail fast if GPU missing
import sys

import torch, torchvision, ultralytics

print(f"python {sys.version.split()[0]} | torch {torch.__version__} | "
      f"torchvision {torchvision.__version__} | ultralytics {ultralytics.__version__}")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA NOT AVAILABLE - enable GPU accelerator in Notebook Settings, then re-run.")'''),
    code('''\
# 3. Normalize SDNET naming (Decks/Cracked/Non-cracked) -> prepare_data.py layout (deck/cracked/uncracked)
#    Symlinks: zero-copy, works because prepare_data.py only reads through them.
import os

NORM = Path("/kaggle/working/sdnet_normalized")
MAP = {"decks": "deck", "pavements": "pavements", "walls": "walls"}

if NORM.exists():
    import shutil; shutil.rmtree(NORM)

for src_d in sorted(RAW.iterdir()):
    if not src_d.is_dir():
        continue
    structure = MAP.get(src_d.name.lower())
    if structure is None:
        continue
    for cond_d in sorted(src_d.iterdir()):
        if not cond_d.is_dir():
            continue
        low = cond_d.name.lower()
        if "non" in low:
            cond = "uncracked"
        elif "crack" in low:
            cond = "cracked"
        else:
            continue
        dest = NORM / structure / cond
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(cond_d, dest)
        except OSError:
            import shutil
            shutil.copytree(cond_d, dest, dirs_exist_ok=True)
        print(dest, "->", cond_d)
print("Normalized tree ready.")'''),
    code('''\
# 3b. (OPTIONAL) Build the out-of-scope "z_other" abstain class.
#     Attaches any OTHER image dataset (everyday photos: sky, grass, people,
#     cars, wood...) so the model can abstain on non-structural surfaces.
#     Budget matches the structural classes (~3200): with fewer samples the
#     model almost never predicts z_other (observed recall 0.275 @ 2000).
#     prepare_data --balance --min-per-class fills any residual gap by
#     oversampling. Delete/ignore this cell to train the plain 6-class model.
import random

MAX_OTHER = 3200
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

other_pool = []
for ds in sorted(Path("/kaggle/input").iterdir()):
    if ds.resolve() == RAW.resolve():
        continue  # skip SDNET itself
    imgs = [p for p in ds.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file()]
    if imgs:
        other_pool.extend(imgs)

if other_pool:
    random.seed(42)
    sample = random.sample(other_pool, min(MAX_OTHER, len(other_pool)))
    dest = NORM / "other" / "mixed"
    dest.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(sample):
        try:
            os.symlink(p, dest / f"{i:05d}_{p.name}")
        except OSError:
            import shutil
            shutil.copy2(p, dest / f"{i:05d}_{p.name}")
    print(f"z_other class: linked {len(sample)} diverse images from {len({p.parent for p in sample})} folders")
else:
    print("No extra dataset attached - training the plain 6-class model (fine).")'''),
    code('''\
# 4. Build data/processed: balanced 80/10/10 split, max 3200 images/class (~16k train / 2k / 2k)
import subprocess, sys

cmd = [
    sys.executable, str(PROJECT / "scripts/prepare_data.py"),
    "--raw", str(NORM),
    "--output", str(PROJECT / "data/processed"),
    "--balance", "--max-per-class", "3200", "--min-per-class", "3200",
    "--seed", "42",
]
p = subprocess.run(cmd, capture_output=True, text=True)
print(p.stdout[-3000:])
if p.returncode != 0:
    print(p.stderr[-2000:])
assert p.returncode == 0, "prepare_data failed - see output above."
DATA = PROJECT / "data/processed"'''),
    code('''\
# 5. Verify splits + class names match config.py CLASS_NAMES (YOLO sorts folders alphabetically;
#    deck_* < pavement_* < wall_* < z_other, so ids 0-5 always match config; id 6 = z_other only
#    when cell 3b produced that class)
import sys
from pathlib import Path

sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))  # for config import inside train.py
EXPECTED = {"train", "val", "test"}
BASE_CLASSES = ["deck_cracked", "deck_uncracked", "pavement_cracked",
                "pavement_uncracked", "wall_cracked", "wall_uncracked"]
ALLOWED = set(BASE_CLASSES) | {"z_other"}

for split in sorted(EXPECTED):
    classes = sorted(p.name for p in (DATA / split).iterdir() if p.is_dir())
    n = sum(len(list((DATA / split / c).glob("*"))) for c in classes)
    assert set(classes) <= ALLOWED and classes == sorted(classes), f"{split}: unexpected class folders: {classes}"
    print(f"{split}: {n} images, classes={classes}")
print("Dataset verified. Class id order matches config.py CLASS_NAMES.")'''),
    code('''\
# 6. TRAINING - 150 epochs, batch 64, patience 20, device auto (GPU), workers auto (4 on Linux)
#    T4: ~2-4 h. Console streamed to /kaggle/working/training_console.log
#    Resume after an interrupted session: keep last.pt in shm_vision_artifacts, set RESUME=True.
import subprocess, sys, time

EPOCHS = 150
BATCH = 64
RESUME = False

if RESUME:
    cmd = [sys.executable, str(PROJECT / "src/train.py"),
           "--resume", str(PROJECT / "runs/classify/shm_classification/weights/last.pt")]
else:
    cmd = [
        sys.executable, str(PROJECT / "src/train.py"),
        "--data", str(DATA),
        "--config", str(PROJECT / "config/hyperparams.yaml"),
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH),
        "--name", "shm_classification",
    ]

print(" ".join(cmd))
t0 = time.time()
with open("/kaggle/working/training_console.log", "a") as log:
    p = subprocess.run(cmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
print(f"train exit code: {p.returncode} after {(time.time() - t0) / 60:.1f} min")
if p.returncode != 0:
    print(open("/kaggle/working/training_console.log").read()[-3000:])
assert p.returncode == 0, "Training failed - see training_console.log."'''),
    code('''\
# 7. Training curves from ultralytics results.csv
import pandas as pd
import plotly.express as px

run_dir = PROJECT / "runs/classify/shm_classification"
df = pd.read_csv(run_dir / "results.csv")
df.columns = [c.strip() for c in df.columns]

metrics = [m for m in ("train/loss", "val/loss", "metrics/accuracy_top1", "metrics/accuracy_top5")
           if m in df.columns]
px.line(df, x="epoch", y=metrics, title="Training curves").show()'''),
    code('''\
# 8. Evaluate best.pt on the test split (top-1/top-5, per-class P/R/F1, confusion matrix, ECE)
import subprocess, sys

cmd = [
    sys.executable, str(PROJECT / "scripts/evaluate.py"),
    "--weights", str(PROJECT / "runs/classify/shm_classification/weights/best.pt"),
    "--data", str(DATA / "test"),
    "--output", str(PROJECT / "runs/evaluation"),
    "--device", "0",
]
p = subprocess.run(cmd, capture_output=True, text=True)
print(p.stdout[-4000:])
if p.stderr:
    print(p.stderr[-2000:])
assert p.returncode == 0, "Evaluation failed - see output above."'''),
    code('''\
# 9. Package artifacts for download via the notebook Output tab
import shutil

ART = Path("/kaggle/working/shm_vision_artifacts")
ART.mkdir(exist_ok=True)

run_dir = PROJECT / "runs/classify/shm_classification"
for f in [
    run_dir / "weights/best.pt",
    run_dir / "weights/last.pt",
    run_dir / "results.csv",
    run_dir / "results.png",
    run_dir / "confusion_matrix.png",
    PROJECT / "runs/evaluation/evaluation_results.json",
    PROJECT / "runs/evaluation/per_class_metrics.csv",
    PROJECT / "runs/evaluation/confusion_matrix.png",
    "/kaggle/working/training_console.log",
]:
    if Path(f).exists():
        shutil.copy2(f, ART / Path(f).name)
    else:
        print("missing (skipped):", f)

print("Artifacts:", sorted(p.name for p in ART.iterdir()))'''),
    code('''\
# 10. Smoke test: reload best.pt and predict on one test image
from ultralytics import YOLO

best = YOLO(PROJECT / "runs/classify/shm_classification/weights/best.pt")
img = next((DATA / "test" / "wall_cracked").glob("*.jpg"))
r = best.predict(str(img), device=0, verbose=False)[0]
name = best.names[int(r.probs.top1)]
print(f"{img.name}: {name} (conf={float(r.probs.top1conf):.3f})")
print("Model classes:", best.names)
print("Model reload + inference OK. Download shm_vision_artifacts from the Output tab.")'''),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(cells):
    c["id"] = f"cell-{i}"

out = ROOT / "kaggle" / "shm_vision_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size} bytes, {len(cells)} cells)")
