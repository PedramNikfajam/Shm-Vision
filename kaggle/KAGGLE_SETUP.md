# Kaggle Setup Guide — SHM Vision (SDNET2018)

Train on Kaggle GPU using the SDNET2018 dataset. **No code upload needed** — project sources are embedded in the notebook.

> **Retrain note (v2.2):** after the 2026-09 run (top1=0.778, best epoch 7/27,
> deck_cracked recall 0.56, z_other recall 0.275), the pipeline now uses:
> **yolov8s-cls @ 256px** (SDNET-native resolution — 320 was upscaling),
> **patience 30**, **cos_lr**, **flipud 0.5**, **label_smoothing 0.05**, and
> **z_other oversampled to 3200/class** to match the structural classes.
> Split is leak-free grouped (tiles of one surface never cross train/val/test).

**Optional — abstain class:** attach ANY second Kaggle dataset with diverse
everyday photos (people, sky, grass, cars...). Cell 3b samples up to 3200 into
the `z_other` class (then oversampled to match the structural classes) so the
model can say "not a structural surface" on random pictures instead of forcing
deck/pavement/wall. Skip cell 3b for 6-class.

Good candidate datasets to attach (search Kaggle by name, pick any one):
- **Intel Image Classification** (`puneet6060/intel-image-classification`) — city scenes, forest, sea, buildings
- **Animals-10** or any animal/landscape dataset
- Or upload ~500 of your own phone photos in a zip

---

## 1. What you need

| Item | Source |
|------|--------|
| `kaggle/shm_vision_kaggle.ipynb` | This repo (self-contained) |
| SDNET2018 dataset | Already on Kaggle: `structural-defects-network-concrete-crack-images` |

Nothing else. No zip, no weights upload — `yolov8s-cls.pt` downloads automatically at train time (needs Internet ON).

> Notebook embeds `src/train.py`, `scripts/prepare_data.py`, `scripts/evaluate.py`, `config/hyperparams.yaml`. After editing any of these locally, regenerate: `python kaggle/_build_notebook.py`.

## 2. Create the notebook

1. Kaggle → **Code** → **New Notebook**.
2. **File → Import Notebook** → `kaggle/shm_vision_kaggle.ipynb` from this repo.
3. Notebook **Settings** (right panel):
   - **Accelerator → GPU T4 x2** (or P100).
   - **Internet → ON** (for the one-time `yolov8n-cls.pt` download).
   - **Add Input** → *Datasets* → `structural-defects-network-concrete-crack-images` (your SDNET2018 dataset).

## 3. Run cells top-to-bottom

| Cell | What it does |
|------|-------------|
| 1 | Auto-discovers SDNET2018 under `/kaggle/input` (slug-agnostic) |
| 2 | Materializes embedded project code (`src/train.py`, `scripts/prepare_data.py`, `scripts/evaluate.py`, `config/hyperparams.yaml`) into `/kaggle/working/shm-vision` |
| 3 | Normalizes SDNET naming: `Decks/Cracked/Non-cracked` → `deck/cracked/uncracked` (symlinks, zero-copy) |
| 3b | **Optional** `z_other` abstain class from any second attached dataset (≤3200 diverse images) |
| 4 | Runs `prepare_data.py`: leak-free grouped split by surface ID, balanced + oversampled to 3200/class, seed 42 |
| 5 | Verifies splits + confirms class-id order matches `config.py CLASS_NAMES` |
| 6 | **TRAINING** — 150 epochs, yolov8s-cls @ 256px, batch 64, patience 30, cosine LR. Log: `/kaggle/working/training_console.log` |
| 7 | Training curves from `results.csv` |
| 8 | Evaluates `best.pt` on test split → per-class P/R/F1, confusion matrix, ECE |
| 9 | Packages artifacts into `/kaggle/working/shm_vision_artifacts/` |
| 10 | Smoke test: reload `best.pt`, predict one test image |

Cell 6 duration: ~2–4 h on T4 (256px epochs are fast). Session limit 12 h — comfortable.

## 4. Download results

Notebook **Output** tab → `shm_vision_artifacts/` containing:
- `best.pt` — trained model
- `last.pt` — resume checkpoint
- `evaluation_results.json`, `per_class_metrics.csv`, `confusion_matrix.png`
- `results.png`, `results.csv` — training curves
- `training_console.log`

## 5. Use at home

Drop `best.pt` into the project root (or `runs/classify/shm_classification/weights/`), then:

```powershell
streamlit run app.py
```

Optional: `python src/train.py --export --weights runs/classify/shm_classification/weights/best.pt --format onnx`.

## 6. Resume after an interrupted session

1. Download previous session's `shm_vision_artifacts/` (contains `last.pt`).
2. Upload `last.pt` as a small Kaggle dataset (or attach the old notebook's output).
3. In cell 6, copy it into place before training:
   ```python
   shutil.copy2("/kaggle/input/<your-upload-slug>/last.pt",
                PROJECT / "runs/classify/shm_classification/weights/last.pt")
   RESUME = True
   ```
4. Re-run cells 1–6.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Cell 1: assert SDNET not found | Dataset not attached — Settings → Add Input. |
| Cell 2 env check: CUDA False | Enable GPU accelerator; check remaining GPU quota. |
| Train: yolov8s-cls.pt download fails | Internet OFF — Settings → Internet ON. |
| Training OOM | Lower `BATCH = 64` → `32` in cell 6. |
| Session died mid-training | Section 6 above. |
