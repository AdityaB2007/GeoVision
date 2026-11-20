# GeoVision

**GeoVision** is an end-to-end computer vision system for classifying Copernicus **Sentinel-2** satellite imagery (EuroSAT) into **10 land-use and land-cover (LULC) categories**, combining a progressive PyTorch ML pipeline with a FastAPI inference server and a vanilla JavaScript web interface.

---

## Key Features

- **Verified Dataset & Leak-Free Splitting (`src/data/`):** Automatic download from Zenodo (`DOI 10.5281/zenodo.7711810`, verified MD5) with HuggingFace `torchgeo/eurosat` fallback, official & spatial split manifests, $D_4$ dihedral satellite augmentations (`RandomRotate90`, H/V flips), and 13-band Sentinel-2 helpers (`compute_ndvi`, `ms_to_rgb`).
- **Three-Tier Progressive Modeling (`src/models/`):**
  1. **Classical Baseline (`baseline_logreg`):** 122 handcrafted RGB/HSV histogram, 2×2 spatial quadrant, and luminance gradient-magnitude features + `StandardScaler` + `LogisticRegression`.
  2. **Custom CNN (`GeoVisionCNN`):** 4-block Conv–BatchNorm–ReLU–MaxPool architecture (469,354 parameters) with Global Average Pooling designed from scratch for `64×64` patches.
  3. **Transfer Learning (`GeoVisionTransferNet`):** Two-stage fine-tuning (`ResNet-18`, `ResNet-50`, `EfficientNet-B0`) using Stage 1 linear probing followed by Stage 2 selective `layer4` unfreezing with differential learning rates.
- **Quantitative Evaluation & Calibration (`src/evaluation/`):** Accuracy, Top-3 Accuracy, Macro/Weighted F1, Confusion Matrices, One-vs-Rest ROC curves, 10-bin Expected Calibration Error (ECE) reliability diagrams, and Top-15 confident misclassification error reports.
- **Visual Explainability with Grad-CAM (`src/explainability/`):** Hook-based Grad-CAM heatmaps across `GeoVisionCNN`, `ResNet-18/50`, and `EfficientNet-B0`, supporting perceptual colormaps, opacity blending, 10-class test galleries, and counterfactual class explanations (`POST /explain`).
- **Production FastAPI + Vanilla JS App (`backend/`, `frontend/`):** Serves the interactive dark-mode UI directly alongside `/predict`, `/explain`, `/health`, `/model-info`, and `/classes`, reconstructing exact training preprocessing from self-contained `.pt` checkpoints.

---

## Empirical Benchmark Results (Full 5,400-Image Official Test Split)

All three models were evaluated on the exact same held-out **5,400-image** TorchGeo official test split (`eurosat-test.txt`):

| Model | Train Samples | Input Size | Top-1 Accuracy | Top-3 Accuracy | Macro F1 | Macro ROC-AUC | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline_logreg`** (122 features) | 16,200 | 64×64 | 85.00% | 97.70% | 84.35% | 0.9831 | **0.0262** |
| **`custom_cnn`** (`GeoVisionCNN`) | 4,000 | 64×64 | 87.43% | 98.78% | 86.95% | 0.9900 | 0.0641 |
| **`transfer_resnet18`** (`ResNet-18`, 2-stage) | 2,500 | 128×128 | **93.28%** | **99.46%** | **93.13%** | **0.9966** | 0.0506 |

Diagnostic plots & JSON reports are saved in `outputs/evaluation/` (including [`model_comparison.png`](outputs/evaluation/model_comparison.png)).

---

## Project Structure

```text
GeoVision/
├── backend/
│   ├── app.py                  # FastAPI endpoints (/health, /predict, /explain, /classes, /model-info) + static UI mount
│   ├── inference.py            # Checkpoint loading, PyTorch inference & thread-safe Grad-CAM engine
│   └── schemas.py              # Pydantic v2 response contracts
├── docs/
│   ├── EUROSAT_DATASET_RESEARCH.md   # Deep research on EuroSAT Zenodo/HF sources, 13 bands & splits
│   └── ML_CONCEPTS_GUIDE.md          # End-to-end ML/DL teaching curriculum mapped to code
├── frontend/
│   ├── index.html              # Interactive satellite upload, Grad-CAM overlay & 10-class probability UI
│   ├── script.js               # Drag-and-drop / file picker wired to FastAPI /health & /explain
│   └── styles.css
├── outputs/
│   ├── evaluation/             # Confusion matrices, ROC curves, reliability diagrams, comparison charts
│   ├── explainability/         # 10-class Grad-CAM gallery & per-sample attribution metadata
│   └── training/               # Epoch loss/accuracy histories & learning curves
├── src/
│   ├── config.py               # Paths, 10 EuroSAT classes, 13-band metadata, ImageNet stats
│   ├── data/
│   │   ├── download.py         # Zenodo + HuggingFace downloader with MD5 verification
│   │   ├── preprocessing.py    # Normalization, D4 satellite augmentation, NDVI & MS->RGB helpers
│   │   └── dataset.py          # Official, spatial & stratified splits + RGB/MS PyTorch Datasets
│   ├── models/
│   │   ├── baseline.py         # Handcrafted 122-feature extractor & sklearn classifier
│   │   ├── custom_cnn.py       # 4-stage GeoVisionCNN from scratch
│   │   └── transfer.py         # ResNet-18/50 & EfficientNet-B0 two-stage transfer learning
│   ├── explainability/
│   │   ├── gradcam.py          # Hook-based Grad-CAM attribution & target-layer resolver
│   │   ├── visualization.py    # Perceptual colormaps, alpha-blending & gallery renderer
│   │   └── generate_gallery.py # CLI to generate 10-class EuroSAT Grad-CAM gallery
│   ├── training/
│   │   └── train.py            # Unified training CLI with AdamW, cosine LR & label smoothing
│   └── evaluation/
│       └── evaluate.py         # Metrics, calibration ECE, error analysis & comparison report
├── tests/
│   ├── test_preprocessing_and_models.py
│   ├── test_gradcam.py
│   └── test_api.py
└── requirements.txt
```

---

## Quickstart

### 1. Create Environment & Install Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Download EuroSAT Sentinel-2 RGB & Official Splits
```powershell
.\.venv\Scripts\python.exe -m src.data.download
```

### 3. Train & Evaluate Models

**CPU-friendly settings (reproduces the benchmark table above):**
```powershell
.\.venv\Scripts\python.exe -m src.training.train --model baseline --split-strategy official
.\.venv\Scripts\python.exe -m src.training.train --model custom_cnn --split-strategy official --max-per-class 400 --epochs 10 --input-size 64
.\.venv\Scripts\python.exe -m src.training.train --model transfer --backbone resnet18 --split-strategy official --max-per-class 250 --input-size 128 --stage1-epochs 2 --stage2-epochs 3 --set-best
```

**Full-dataset GPU training (recommended when CUDA is available):**
```powershell
.\.venv\Scripts\python.exe -m src.training.train --model custom_cnn --split-strategy official --epochs 25 --input-size 64
.\.venv\Scripts\python.exe -m src.training.train --model transfer --backbone resnet18 --split-strategy official --input-size 224 --stage1-epochs 5 --stage2-epochs 12 --set-best
```

**Evaluate across the full 5,400-image official test split & regenerate comparisons:**
```powershell
.\.venv\Scripts\python.exe -m src.evaluation.evaluate --checkpoint models/best_model.pt --split-strategy official
.\.venv\Scripts\python.exe -m src.evaluation.evaluate --compare
```

**Generate the 10-class Grad-CAM attention gallery (`outputs/explainability/gradcam_gallery.png`):**
```powershell
.\.venv\Scripts\python.exe -m src.explainability.generate_gallery --checkpoint models/best_model.pt --colormap turbo --alpha 0.45
```

### 4. Launch the Interactive Web App
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --reload
```
Then open **http://127.0.0.1:8000** (or Swagger API docs at **http://127.0.0.1:8000/docs**) to classify satellite imagery, inspect **Grad-CAM heatmaps** (`Overlay` / `Heatmap` / `Original`), adjust overlay opacity, or query counterfactual explanations (`POST /explain?target_class=Highway`).

### 5. Run Tests
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## Learning Resources in This Repo

- **[EuroSAT Sentinel-2 Deep Research Guide](docs/EUROSAT_DATASET_RESEARCH.md):** Where to find the dataset (Zenodo DOI, HuggingFace, TorchGeo), why `madm.dfki.de` fails, all 13 Sentinel-2 spectral bands, and random vs. spatial splits.
- **[GeoVision ML & Deep Learning Curriculum](docs/ML_CONCEPTS_GUIDE.md):** Detailed concept-by-concept breakdown of normalization, data leakage, $D_4$ augmentation, convolutions, transfer learning, calibration, and FastAPI serving.
