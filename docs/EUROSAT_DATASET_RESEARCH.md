# Deep Research Guide: Finding & Using the EuroSAT Sentinel-2 Dataset

This guide documents verified official sources, broken mirror workarounds, spectral band physics, dataset splits, and licensing for **EuroSAT** (Helber et al., IGARSS 2018 / IEEE JSTARS 2019).

---

## 1. Quick Summary

| Property | Value |
| :--- | :--- |
| **Satellite Mission** | Copernicus **Sentinel-2** (Multi-Spectral Instrument, MSI) |
| **Total Images** | **27,000** geo-referenced, labeled image patches |
| **Classes** | **10** Land-Use & Land-Cover (LULC) categories |
| **Spatial Size** | **64 × 64 pixels** per patch |
| **Ground Sampling Distance** | **10 m/pixel** (optical RGB & B08 NIR) -> **640 m × 640 m** per patch |
| **Geographic Coverage** | 34 European countries |
| **License** | MIT License (dataset) + Copernicus Sentinel Data Terms |
| **Official DOI** | `10.5281/zenodo.7711810` |

---

## 2. Official Download Sources (And Why Default Loaders Fail)

### Why `madm.dfki.de` / `torchvision.datasets.EuroSAT(download=True)` Often Fails
Originally, the authors hosted `EuroSAT.zip` and `EuroSATallBands.zip` on a DFKI server (`madm.dfki.de`). That server has become deprecated/unstable (frequent SSL certificate errors or timeouts), causing older tutorials and `torchvision.datasets.EuroSAT(..., download=True)` calls to fail.

### Primary Official Archive — Zenodo (`10.5281/zenodo.7711810`)
The dataset is officially mirrored and version-archived on **Zenodo** (`https://zenodo.org/records/7711810`):

| Archive | Contents | Format | Verified Size | Verified MD5 Checksum | Direct Download URL |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`EuroSAT_RGB.zip`** | 3 optical bands (B04, B03, B02) | 8-bit `.jpg` | **90.27 MB** | `f46e308c4d50d4bf32fedad2d3d62f3b` | `https://zenodo.org/api/records/7711810/files/EuroSAT_RGB.zip/content` |
| **`EuroSAT_MS.zip`** | All **13 spectral bands** | GeoTIFF `.tif` | **1,969.72 MB** (~1.97 GB) | `091174add3c8e680a49244acf185b9f0` | `https://zenodo.org/api/records/7711810/files/EuroSAT_MS.zip/content` |

### Verified Ecosystem Mirrors & Loaders
1. **GeoVision Downloader (Implemented in `src/data/download.py`):**
   Streams `EuroSAT_RGB.zip` from Zenodo, verifies the MD5 checksum `f46e308c4d50d4bf32fedad2d3d62f3b`, automatically falls back to HuggingFace `torchgeo/eurosat` if Zenodo is unreachable, extracts cleanly, and downloads all 6 official split manifests.
2. **HuggingFace Hub (`torchgeo/eurosat`):**
   - URL: `https://huggingface.co/datasets/torchgeo/eurosat`
   - Hosts unmodified archives (`EuroSAT.zip`, `EuroSATallBands.zip`, `EuroSAT100.zip` tutorial sample) **and** deterministic benchmark split `.txt` files.
   - Other community mirrors on HuggingFace include `timm/eurosat-rgb`, `blanchon/EuroSAT_RGB`, and `GFM-Bench/EuroSAT`.
3. **TorchGeo (PyTorch Geospatial Library):**
   ```python
   from torchgeo.datasets import EuroSAT
   train_ds = EuroSAT(root="data/eurosat", split="train", download=True)
   ```
   Defaults to the 13-band multi-spectral `.tif` version (`EuroSATallBands.zip`).
4. **Torchvision Manual Workaround:**
   If you want to use `torchvision.datasets.EuroSAT(..., download=False)`, extract `EuroSAT_RGB.zip` so the 10 class subfolders live directly under `<root>/eurosat/<ClassName>/`.
5. **TensorFlow Datasets (`tfds`) & Kaggle:**
   `tfds.load("eurosat/rgb")` or `tfds.load("eurosat/all")` (note: TFDS only provides a single `train` split of 27,000 samples, so you must split manually).

---

## 3. Exact 10-Class Distribution (27,000 Total Patches)

Alphabetical order below matches PyTorch `ImageFolder` class indices `0..9`:

| Index | Folder Name | Image Count | Share (%) | Visual / Physical Signature |
| :---: | :--- | :---: | :---: | :--- |
| `0` | `AnnualCrop` | 3,000 | 11.11% | Rectangular seasonal fields, bare soil / harvest cycles |
| `1` | `Forest` | 3,000 | 11.11% | Dense tree canopy, dark textured green |
| `2` | `HerbaceousVegetation` | 3,000 | 11.11% | Natural grasslands, scrubland, unmanaged meadow |
| `3` | `Highway` | 2,500 | 9.26% | Linear multi-lane roads and highway interchanges |
| `4` | `Industrial` | 2,500 | 9.26% | Large bright warehouse roofs, factories, parking lots |
| `5` | `Pasture` | 2,000 | 7.41% | Managed grazing fields, uniform soft green *(smallest class)* |
| `6` | `PermanentCrop` | 2,500 | 9.26% | Orchards, vineyards, olive groves (striped/dotted canopy) |
| `7` | `Residential` | 3,000 | 11.11% | Dense street grids, houses, small yards |
| `8` | `River` | 2,500 | 9.26% | Sinuous freshwater channels crossing land |
| `9` | `SeaLake` | 3,000 | 11.11% | Open sea, bays, and lakes *(uniform dark blue/teal)* |

Note the **mild class imbalance**: `Pasture` has 2,000 samples while `Forest`, `AnnualCrop`, `Residential`, etc. have 3,000. That is why GeoVision tracks **Macro-F1** alongside Top-1 Accuracy!

---

## 4. Sentinel-2 13 Spectral Bands (`EuroSAT_MS`)

While `EuroSAT_RGB` provides bands **B04 (Red), B03 (Green), B02 (Blue)** as standard 8-bit JPEGs, Sentinel-2's Multi-Spectral Instrument actually captures **13 bands** from Ultra-Blue (443 nm) through Short-Wave Infrared (2190 nm):

| Band | Name / Purpose | Central Wavelength | Native Resolution | In `EuroSAT_RGB`? |
| :--- | :--- | :---: | :---: | :---: |
| **B01** | Coastal Aerosol / Ultra Blue | ~443 nm | 60 m | No |
| **B02** | **Blue** | ~490 nm | **10 m** | **Yes (B)** |
| **B03** | **Green** | ~560 nm | **10 m** | **Yes (G)** |
| **B04** | **Red** | ~665 nm | **10 m** | **Yes (R)** |
| **B05** | Vegetation Red Edge 1 | ~705 nm | 20 m | No |
| **B06** | Vegetation Red Edge 2 | ~740 nm | 20 m | No |
| **B07** | Vegetation Red Edge 3 | ~783 nm | 20 m | No |
| **B08** | **Broad Near-Infrared (NIR)** | ~842 nm | **10 m** | No |
| **B8A** | Narrow NIR | ~865 nm | 20 m | No |
| **B09** | Water Vapour | ~945 nm | 60 m | No |
| **B10** | SWIR – Cirrus | ~1375 nm | 60 m | No |
| **B11** | Short-Wave Infrared (SWIR 1) | ~1610 nm | 20 m | No |
| **B12** | Short-Wave Infrared (SWIR 2) | ~2190 nm | 20 m | No |

### Why Multi-Spectral Matters (and How RGB Was Generated)
* **NDVI (`(B08 - B04) / (B08 + B04)`):** Healthy plants strongly absorb Red (`B04`) and strongly reflect NIR (`B08`), separating `Forest`, `AnnualCrop`, `Pasture`, and `HerbaceousVegetation` even when RGB looks similarly green.
* **Official RGB Creation Formula:** Helber et al. converted the raw 13-band GeoTIFF patches to RGB JPEGs using GDAL clipping `0..2750` top-of-atmosphere reflectance into `1..255`:
  ```bash
  gdal_translate --config GDAL_PAM_ENABLED NO -of JPEG -co QUALITY=100 -ot Byte -a_nodata 0 -scale 0 2750 1 255 -b 4 -b 3 -b 2 <input.tif> <output.jpg>
  ```
  (Implemented in Python inside `src/data/preprocessing.py` as `ms_to_rgb()` and `compute_ndvi()`, with `EuroSATMultispectralDataset` in `src/data/dataset.py` ready for 13-band training.)

---

## 5. Dataset Split Methodologies & Spatial Data Leakage

EuroSAT was originally released as a single unsplit pool of 27,000 images. GeoVision supports two standardized TorchGeo split schemes (`https://huggingface.co/datasets/torchgeo/eurosat`):

1. **Standard Official Split (`--split-strategy official`):**
   - Files: `eurosat-train.txt` (16,200), `eurosat-val.txt` (5,400), `eurosat-test.txt` (5,400) — exact **60% / 20% / 20%** split.
   - Enables fair apples-to-apples comparison against published papers (e.g., *In-Domain Representation Learning for Remote Sensing*).
2. **Spatial Split (`--split-strategy spatial`):**
   - Files: `eurosat-spatial-train.txt` (16,200), `eurosat-spatial-val.txt` (5,400), `eurosat-spatial-test.txt` (5,400).
   - **Why Spatial Splits Matter:** Under a random split, two adjacent 640 m × 640 m patches from the same German highway or French valley can land in `train` and `test` respectively. A model can memorize regional soil color or cloud haze rather than true land cover (**spatial autocorrelation leakage**). Spatial splits hold out whole geographic regions for validation/testing, providing a stricter test of out-of-region generalization.

---

## 6. Official References & Citations

- **IEEE JSTARS (2019):** Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. *"EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification."* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.
- **IGARSS (2018):** Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. *"Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification."* IEEE International Geoscience and Remote Sensing Symposium (IGARSS), pp. 204–207, 2018.
- **Official Code Repo:** https://github.com/phelber/EuroSAT
- **Zenodo Record:** https://zenodo.org/records/7711810
