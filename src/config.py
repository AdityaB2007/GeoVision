"""Central configuration, domain constants, and dataset metadata for GeoVision.

Keeping dataset paths, class mappings, spectral band definitions, and
normalization constants in a single module ensures that training, evaluation,
and FastAPI inference never disagree on class order or preprocessing parameters
(preventing train/serve skew).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem paths (anchored relative to repository root)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
SPLITS_DIR: Path = DATA_PROCESSED_DIR / "splits"
MODELS_DIR: Path = PROJECT_ROOT / "models"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.pt"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
TRAINING_OUTPUTS_DIR: Path = OUTPUTS_DIR / "training"
EVALUATION_OUTPUTS_DIR: Path = OUTPUTS_DIR / "evaluation"
FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"

# ---------------------------------------------------------------------------
# EuroSAT 10-class taxonomy (alphabetical order matches ImageFolder & splits)
# ---------------------------------------------------------------------------
EUROSAT_CLASSES: tuple[str, ...] = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)

CLASS_TO_IDX: dict[str, int] = {
    name: idx for idx, name in enumerate(EUROSAT_CLASSES)
}
IDX_TO_CLASS: dict[int, str] = {
    idx: name for idx, name in enumerate(EUROSAT_CLASSES)
}

CLASS_DISPLAY_NAMES: dict[str, str] = {
    "AnnualCrop": "Annual Crop",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation",
    "Highway": "Highway",
    "Industrial": "Industrial",
    "Pasture": "Pasture",
    "PermanentCrop": "Permanent Crop",
    "Residential": "Residential",
    "River": "River",
    "SeaLake": "Sea & Lake",
}

CLASS_DESCRIPTIONS: dict[str, str] = {
    "AnnualCrop": (
        "Seasonal agricultural fields (e.g., wheat, corn, barley) replanted "
        "yearly, showing sharp rectangular parcel boundaries and bare-soil/crop cycles."
    ),
    "Forest": (
        "Dense tree canopy with deep green texture, low albedo, and irregular "
        "natural boundaries across deciduous or coniferous woodland."
    ),
    "HerbaceousVegetation": (
        "Natural grasslands, scrub, and unmanaged meadow cover lacking the "
        "geometric ploughing rows of active croplands."
    ),
    "Highway": (
        "Linear paved multi-lane transportation corridors cutting across "
        "surrounding terrain, often flanked by embankments or interchanges."
    ),
    "Industrial": (
        "Large commercial warehouses, factories, logistics hubs, and parking "
        "structures characterized by bright flat roofs and high impervious cover."
    ),
    "Pasture": (
        "Managed grazing meadows and open grass fields with relatively uniform "
        "green reflectance and soft agricultural borders."
    ),
    "PermanentCrop": (
        "Perennial plantations such as orchards, vineyards, and olive groves "
        "displaying structured dotted or striped canopy rows."
    ),
    "Residential": (
        "Dense urban and suburban neighborhoods composed of fine-grained street "
        "grids, houses, gardens, and small rooftops."
    ),
    "River": (
        "Winding natural or canalized freshwater channels cutting sinuous paths "
        "through surrounding vegetated or urban landscapes."
    ),
    "SeaLake": (
        "Open marine waters, coastal bays, or large inland lakes with spatially "
        "uniform dark blue-teal reflectance."
    ),
}

# ---------------------------------------------------------------------------
# Sentinel-2 Multi-Spectral Instrument (MSI) 13-band metadata
# ---------------------------------------------------------------------------
SENTINEL2_BANDS: dict[str, dict[str, object]] = {
    "B01": {"index": 0, "name": "Coastal Aerosol", "wavelength_nm": 443, "resolution_m": 60},
    "B02": {"index": 1, "name": "Blue", "wavelength_nm": 490, "resolution_m": 10},
    "B03": {"index": 2, "name": "Green", "wavelength_nm": 560, "resolution_m": 10},
    "B04": {"index": 3, "name": "Red", "wavelength_nm": 665, "resolution_m": 10},
    "B05": {"index": 4, "name": "Vegetation Red Edge 1", "wavelength_nm": 705, "resolution_m": 20},
    "B06": {"index": 5, "name": "Vegetation Red Edge 2", "wavelength_nm": 740, "resolution_m": 20},
    "B07": {"index": 6, "name": "Vegetation Red Edge 3", "wavelength_nm": 783, "resolution_m": 20},
    "B08": {"index": 7, "name": "NIR (Broad)", "wavelength_nm": 842, "resolution_m": 10},
    "B8A": {"index": 8, "name": "NIR (Narrow)", "wavelength_nm": 865, "resolution_m": 20},
    "B09": {"index": 9, "name": "Water Vapour", "wavelength_nm": 945, "resolution_m": 60},
    "B10": {"index": 10, "name": "SWIR - Cirrus", "wavelength_nm": 1375, "resolution_m": 60},
    "B11": {"index": 11, "name": "SWIR 1", "wavelength_nm": 1610, "resolution_m": 20},
    "B12": {"index": 12, "name": "SWIR 2", "wavelength_nm": 2190, "resolution_m": 20},
}

# ---------------------------------------------------------------------------
# Normalization & Dataset Download Sources
# ---------------------------------------------------------------------------
# ImageNet RGB channel statistics are required when fine-tuning backbones
# pre-trained on ImageNet (e.g., ResNet-18, EfficientNet-B0) because the first
# conv layer's weights expect inputs centered and scaled identically to pre-training.
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

EUROSAT_NATIVE_SIZE: int = 64
EUROSAT_RGB_ZENODO_URL: str = (
    "https://zenodo.org/api/records/7711810/files/EuroSAT_RGB.zip/content"
)
EUROSAT_RGB_MD5: str = "f46e308c4d50d4bf32fedad2d3d62f3b"
EUROSAT_RGB_HF_FALLBACK_URL: str = (
    "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip"
)

TORCHGEO_SPLITS_BASE_URL: str = (
    "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main"
)
OFFICIAL_SPLIT_FILES: tuple[str, ...] = (
    "eurosat-train.txt",
    "eurosat-val.txt",
    "eurosat-test.txt",
    "eurosat-spatial-train.txt",
    "eurosat-spatial-val.txt",
    "eurosat-spatial-test.txt",
)

MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB limit matching frontend validation
