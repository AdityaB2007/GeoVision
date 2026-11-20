"""FastAPI application serving GeoVision ML inference and static frontend."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.inference import GeoVisionInferenceEngine
from backend.schemas import (
    ClassInfo,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.config import (
    CLASS_DESCRIPTIONS,
    CLASS_DISPLAY_NAMES,
    EUROSAT_CLASSES,
    FRONTEND_DIR,
    MAX_UPLOAD_BYTES,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the PyTorch checkpoint once on startup and release on shutdown."""
    app.state.engine = GeoVisionInferenceEngine()
    yield
    app.state.engine = None


app = FastAPI(
    title="GeoVision Satellite Land-Cover API",
    description=(
        "FastAPI inference service for EuroSAT Sentinel-2 satellite imagery "
        "classification using PyTorch deep learning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_engine() -> GeoVisionInferenceEngine:
    engine: GeoVisionInferenceEngine | None = getattr(app.state, "engine", None)
    if engine is None:
        engine = GeoVisionInferenceEngine()
        app.state.engine = engine
    return engine


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Report server liveness and loaded model metadata."""
    engine = _get_engine()
    test_acc = engine.metrics.get("test_accuracy")
    return HealthResponse(
        status="ok" if engine.model_loaded else "degraded",
        model_loaded=engine.model_loaded,
        model_name=engine.run_name,
        backbone=engine.backbone,
        device=str(engine.device),
        input_size=engine.input_size,
        num_classes=len(engine.class_names),
        test_accuracy=float(test_acc) if test_acc is not None else None,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info() -> ModelInfoResponse:
    """Return architecture, normalization constants, and evaluation metrics."""
    engine = _get_engine()
    if not engine.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="No trained model checkpoint loaded. Train a model with --set-best first.",
        )
    return ModelInfoResponse(
        model_name=engine.run_name or "GeoVision",
        model_type=engine.model_type or "unknown",
        backbone=engine.backbone or "unknown",
        num_classes=len(engine.class_names),
        class_names=engine.class_names,
        input_size=int(engine.input_size or 64),
        normalization_mean=engine.mean,
        normalization_std=engine.std,
        metrics=engine.metrics,
        training_config=engine.training_config,
        saved_at=engine.saved_at,
    )


@app.get("/classes", response_model=list[ClassInfo], tags=["System"])
async def list_classes() -> list[ClassInfo]:
    """List the 10 EuroSAT Sentinel-2 land-cover classes and descriptions."""
    return [
        ClassInfo(
            class_name=cls_name,
            display_name=CLASS_DISPLAY_NAMES.get(cls_name, cls_name),
            description=CLASS_DESCRIPTIONS.get(cls_name, ""),
        )
        for cls_name in EUROSAT_CLASSES
    ]


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Classify an uploaded RGB satellite patch into 10 EuroSAT classes."""
    engine = _get_engine()
    if not engine.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint is not loaded on the server.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File exceeds maximum allowed size of 10 MB.",
        )

    try:
        result = engine.predict_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(**result)


# Mount static frontend last so API routes (/health, /predict, /classes,
# /model-info, /docs) always take precedence over static file lookup.
if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )
