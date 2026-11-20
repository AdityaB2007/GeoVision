"""Pydantic v2 response schemas for the GeoVision FastAPI backend."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    class_name: str = Field(..., description="Canonical EuroSAT class identifier")
    display_name: str = Field(..., description="Human-readable land-cover label")
    probability: float = Field(..., ge=0.0, le=1.0)
    percentage: float = Field(..., ge=0.0, le=100.0)


class PredictionResponse(BaseModel):
    predicted_class: str
    display_name: str
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_percentage: float = Field(..., ge=0.0, le=100.0)
    probabilities: list[ClassProbability]
    model_name: str
    backbone: str
    input_size: int
    inference_time_ms: float


class GradCAMResponse(PredictionResponse):
    target_class: str
    target_display_name: str
    target_layer: str
    colormap: str
    alpha: float = Field(..., ge=0.0, le=1.0)
    heatmap_base64: str
    overlay_base64: str
    gradcam_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    backbone: str | None = None
    device: str
    input_size: int | None = None
    num_classes: int
    test_accuracy: float | None = None


class ClassInfo(BaseModel):
    class_name: str
    display_name: str
    description: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    backbone: str
    num_classes: int
    class_names: list[str]
    input_size: int
    normalization_mean: list[float]
    normalization_std: list[float]
    metrics: dict[str, float | int | None]
    training_config: dict[str, object]
    saved_at: str | None = None
