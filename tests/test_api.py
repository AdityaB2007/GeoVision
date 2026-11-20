"""Integration tests for the FastAPI inference backend and static UI."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app
from src.config import BEST_MODEL_PATH, DATA_RAW_DIR, EUROSAT_CLASSES


def _make_png_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(34, 139, 34)).save(buf, format="PNG")
    return buf.getvalue()


def test_root_serves_frontend_html() -> None:
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "GeoVision" in resp.text


def test_health_and_classes_endpoints() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["num_classes"] == 10
        assert body["model_loaded"] is BEST_MODEL_PATH.exists()

        classes_resp = client.get("/classes")
        assert classes_resp.status_code == 200
        cls_names = [item["class_name"] for item in classes_resp.json()]
        assert cls_names == list(EUROSAT_CLASSES)


def test_model_info_and_predict_valid_and_invalid() -> None:
    assert BEST_MODEL_PATH.exists(), "Expected trained models/best_model.pt before API tests"

    with TestClient(app) as client:
        info = client.get("/model-info")
        assert info.status_code == 200
        info_json = info.json()
        assert info_json["class_names"] == list(EUROSAT_CLASSES)

        # Test prediction on a real EuroSAT image if present, plus a synthetic PNG
        sample_candidates = list((DATA_RAW_DIR / "eurosat_rgb").rglob("*.jpg"))
        payload_bytes = (
            sample_candidates[0].read_bytes()
            if sample_candidates
            else _make_png_bytes((64, 64))
        )

        pred = client.post(
            "/predict",
            files={"file": ("patch.jpg", payload_bytes, "image/jpeg")},
        )
        assert pred.status_code == 200
        p = pred.json()
        assert p["predicted_class"] in EUROSAT_CLASSES
        assert len(p["probabilities"]) == 10
        prob_sum = sum(item["probability"] for item in p["probabilities"])
        assert abs(prob_sum - 1.0) < 1e-3
        assert abs(p["confidence"] - p["probabilities"][0]["probability"]) < 1e-5

        # Invalid image bytes -> HTTP 400
        bad = client.post(
            "/predict",
            files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
        )
        assert bad.status_code == 400

        # Oversized (> 10 MB) payload -> HTTP 413
        huge = client.post(
            "/predict",
            files={"file": ("huge.bin", b"x" * (10 * 1024 * 1024 + 1), "image/png")},
        )
        assert huge.status_code == 413
