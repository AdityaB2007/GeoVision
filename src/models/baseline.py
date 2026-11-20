"""Classical computer-vision baseline using handcrafted spectral & edge features.

Why start with a baseline?
    Before training deep neural networks, disciplined ML engineering builds
    a fast, interpretable classical model to establish a quantitative floor.
    Comparing a classical model against a CNN directly answers: "How much does
    spatial representation learning actually buy us over color histograms?"

Handcrafted 122-dimensional feature vector per patch:
    - 48 features: Per-channel RGB 16-bin normalized histograms (global color)
    - 48 features: Per-channel HSV 16-bin normalized histograms (separates hue
      from illumination, e.g. green canopy vs dark blue water)
    - 24 features: 2x2 spatial quadrant RGB means and standard deviations
      (coarse spatial layout)
    -  2 features: Mean and std of finite-difference gradient magnitude
      (texture / urban edge density: Residential vs smooth SeaLake)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def _channel_histograms(arr: np.ndarray, bins: int = 16) -> np.ndarray:
    """Compute normalized per-channel histograms over [0, 1] values."""
    hists: list[np.ndarray] = []
    for c in range(arr.shape[2]):
        hist, _ = np.histogram(arr[:, :, c], bins=bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        hists.append(hist)
    return np.concatenate(hists, axis=0)


def extract_handcrafted_features(image_path: Path) -> np.ndarray:
    """Extract a 122-element feature vector from a single satellite RGB patch."""
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        hsv_img = rgb_img.convert("HSV")

    rgb = np.asarray(rgb_img, dtype=np.float32) / 255.0
    hsv = np.asarray(hsv_img, dtype=np.float32) / 255.0

    # 1 & 2. Global RGB and HSV color distributions (48 + 48 = 96 dims)
    rgb_hist = _channel_histograms(rgb, bins=16)
    hsv_hist = _channel_histograms(hsv, bins=16)

    # 3. 2x2 quadrant RGB mean & std (4 quadrants * 6 stats = 24 dims)
    h, w, _ = rgb.shape
    mid_h, mid_w = max(1, h // 2), max(1, w // 2)
    quadrants = (
        rgb[:mid_h, :mid_w],
        rgb[:mid_h, mid_w:],
        rgb[mid_h:, :mid_w],
        rgb[mid_h:, mid_w:],
    )
    quad_stats: list[np.ndarray] = []
    for quad in quadrants:
        quad_stats.append(quad.mean(axis=(0, 1)))
        quad_stats.append(quad.std(axis=(0, 1)))
    quad_vec = np.concatenate(quad_stats, axis=0).astype(np.float32)

    # 4. Edge/texture density from luminance gradient magnitude (2 dims)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad_mag = np.sqrt(dx * dx + dy * dy)
    edge_stats = np.array(
        [float(grad_mag.mean()), float(grad_mag.std())], dtype=np.float32
    )

    return np.concatenate([rgb_hist, hsv_hist, quad_vec, edge_stats], axis=0)


def extract_feature_matrix(
    image_paths: Sequence[Path], desc: str = "Extracting features"
) -> np.ndarray:
    """Extract feature matrix of shape (N, 122) for a sequence of image paths."""
    rows = [
        extract_handcrafted_features(path)
        for path in tqdm(image_paths, desc=desc, leave=False)
    ]
    return np.vstack(rows).astype(np.float32)


class BaselineClassifier:
    """Classical ML classifier wrapping StandardScaler + LogisticRegression / RF.

    Standardizing features (zero mean, unit variance) is critical for
    Logistic Regression with L2 regularization so features with larger raw
    scales do not dominate the penalty or slow down L-BFGS convergence.
    """

    def __init__(
        self,
        algorithm: Literal["logreg", "rf"] = "logreg",
        random_state: int = 42,
    ) -> None:
        self.algorithm = algorithm
        self.random_state = random_state
        if algorithm == "logreg":
            estimator = LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=random_state,
            )
            self.pipeline = Pipeline(
                [("scaler", StandardScaler()), ("clf", estimator)]
            )
        elif algorithm == "rf":
            estimator = RandomForestClassifier(
                n_estimators=200,
                n_jobs=-1,
                random_state=random_state,
            )
            self.pipeline = Pipeline([("clf", estimator)])
        else:
            raise ValueError(f"Unsupported baseline algorithm: {algorithm}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineClassifier":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def save(self, path: Path, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": self.algorithm,
            "random_state": self.random_state,
            "pipeline": self.pipeline,
            "metadata": metadata or {},
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> tuple["BaselineClassifier", dict]:
        payload = joblib.load(path)
        instance = cls(
            algorithm=payload["algorithm"],
            random_state=payload.get("random_state", 42),
        )
        instance.pipeline = payload["pipeline"]
        return instance, payload.get("metadata", {})
