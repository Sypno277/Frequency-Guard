"""Golden-dataset regression tests (Masterplan §6, §8).

Locks feature vectors + verdicts for a small labeled snapshot so any drift
beyond tolerance fails the build. The canonical lock list lives in
``data/golden/golden_manifest.csv`` (path, label, expected verdict); the
images are the deterministic, procedurally-generated demo set.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from python_services.frequency_guard.config import Settings
from python_services.frequency_guard.features.extractor import extract_features
from python_services.frequency_guard.io.loader import load_image_file
from python_services.frequency_guard.preprocess import preprocess

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = ROOT / "data" / "demo_dataset"
GOLDEN_DIR = ROOT / "data" / "golden"

# Tolerance for feature-vector drift (L2 norm of delta / norm of reference).
FEATURE_TOLERANCE = 0.02


# Read the locked sample list from the manifest (single source of truth).
# Columns: relative_path, label, expected_verdict.
def _read_golden_manifest() -> list[tuple[str, int]]:
    manifest_path = GOLDEN_DIR / "golden_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Golden manifest not found at {manifest_path}")
    rows: list[tuple[str, int]] = []
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append((row["relative_path"], int(row["label"])))
    return rows


GOLDEN_SAMPLES = _read_golden_manifest()


def _feature_vector(path: str) -> np.ndarray:
    # Must match the settings used to train the model (defaults, 89 features).
    settings = Settings()
    loaded = load_image_file(DEMO_DIR / path, settings)
    prep = preprocess(loaded.bgr, settings)
    return extract_features(prep, settings).vector


@pytest.fixture(scope="module")
def reference_vectors() -> dict[str, np.ndarray]:
    """Snapshots that don't exist yet are generated on first run."""
    ref_dir = ROOT / "reports" / "golden_cache"
    ref_dir.mkdir(parents=True, exist_ok=True)
    refs: dict[str, np.ndarray] = {}
    for rel, _label in GOLDEN_SAMPLES:
        key = hashlib.sha256(rel.encode()).hexdigest()
        ref_path = ref_dir / f"{key}.npy"
        if ref_path.exists():
            refs[rel] = np.load(ref_path)
        else:
            vec = _feature_vector(rel)
            np.save(ref_path, vec)
            refs[rel] = vec
    return refs


def test_feature_vectors_stable(reference_vectors: dict[str, np.ndarray]) -> None:
    """Feature vectors must not drift more than tolerance from the snapshot."""
    for rel, ref in reference_vectors.items():
        vec = _feature_vector(rel)
        assert vec.shape == ref.shape, f"{rel}: shape changed {ref.shape} -> {vec.shape}"
        ratio = np.linalg.norm(vec - ref) / (np.linalg.norm(ref) + 1e-9)
        assert (
            ratio < FEATURE_TOLERANCE
        ), f"{rel}: feature drift {ratio:.4f} exceeds tolerance {FEATURE_TOLERANCE}"


@pytest.mark.skipif(
    not (ROOT / "checkpoints" / "ensemble.joblib").exists(), reason="Run scripts/train_demo.py first"
)
def test_verdicts_match_golden(reference_vectors: dict[str, np.ndarray]) -> None:
    """Verdicts on the golden sample must remain correct against the loaded model."""
    from python_services.frequency_guard.api.inference import InferenceService
    from python_services.frequency_guard.config import ensure_dirs

    settings = Settings()
    ensure_dirs(settings)
    service = InferenceService(settings)
    service.ensure_model()

    for rel, label in GOLDEN_SAMPLES:
        vec = _feature_vector(rel).reshape(1, -1)
        raw_fake = float(service._ensemble.predict_proba(vec)[0, 1])
        calib_fake = float(service._calibrator.predict_proba(np.asarray([raw_fake]))[0])
        predicted = 1 if calib_fake >= service._threshold else 0
        assert predicted == label, f"{rel}: predicted {predicted}, expected {label}"
