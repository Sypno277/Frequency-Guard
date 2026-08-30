"""Bootstrap: train the full pipeline on procedurally generated images.

No labeled dataset ships with the repo (Masterplan §4.5 expects an external
manifest), so this script creates one synthetically to prove the entire
chain end-to-end:

- "real" class: smooth 1/f power-law textures + sensor-like noise
  (natural spectral statistics per Masterplan §4.1).
- "fake" class: flat/boosted high-frequency spectra with periodic
  upsampling peaks and synthetic noise (diffusion/GAN signatures).

The images are written under ``data/demo_dataset/`` with a manifest CSV,
then routed through the REAL pipeline: preprocess → extractors → stratified
CV ensemble → isotonic calibration → attribution → evaluation → reports.
Artifacts land in ``checkpoints/``; metrics in ``reports/``.

Run:
    python -m scripts.train_demo

The API labels its model "demo" until a real manifest is trained via
``train_from_manifest``.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from python_services.frequency_guard.config import Settings, ensure_dirs, load_settings
from python_services.frequency_guard.evaluation.evaluate import (
    current_peak_rss_mb,
    evaluate_model,
    persist_report,
)
from python_services.frequency_guard.features.extractor import cache_key, extract_features, global_cache
from python_services.frequency_guard.io.loader import load_image_file
from python_services.frequency_guard.logging import configure_logging, get_logger
from python_services.frequency_guard.models.attribution import FamilyAttributor
from python_services.frequency_guard.preprocess import preprocess
from python_services.frequency_guard.training.train_pipeline import train_from_manifest

log = get_logger(__name__)

N_PER_CLASS = 60  # small but enough for a stable demo model


def _synthetic_real(seed: int, size: int = 256) -> np.ndarray:
    """Natural-like image: 1/f spectrum + smooth structure + sensor noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    fx = np.fft.fftfreq(size)[:, None]
    fy = np.fft.fftfreq(size)[None, :]
    freq = np.sqrt(fx**2 + fy**2)
    freq[0, 0] = 1.0

    # A few random smooth blobs (low-frequency structure).
    amplitude = 1.0 / np.power(freq, 1.05)  # pink-ish energy → slope ≈ -2.1 in log-log of magnitude^2
    phase = rng.uniform(0, 2 * np.pi, (size, size))
    base = np.real(np.fft.ifft2(amplitude * np.exp(1j * phase)))

    # Gentle spatial gradients for texture realism.
    gradient = 0.15 * np.sin(xx / (10 + seed % 17)) * np.cos(yy / (12 + seed % 23))

    img = base / (np.abs(base).max() + 1e-9)
    img = img + 0.3 * gradient
    # Sensor-like white noise at realistic amplitude.
    img += rng.normal(0, 0.02, (size, size))
    img = np.clip(img * 0.5 + 0.5, 0, 1)
    return (img * 255).astype(np.uint8)


def _synthetic_fake(seed: int, size: int = 256) -> np.ndarray:
    """Generator-like image: flatter spectrum + periodic upsampling peaks."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    fx = np.fft.fftfreq(size)[:, None]
    fy = np.fft.fftfreq(size)[None, :]
    freq = np.sqrt(fx**2 + fy**2)
    freq[0, 0] = 1.0

    # Much weaker 1/f decay → unnaturally flat mid/high bands.
    amplitude = 1.0 / np.power(freq, 0.45)
    phase = rng.uniform(0, 2 * np.pi, (size, size))
    base = np.real(np.fft.ifft2(amplitude * np.exp(1j * phase)))

    # Checkerboard upsampling fingerprint: strong energy at ω≈π/2 harmonics.
    checker = 0.35 * np.sin(xx * np.pi / 2) * np.sin(yy * np.pi / 2)

    img = base / (np.abs(base).max() + 1e-9)
    img = img + checker
    # Over-smoothed synthetic noise floor.
    img += rng.normal(0, 0.004, (size, size))
    img = np.clip(img * 0.5 + 0.5, 0, 1)
    return (img * 255).astype(np.uint8)


# --- degradation augmentations (Masterplan §4.5) ------------------------
# Demo-classifier realism requires surviving re-capture-like transformations.
# We emit augmented variants so the model learns resilient frequency cues
# rather than over-fitting the pristine checkerboard signature.


def _jpeg_degrade(img: np.ndarray, quality: int = 60) -> np.ndarray:
    """Re-encode at a moderate JPEG quality (simulated re-compression)."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert decoded is not None
    return decoded


def _resize_cycle(img: np.ndarray) -> np.ndarray:
    """Upscale then downscale — simulates re-save after a size change."""
    up = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return cv2.resize(up, None, fx=2 / 3, fy=2 / 3, interpolation=cv2.INTER_AREA)


def _center_crop(img: np.ndarray) -> np.ndarray:
    """Tight 75% center crop — simulates re-capture framing loss."""
    h, w = img.shape[:2]
    return img[h // 8 : -h // 8 or None, w // 8 : -w // 8 or None]


def _degrade_variants(img: np.ndarray) -> list[np.ndarray]:
    """Return [pristine, jpeg60, resize_cycle, center_crop] per sample."""
    return [img, _jpeg_degrade(img), _resize_cycle(img), _center_crop(img)]


def build_demo_dataset(root: Path, n_per_class: int = N_PER_CLASS) -> Path:
    """Write demo images + manifest.csv; returns the manifest path."""
    real_dir = root / "real"
    fake_dir = root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, int]] = []
    suffixes = ["", "_jpeg60", "_resize", "_crop"]
    for i in range(n_per_class):
        # Deterministic per-seed variants keep the set reproducible.
        real_variants = _degrade_variants(_synthetic_real(seed=1000 + i))
        fake_variants = _degrade_variants(_synthetic_fake(seed=2000 + i))

        for suffix, real_img, fake_img in zip(suffixes, real_variants, fake_variants, strict=True):
            real_path = real_dir / f"real_{i:04d}{suffix}.png"
            fake_path = fake_dir / f"fake_{i:04d}{suffix}.png"
            cv2.imwrite(str(real_path), real_img)
            cv2.imwrite(str(fake_path), fake_img)
            rows.append((str(real_path), 0))
            rows.append((str(fake_path), 1))

    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "label"])
        writer.writerows(rows)

    log.info("demo dataset built", extra={"fields": {"manifest": str(manifest_path), "n": len(rows)}})
    return manifest_path


def main() -> None:
    """Full bootstrap: dataset → train → calibrate → attribute → evaluate."""
    settings: Settings = load_settings()
    configure_logging(level=settings.log_level)
    ensure_dirs(settings)

    data_root = settings.data_dir / "demo_dataset"
    if not (data_root / "manifest.csv").exists():
        build_demo_dataset(data_root)
    manifest = data_root / "manifest.csv"

    result = train_from_manifest(manifest, settings.model_dir, settings=settings)

    # Persist attributor (heuristic mode until family labels exist).
    attributor = FamilyAttributor()
    attributor.save(settings.model_dir / "attributor.joblib")

    # ---- held-out-style self-evaluation on the training set -------------
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    with manifest.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            loaded = load_image_file(row["image_path"], settings)
            prep = preprocess(loaded.bgr, settings)
            key = cache_key(row["image_path"], prep, settings)
            bundle = global_cache.get_or_compute(key, lambda p=prep: extract_features(p, settings))
            X_rows.append(bundle.vector)
            y_rows.append(int(row["label"]))

    X = np.stack(X_rows)
    y = np.asarray(y_rows)
    raw_prob = result.ensemble.predict_proba(X)[:, 1]
    calib_prob = result.calibrator.predict_proba(raw_prob)

    report = evaluate_model(
        y_true=y,
        raw_fake_prob=raw_prob,
        calibrated_fake_prob=calib_prob,
        settings=settings,
        predict_fn=lambda M: result.ensemble.predict(M),
        latency_probe=X[:8],
    )
    persist_report(report, settings)

    # Performance panel artifact consumed by GET /api/v1/model/performance.
    perf_payload = report.to_dict()
    (settings.model_dir / "performance.json").write_text(
        __import__("json").dumps(perf_payload, indent=2), encoding="utf-8"
    )

    print("\n=== Frequency Guard v2 demo training complete ===")
    print(f"accuracy : {report.accuracy:.4f}")
    print(f"f1       : {report.f1:.4f}")
    print(f"roc_auc  : {report.roc_auc:.4f}")
    print(f"ece      : {report.ece:.4f}")
    print(f"fpr@thr  : {report.fpr_at_threshold:.4f} (threshold={report.threshold})")
    print(f"peak RSS : {current_peak_rss_mb():.1f} MB")
    print(f"artifacts: {settings.model_dir}")


if __name__ == "__main__":
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    main()
