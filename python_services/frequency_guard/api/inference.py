"""Core inference service (Masterplan §3.1 data flow, §5.1).

Single entry point used by both ``/api/v1/analyze`` and batch jobs:

    bytes → load/validate → preprocess → feature extraction (LRU-cached)
          → ensemble predict → calibrate → threshold gate
          → family attribution → explainability heatmap
          → AnalyzeResponse payload

Also owns model lifecycle: lazy-loading persisted artifacts, reporting
model metadata for ``/api/v1/model`` and the performance panel.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import Settings, ensure_dirs
from ..explainability.heatmap import compute_explainability
from ..explainability.semantic_arbiter import run_semantic_arbiter
from ..features.augmentation import tta_variants
from ..features.extractor import FeatureBundle, cache_key, extract_features, global_cache
from ..io.loader import load_image_bytes
from ..io.provenance import extract_provenance
from ..logging import get_logger
from ..models.attribution import FamilyAttributor
from ..models.calibration import ProbabilityCalibrator
from ..models.classifier import EnsembleClassifier, StackedMetaClassifier
from ..models.contributions import ContributionResult, compute_contributions
from ..models.explanation import synthesize_explanation
from ..models.verdict import clamp_threshold, resolve_verdict
from ..preprocess import PreprocessedImage, preprocess
from .schemas import (
    AnalyzeResponse,
    AzimuthalPoint,
    EvidenceBreakdown,
    ExplainabilityPayload,
    ExplanationPayload,
    FamilyProbability,
    FeatureContribution,
    FeatureReadings,
    ProvenancePayload,
    SemanticEvidence,
    SemanticFlagModel,
    SpectrumBin,
    WaveletBand,
)

log = get_logger(__name__)

MODEL_VERSION = "2.0.0"


class ModelNotReadyError(RuntimeError):
    """Raised when no trained model artifact exists yet."""


@dataclass(frozen=True)
class InferenceOutcome:
    """Raw result before schema serialization."""

    response: AnalyzeResponse
    latency_ms: float


class InferenceService:
    """Loads model artifacts once and serves analyze requests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ensemble: EnsembleClassifier | None = None
        self._calibrator: ProbabilityCalibrator | None = None
        self._meta: StackedMetaClassifier | None = None
        self._attributor: FamilyAttributor | None = None
        self._threshold: float = 0.5
        self._training_mode: str = "demo"
        self._trained_at: str | None = None
        self._metrics: dict[str, float] | None = None
        self._last_bundle: Any | None = None

    # --- model lifecycle -------------------------------------------------

    @property
    def model_ready(self) -> bool:
        """True when ensemble + calibrator artifacts are loaded."""
        return self._ensemble is not None and self._calibrator is not None

    def ensure_model(self) -> None:
        """Load persisted artifacts if present; raise otherwise."""
        if self.model_ready:
            return
        ensure_dirs(self.settings)
        ensemble_path = self.settings.model_dir / "ensemble.joblib"
        calibrator_path = self.settings.model_dir / "calibrator.joblib"
        attributor_path = self.settings.model_dir / "attributor.joblib"
        report_path = self.settings.model_dir / "report.json"

        if not ensemble_path.exists() or not calibrator_path.exists():
            raise ModelNotReadyError(
                "No trained model found. Run scripts/train_demo.py or supply a manifest first."
            )

        self._ensemble = EnsembleClassifier.load(ensemble_path)
        self._calibrator = ProbabilityCalibrator.load(calibrator_path)
        meta_path = self.settings.model_dir / "meta_learner.joblib"
        self._meta = StackedMetaClassifier.load(meta_path) if meta_path.exists() else None
        self._attributor = (
            FamilyAttributor.load(attributor_path) if attributor_path.exists() else FamilyAttributor()
        )

        # Load run metadata persisted at train/eval time (threshold tuned on
        # validation, held-out metrics, timestamp). performance.json is the
        # authoritative artifact written by evaluate.py; report.json carries
        # the threshold written by the training pipeline when available.
        perf_payload: dict[str, Any] | None = None
        perf_path = self.settings.model_dir / "performance.json"
        if perf_path.exists():
            try:
                perf_payload = json.loads(perf_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                perf_payload = None

        if perf_payload is not None:
            self._metrics = {
                k: float(perf_payload[k])
                for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "ece")
                if isinstance(perf_payload.get(k), int | float)
            }
            if isinstance(perf_payload.get("generated_at"), str):
                self._trained_at = perf_payload["generated_at"]

        self._threshold = self._resolve_threshold(report_path, perf_payload)
        log.info("model artifacts loaded", extra={"fields": {"dir": str(self.settings.model_dir)}})

    @staticmethod
    def _resolve_threshold(report_path: Path, perf_payload: dict[str, Any] | None) -> float:
        """Read the tuned decision threshold (perf report → train report → 0.5).

        The result is always sanitized through :func:`clamp_threshold`, which
        rejects any persisted threshold >= 0.5. A detector whose "fake" gate
        sits at 1.0 (or even 0.9) can never produce ``is_ai=True`` for a real
        input — the reported "AI image → high-confidence Authentic" bug.
        """
        raw: float | None = None
        if perf_payload is not None:
            candidate = perf_payload.get("threshold")
            if isinstance(candidate, int | float):
                raw = float(candidate)
        if raw is None:
            try:
                report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
                candidate = report.get("threshold")
                if isinstance(candidate, int | float):
                    raw = float(candidate)
            except Exception:
                raw = None
        return clamp_threshold(raw)

    def set_training_metadata(
        self,
        training_mode: str,
        trained_at: str | None,
        metrics: dict[str, float] | None,
        threshold: float,
    ) -> None:
        """Attach run metadata after training completes in-process."""
        self._training_mode = training_mode
        self._trained_at = trained_at
        self._metrics = metrics
        self._threshold = threshold

    def reload(self) -> None:
        """Drop cached artifacts so the next request re-reads from disk."""
        self._ensemble = None
        self._calibrator = None
        self._meta = None
        self._attributor = None
        self.ensure_model()

    # --- analysis ----------------------------------------------------------

    def _raw_fake_probability(self, vector: np.ndarray) -> tuple[float, np.ndarray]:
        """Stacked meta-learner (E2.1) when fitted, else probability-average.

        Returns (fake_probability, member_fake_probs).
        """
        assert self._ensemble is not None
        member_probs = self._ensemble.member_probabilities(vector)[:, :, 1]
        if self._meta is not None and self._meta.fitted:
            stacked = self._ensemble.member_probabilities(vector)
            return float(self._meta.predict_fake(stacked)[0]), member_probs[0]
        return float(self._ensemble.predict_proba(vector)[0, 1]), member_probs[0]

    def _calibrated_probability(
        self,
        prep_variants: list[Any],
        use_tta: bool,
        cache_salt: str,
    ) -> tuple[float, np.ndarray, bool]:
        """Calibrated fake probability over one image or TTA variants (E3.2).

        Args:
            prep_variants: preprocessed variants (1, or 5 for TTA).
            use_tta: whether to average over all variants.
            cache_salt: image-unique string (e.g. sha256) so the feature
                cache keys cannot collide across different images that share
                the same preprocessed size (E3.2 hardening). Without it, the
                batch path's second file would reuse the first's features.

        Returns (calibrated_fake, member_probs_of_primary, tta_used).
        """
        assert self._calibrator is not None
        raws: list[float] = []
        member_primary = np.zeros(4, dtype=np.float64)
        used_tta = False
        for i, prep in enumerate(prep_variants if use_tta else prep_variants[:1]):
            key = cache_key(f"{cache_salt}::tta::{i}", prep, self.settings)

            def _compute(p: PreprocessedImage = prep, s: Settings = self.settings) -> FeatureBundle:
                return extract_features(p, s)

            bundle = global_cache.get_or_compute(key, _compute)
            raw, member = self._raw_fake_probability(bundle.vector.reshape(1, -1))
            raws.append(raw)
            if i == 0:
                member_primary = member
                self._last_bundle = bundle
        mean_raw = float(np.mean(raws))
        calibrated = float(np.clip(self._calibrator.predict_proba(np.asarray([mean_raw]))[0], 0.0, 1.0))
        return calibrated, member_primary, used_tta or use_tta

    def _evidence_breakdown(self, bundle: Any, member_probs: np.ndarray) -> EvidenceBreakdown:
        """Per-feature contributions + agreement meter (E-Masterplan E4).

        Best-effort: returns None-schema (empty contributions) if the
        ensemble lacks feature names, so analysis never fails on explainability.
        """
        assert self._ensemble is not None
        try:
            names = self._ensemble.feature_names or tuple(f"f{i}" for i in range(bundle.vector.size))
            result = compute_contributions(self._ensemble, names, bundle.vector.reshape(1, -1))
            return _evidence_from_result(result)
        except Exception:  # noqa: BLE001 - explainability must never break analysis
            log.warning("evidence breakdown failed", exc_info=True)
            return EvidenceBreakdown(contributions=[], member_probabilities={}, agreement=1.0)

    def analyze_bytes(
        self,
        buffer: bytes,
        source: str,
        use_tta: bool = False,
        threshold: float | None = None,
    ) -> InferenceOutcome:
        """Run the full pipeline over raw image bytes.

        Args:
            buffer: raw image bytes.
            source: display name for tracing.
            use_tta: high-scrutiny mode — average predictions over 5
                deterministic variants (E-Masterplan E3.2). ~5x extraction
                cost; opt-in only.

        Raises:
            LoadError: undecodable/unsupported payload (maps to 4xx).
            ModelNotReadyError: no trained artifacts.
        """
        start = time.perf_counter()
        self.ensure_model()

        loaded = load_image_bytes(buffer, source, self.settings)
        sha256 = hashlib.sha256(buffer).hexdigest()
        primary_prep = preprocess(loaded.bgr, self.settings)

        preps: list[Any] = [primary_prep]
        if use_tta:
            for i, variant in enumerate(tta_variants(loaded.bgr, self.settings)):
                if i == 0:
                    continue  # variant 0 is the untouched image (= primary_prep)
                preps.append(preprocess(variant, self.settings))

        calibrated_fake, member_probs, tta_used = self._calibrated_probability(preps, use_tta, sha256)
        bundle = self._last_bundle
        assert bundle is not None

        # Single source of truth for verdict semantics (D0.1/D2.2). This
        # guarantees a confident "ai"/"authentic" label is never emitted at
        # confidence < 50, and that a persisted threshold >= 0.5 can never
        # make the AI gate unreachable. An optional per-request threshold
        # override lets a tuned operating point be applied at serve time
        # without retraining (WI-3).
        effective_threshold = clamp_threshold(threshold) if threshold is not None else self._threshold
        verdict = resolve_verdict(calibrated_fake, effective_threshold)
        is_ai = verdict.is_ai
        confidence = verdict.confidence
        verdict_state = verdict.state
        # Ensemble-agreement gate (D3.3): low agreement forces "uncertain"
        # even at high confidence. Hardening: a low-agreement call must NOT
        # emit a hard AI positive unless the calibrated probability is
        # near-certain (>= 0.9). This removes the leak where a real photo
        # with agreement 0.03-0.08 still returned is_ai=True (benchmark:
        # 14/20 real photos flagged AI at 70% FPR). When probability >= 0.9,
        # keep the confident "ai" label rather than downgrading to "uncertain".
        evidence = self._evidence_breakdown(bundle, member_probs)
        if evidence.agreement < 0.5 and calibrated_fake < 0.9:
            verdict_state = "uncertain"
            is_ai = False
            log.info(
                "agreement gate suppressed hard AI",
                extra={"fields": {"agreement": evidence.agreement, "fake_prob": calibrated_fake}},
            )

        assert self._attributor is not None
        attribution = self._attributor.attribute(calibrated_fake)

        explanation = compute_explainability(primary_prep.y_channel, self.settings.tiles_per_side)

        # --- FrequencyGuard Phase 1: provenance ---------------------------
        provenance = extract_provenance(buffer)
        # Phase 1 bifurcation: a C2PA manifest short-circuits to a verdict.
        # We surface it as a signal and let the semantic/explanation layers
        # reflect it, but never skip the ML pipeline (attestation != proof).
        provenance_payload = ProvenancePayload(
            has_c2pa=provenance.has_c2pa,
            has_exif=provenance.has_exif,
            has_iptc=provenance.has_iptc,
            exif_tags=provenance.exif_tags,
            provenance_void=provenance.provenance_void,
            void_penalty=round(provenance.void_penalty, 4),
            bifurcated=provenance.bifurcated,
            summary=provenance.summary,
        )

        # --- FrequencyGuard Phase 3: semantic arbitration -----------------
        # Only runs when confidence is in the 30-70% "uncanny" band.
        # primary_prep.rgb is float32 0..1 in RGB order; the arbiter's OpenCV
        # helpers expect uint8 BGR, so convert here.
        bgr_for_semantic = cv2.cvtColor(
            (np.clip(primary_prep.rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        semantic = run_semantic_arbiter(bgr_for_semantic, calibrated_fake)
        semantic_payload = SemanticEvidence(
            triggered=semantic.triggered,
            suspicious=semantic.suspicious,
            flags=[
                SemanticFlagModel(
                    node=f.node,
                    label=f.label,
                    score=round(f.score, 4),
                    bbox=list(f.bbox) if f.bbox else None,
                )
                for f in semantic.flags
            ],
            summary=semantic.summary,
        )

        # --- FrequencyGuard Phase 4: linguistic synthesis -----------------
        top_family, top_family_prob = attribution.top()
        top_contribs = sorted(
            ((c.feature, c.contribution) for c in evidence.contributions),
            key=lambda t: abs(t[1]),
            reverse=True,
        )
        expl = synthesize_explanation(
            verdict_state=verdict_state,
            confidence=confidence,
            fake_probability=calibrated_fake,
            agreement=evidence.agreement,
            provenance_summary=provenance.summary,
            provenance_void=provenance.provenance_void,
            provenance_bifurcated=provenance.bifurcated,
            semantic_flags=[f.as_dict() for f in semantic.flags],
            top_family=top_family,
            top_family_prob=top_family_prob,
            member_probabilities=evidence.member_probabilities,
            top_contributions=top_contribs,
            generator_attribution=attribution.as_dict(),
        )
        explanation_payload = ExplanationPayload(
            severity=expl.severity,
            narrative=expl.narrative,
            sentence_count=expl.sentence_count,
        )

        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)

        response = AnalyzeResponse(
            request_id=uuid.uuid4().hex[:16],
            is_ai=bool(is_ai),
            verdict_state=verdict_state,
            confidence=round(confidence, 2),
            fake_probability=round(calibrated_fake, 4),
            threshold=effective_threshold,
            attribution_mode=self._attributor.mode,
            families=[
                FamilyProbability(family=fam, probability=round(prob, 4))
                for fam, prob in zip(attribution.families, attribution.probabilities, strict=True)
            ],
            features=FeatureReadings(
                spectral_slope=round(float(bundle.fft.spectral_slope), 4),
                spectral_flatness=round(float(bundle.fft.flatness), 6),
                phase_entropy=round(float(bundle.fft.phase_entropy), 4),
                high_freq_ratio=round(float(bundle.fft.high_freq_energy_ratio), 6),
                dct_high_freq_ratio=round(float(bundle.dct.high_freq_energy_ratio), 6),
                wavelet_detail_ratio=round(float(bundle.wavelet.detail_energy_ratio), 4),
                noise_kurtosis=round(float(bundle.noise.residual_kurtosis), 4),
                texture_fractal_dim=round(float(bundle.texture.fractal_dimension), 4),
                peak_prominence=round(float(bundle.fft.peak_prominence), 3),
            ),
            spectrum=[
                SpectrumBin(
                    bin=i,
                    frequency=round(i / max(1, len(bundle.fft.radial_norm) - 1), 4),
                    magnitude=round(float(v), 6),
                )
                for i, v in enumerate(bundle.fft.radial_profile)
            ],
            azimuthal=[
                AzimuthalPoint(
                    angle_deg=int(round(360 * i / max(1, len(bundle.fft.azimuthal_profile)))),
                    magnitude=round(float(v), 6),
                )
                for i, v in enumerate(bundle.fft.azimuthal_profile)
            ],
            wavelet_bands=_wavelet_bands(bundle),
            explainability=ExplainabilityPayload(
                saliency_png_base64=explanation.overlay_png_base64,
                patch_inconsistency_score=round(explanation.patch_inconsistency_score, 4),
                mean_spectral_deviation=round(explanation.mean_deviation, 4),
            ),
            evidence=evidence,
            provenance=provenance_payload,
            semantic=semantic_payload,
            explanation=explanation_payload,
            tta_applied=bool(tta_used),
            latency_ms=latency_ms,
            image_size=(int(loaded.height), int(loaded.width)),
            model_version=MODEL_VERSION,
            sha256=sha256,
        )

        return InferenceOutcome(response=response, latency_ms=latency_ms)


def _evidence_from_result(result: ContributionResult) -> EvidenceBreakdown:
    """Map a ContributionResult to the API schema (E-Masterplan E4)."""
    return EvidenceBreakdown(
        contributions=[
            FeatureContribution(feature=c.feature, value=c.value, contribution=c.contribution)
            for c in result.items
        ],
        member_probabilities={
            name: prob for name, prob in zip(result.member_names, result.member_probabilities, strict=True)
        },
        agreement=round(result.agreement, 4),
    )


def _wavelet_bands(bundle: Any) -> list[WaveletBand]:
    """Reshape flat wavelet energy/entropy arrays into per-band records."""
    energies = np.asarray(bundle.wavelet.subband_energies, dtype=np.float64).ravel()
    entropies = np.asarray(bundle.wavelet.subband_entropies, dtype=np.float64).ravel()
    n_bands_per_level = 3
    levels = len(energies) // n_bands_per_level if len(energies) % n_bands_per_level == 0 else 1
    names = ("LH", "HL", "HH")
    bands: list[WaveletBand] = []
    for lvl in range(levels):
        for b in range(n_bands_per_level):
            idx = lvl * n_bands_per_level + b
            if idx >= len(energies):
                break
            # pywt orders details coarse→fine; display finest as level 1.
            bands.append(
                WaveletBand(
                    level=lvl + 1,
                    band=names[b],
                    energy=round(float(energies[idx]), 6),
                    entropy=round(float(entropies[idx]), 6),
                )
            )
    return bands
