"""Pydantic v2 request/response schemas for the Frequency Guard API.

These models define the wire contract consumed by the React dashboard.
Every response carries a ``request_id`` for tracing through structured
logs, and analysis responses include the real feature payload the charts
render (replacing the previous seeded-PRNG data).
"""

from __future__ import annotations

from pydantic import BaseModel

# --- shared value objects ----------------------------------------------


class FamilyProbability(BaseModel):
    family: str
    probability: float


class FeatureReadings(BaseModel):
    """Scalar spectral statistics surfaced in the dashboard with tooltips."""

    spectral_slope: float
    spectral_flatness: float
    phase_entropy: float
    high_freq_ratio: float
    dct_high_freq_ratio: float
    wavelet_detail_ratio: float
    noise_kurtosis: float
    texture_fractal_dim: float
    peak_prominence: float


class SpectrumBin(BaseModel):
    bin: int
    frequency: float  # normalized 0..1 (fraction of Nyquist)
    magnitude: float


class AzimuthalPoint(BaseModel):
    angle_deg: int
    magnitude: float


class WaveletBand(BaseModel):
    level: int
    band: str  # LH | HL | HH
    energy: float
    entropy: float


class ExplainabilityPayload(BaseModel):
    saliency_png_base64: str
    patch_inconsistency_score: float
    mean_spectral_deviation: float


class FeatureContribution(BaseModel):
    """One feature's signed push toward fake (positive) / real (negative)."""

    feature: str
    value: float
    contribution: float


class EvidenceBreakdown(BaseModel):
    """Per-feature verdict attribution + ensemble agreement (E-Masterplan E4)."""

    contributions: list[FeatureContribution]
    member_probabilities: dict[str, float]
    agreement: float  # 1 = unanimous, 0 = maximally split


class ProvenancePayload(BaseModel):
    """Phase 1 provenance signal (FrequencyGuard)."""

    has_c2pa: bool = False
    has_exif: bool = False
    has_iptc: bool = False
    exif_tags: dict[str, object] = {}
    provenance_void: bool = False
    void_penalty: float = 0.0
    bifurcated: bool = False
    summary: str = "No metadata detected."


class SemanticFlagModel(BaseModel):
    """One semantic-anomaly flag from the CPU arbitrer (Phase 3)."""

    node: str  # "anatomy" | "optics" | "topology"
    label: str
    score: float
    bbox: list[int] | None = None  # [x, y, w, h]


class SemanticEvidence(BaseModel):
    """Aggregated phase-3 semantic-arbiter output."""

    triggered: bool
    suspicious: bool
    flags: list[SemanticFlagModel] = []
    summary: str = "Semantic arbitration not triggered."


class ExplanationPayload(BaseModel):
    """Phase 4 natural-language synthesis (FrequencyGuard)."""

    severity: str  # "[DETERMINISTIC AI]" etc.
    narrative: str
    sentence_count: int = 0


class ModelInfo(BaseModel):
    version: str
    trained_at: str | None = None
    training_mode: str  # "demo" | "manifest"
    accuracy: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    ece: float | None = None
    threshold: float | None = None
    n_features: int | None = None
    n_training_samples: int | None = None


class MetricsSnapshot(BaseModel):
    uptime_seconds: float
    total_analyzed: int
    images_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    peak_rss_mb: float
    cache_entries: int


class HistoryEntry(BaseModel):
    id: int
    request_id: str
    created_at: str
    source: str
    verdict: str  # "ai" | "authentic" | "error"
    verdict_state: str | None = None  # "ai" | "authentic" | "uncertain"
    confidence: float
    fake_probability: float
    latency_ms: float
    model_version: str
    file_sha256: str
    severity: str | None = None  # FrequencyGuard Phase 4 severity prefix
    narrative: str | None = None  # FrequencyGuard Phase 4 natural-language summary


# --- single-image response ---------------------------------------------


class AnalyzeResponse(BaseModel):
    request_id: str
    is_ai: bool
    verdict_state: str = "uncertain"  # "ai" | "authentic" | "uncertain"
    confidence: float  # calibrated, 0..100
    fake_probability: float
    threshold: float
    attribution_mode: str  # "supervised" | "heuristic"
    families: list[FamilyProbability]
    features: FeatureReadings
    spectrum: list[SpectrumBin]
    azimuthal: list[AzimuthalPoint]
    wavelet_bands: list[WaveletBand]
    explainability: ExplainabilityPayload
    evidence: EvidenceBreakdown | None = None  # populated when model is ready
    provenance: ProvenancePayload | None = None  # Phase 1 (FrequencyGuard)
    semantic: SemanticEvidence | None = None  # Phase 3 (FrequencyGuard)
    explanation: ExplanationPayload | None = None  # Phase 4 (FrequencyGuard)
    tta_applied: bool = False  # high-scrutiny mode (E-Masterplan E3.2)
    latency_ms: float
    image_size: tuple[int, int]
    model_version: str
    sha256: str


# --- batch endpoints ----------------------------------------------------


class BatchImageResult(BaseModel):
    filename: str
    index: int
    status: str  # "done" | "error"
    is_ai: bool | None = None
    confidence: float | None = None
    fake_probability: float | None = None
    family: str | None = None
    latency_ms: float | None = None
    error: str | None = None


class BatchJobResponse(BaseModel):
    job_id: str
    status: str  # "queued" | "running" | "completed" | "failed"
    total_images: int
    completed: int
    results: list[BatchImageResult] = []
    csv_url: str | None = None


# --- model performance panel --------------------------------------------


class ConfusionMatrix(BaseModel):
    tn: int
    fp: int
    fn: int
    tp: int


class CurvePoints(BaseModel):
    x: list[float]
    y: list[float]


class GeneratorSlice(BaseModel):
    generator: str
    n: int
    accuracy: float
    f1: float


class PerformanceReport(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    ece: float
    threshold: float
    fpr_at_threshold: float
    confusion_matrix: ConfusionMatrix
    roc_curve: CurvePoints
    pr_curve: CurvePoints
    per_generator: list[GeneratorSlice]
    latency_ms: dict[str, float]
    peak_rss_mb: float
    generated_at: str
