/**
 * Typed API client for the Frequency Guard backend (Masterplan §3.2, §5).
 *
 * Replaces the simulated inference path: every dashboard view reads from
 * these functions. Single-image analysis posts to /api/v1/analyze; batch
 * uploads poll /api/v1/jobs/{id} until completion.
 */

// --- wire types (mirror python_services/frequency_guard/api/schemas.py) ---

export interface FamilyProbability {
  family: string;
  probability: number;
}

export interface FeatureReadings {
  spectral_slope: number;
  spectral_flatness: number;
  phase_entropy: number;
  high_freq_ratio: number;
  dct_high_freq_ratio: number;
  wavelet_detail_ratio: number;
  noise_kurtosis: number;
  texture_fractal_dim: number;
  peak_prominence: number;
}

export interface SpectrumBin {
  bin: number;
  frequency: number;
  magnitude: number;
}

export interface AzimuthalPoint {
  angle_deg: number;
  magnitude: number;
}

export interface WaveletBand {
  level: number;
  band: "LH" | "HL" | "HH";
  energy: number;
  entropy: number;
}

export interface ExplainabilityPayload {
  saliency_png_base64: string;
  patch_inconsistency_score: number;
  mean_spectral_deviation: number;
}

export interface EvidenceBreakdown {
  contributions: { feature: string; value: number; contribution: number }[];
  member_probabilities: Record<string, number>;
  agreement: number; // 1 = unanimous, 0 = maximally split
}

export interface ProvenancePayload {
  has_c2pa: boolean;
  has_exif: boolean;
  has_iptc: boolean;
  exif_tags: Record<string, unknown>;
  provenance_void: boolean;
  void_penalty: number;
  bifurcated: boolean;
  summary: string;
}

export interface SemanticFlagModel {
  node: "anatomy" | "optics" | "topology";
  label: string;
  score: number;
  bbox?: number[] | null;
}

export interface SemanticEvidence {
  triggered: boolean;
  suspicious: boolean;
  flags: SemanticFlagModel[];
  summary: string;
}

export interface ExplanationPayload {
  severity: string;
  narrative: string;
  sentence_count: number;
}

export interface AnalyzeResponse {
  request_id: string;
  is_ai: boolean;
  verdict_state: "ai" | "authentic" | "uncertain";
  confidence: number;
  fake_probability: number;
  threshold: number;
  attribution_mode: string;
  families: FamilyProbability[];
  features: FeatureReadings;
  spectrum: SpectrumBin[];
  azimuthal: AzimuthalPoint[];
  wavelet_bands: WaveletBand[];
  explainability: ExplainabilityPayload;
  evidence?: EvidenceBreakdown | null;
  provenance?: ProvenancePayload | null;
  semantic?: SemanticEvidence | null;
  explanation?: ExplanationPayload | null;
  latency_ms: number;
  image_size: [number, number];
  model_version: string;
  sha256: string;
}

export interface BatchImageResult {
  filename: string;
  index: number;
  status: "done" | "error";
  is_ai?: boolean | null;
  confidence?: number | null;
  fake_probability?: number | null;
  family?: string | null;
  latency_ms?: number | null;
  error?: string | null;
}

export interface BatchJobResponse {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  total_images: number;
  completed: number;
  results: BatchImageResult[];
  csv_url?: string | null;
}

export interface ModelInfo {
  version: string;
  trained_at?: string | null;
  training_mode: string;
  accuracy?: number | null;
  f1?: number | null;
  roc_auc?: number | null;
  ece?: number | null;
  threshold?: number | null;
  n_features?: number | null;
  n_training_samples?: number | null;
}

export interface MetricsSnapshot {
  uptime_seconds: number;
  total_analyzed: number;
  images_per_second: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  latency_p99_ms: number;
  peak_rss_mb: number;
  cache_entries: number;
}

export interface HistoryEntry {
  id: number;
  request_id: string;
  created_at: string;
  source: string;
  verdict: string;
  verdict_state?: string | null;
  confidence: number;
  fake_probability: number;
  latency_ms: number;
  model_version: string;
  file_sha256: string;
  severity?: string | null;
  narrative?: string | null;
}

export interface PerformanceReport {
  n_samples?: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  ece: number;
  threshold: number;
  fpr_at_threshold: number;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
  roc_curve: { x: number[]; y: number[] };
  pr_curve: { x: number[]; y: number[] };
  per_generator: { generator: string; n: number; accuracy: number; f1: number }[];
  latency_ms: Record<string, number>;
  peak_rss_mb: number;
  generated_at: string;
}

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* keep statusText */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

/** POST one image for full frequency-domain analysis. */
export async function analyzeImage(file: File): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/v1/analyze`, { method: "POST", body: form });
  return handle<AnalyzeResponse>(res);
}

/** POST a batch of images; returns a job to poll. */
export async function submitBatch(files: File[]): Promise<BatchJobResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${BASE}/api/v1/batch`, { method: "POST", body: form });
  return handle<BatchJobResponse>(res);
}

/** GET current batch-job state. */
export async function getJob(jobId: string): Promise<BatchJobResponse> {
  const res = await fetch(`${BASE}/api/v1/jobs/${jobId}`);
  return handle<BatchJobResponse>(res);
}

/** Poll a batch job until it completes or fails; invokes onTick each poll. */
export async function pollJob(
  jobId: string,
  onTick?: (snapshot: BatchJobResponse) => void,
  intervalMs = 700,
  timeoutMs = 5 * 60 * 1000
): Promise<BatchJobResponse> {
  const deadline = Date.now() + timeoutMs;
  let latest: BatchJobResponse | null = null;
  while (Date.now() < deadline) {
    latest = await getJob(jobId);
    onTick?.(latest);
    if (latest.status === "completed" || latest.status === "failed") return latest;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error("Batch job timed out");
}

/** GET live service metrics snapshot. */
export async function getMetrics(): Promise<MetricsSnapshot> {
  const res = await fetch(`${BASE}/api/v1/metrics`);
  return handle<MetricsSnapshot>(res);
}

/** GET loaded-model metadata. */
export async function getModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${BASE}/api/v1/model`);
  return handle<ModelInfo>(res);
}

/** GET held-out performance report (409 when not yet evaluated). */
export async function getPerformance(): Promise<PerformanceReport> {
  const res = await fetch(`${BASE}/api/v1/model/performance`);
  return handle<PerformanceReport>(res);
}

/** GET recent audit-history rows. */
export async function getHistory(limit = 50): Promise<HistoryEntry[]> {
  const res = await fetch(`${BASE}/api/v1/history?limit=${limit}`);
  return handle<HistoryEntry[]>(res);
}
