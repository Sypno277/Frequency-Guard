"""FastAPI server (Masterplan §3.1 api/server.py, §2 API LAYER).

CPU-only REST surface consumed by the React dashboard:

    POST /api/v1/analyze          single image → verdict + features + heatmap
    POST /api/v1/batch            multi-file upload → async job id
    GET  /api/v1/jobs/{id}        poll batch progress/results
    GET  /api/v1/jobs/{id}/csv    download batch results as CSV
    GET  /api/v1/metrics          live throughput/latency/memory snapshot
    GET  /api/v1/model            model metadata + performance report
    GET  /api/v1/history          audit log rows
    GET  /health                  liveness probe

Batch jobs run on an in-process ThreadPoolExecutor; per-image results are
appended as they complete so polling clients see incremental progress.
Errors from undecodable images map to meaningful 4xx responses — never 500s
(Masterplan §6 integration-test requirement).
"""

from __future__ import annotations

import csv
import io
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ..config import Settings, ensure_dirs, load_settings
from ..evaluation.evaluate import current_peak_rss_mb
from ..features.extractor import global_cache as feature_cache
from ..io.loader import LoadError
from ..logging import configure_logging, get_logger
from .history import HistoryStore
from .inference import InferenceService, ModelNotReadyError
from .schemas import (
    AnalyzeResponse,
    BatchImageResult,
    BatchJobResponse,
    HistoryEntry,
    MetricsSnapshot,
    ModelInfo,
)

log = get_logger(__name__)

START_TIME = time.perf_counter()


@dataclass
class BatchJob:
    """State of one in-flight/completed batch job."""

    id: str
    total: int
    status: str = "queued"
    completed: int = 0
    results: list[BatchImageResult] = field(default_factory=list)
    errors: int = 0


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory (also used by tests via httpx/TestClient)."""
    settings = settings or load_settings()
    configure_logging(level=settings.log_level)
    ensure_dirs(settings)

    app = FastAPI(
        title="Frequency Guard API",
        version="2.0.0",
        description="CPU-only frequency-domain AI-image detection",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    inference = InferenceService(settings)
    history = HistoryStore(settings.history_db)
    executor = ThreadPoolExecutor(max_workers=max(1, settings.batch_workers))
    jobs: dict[str, BatchJob] = {}
    latency_samples: list[float] = []
    total_analyzed: int = 0

    # ------------------------------------------------------------------
    # health + metrics
    # ------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/metrics", response_model=MetricsSnapshot)
    def metrics() -> MetricsSnapshot:
        """Live service snapshot for the dashboard metrics panel."""
        uptime = time.perf_counter() - START_TIME
        arr = sorted(latency_samples)
        n = len(arr)
        p50 = arr[n // 2] if n else 0.0
        p95 = arr[min(n - 1, int(n * 0.95))] if n else 0.0
        p99 = arr[min(n - 1, int(n * 0.99))] if n else 0.0
        return MetricsSnapshot(
            uptime_seconds=round(uptime, 1),
            total_analyzed=total_analyzed,
            images_per_second=round(total_analyzed / uptime, 3) if uptime > 0 else 0.0,
            latency_p50_ms=round(p50, 2),
            latency_p95_ms=round(p95, 2),
            latency_p99_ms=round(p99, 2),
            peak_rss_mb=current_peak_rss_mb(),
            cache_entries=len(feature_cache),
        )

    # ------------------------------------------------------------------
    # single image
    # ------------------------------------------------------------------

    @app.post("/api/v1/analyze", response_model=AnalyzeResponse)
    def analyze(file: UploadFile = File(...), tta: bool = Query(default=False)) -> AnalyzeResponse:
        """Analyze one uploaded image through the full frequency pipeline.

        ``?tta=true`` enables the high-scrutiny mode (E-Masterplan E3.2):
        predictions averaged over 5 deterministic variants; ~5x latency.
        """
        nonlocal total_analyzed
        try:
            buffer = file.file.read()
            outcome = inference.analyze_bytes(buffer, source=file.filename or "upload", use_tta=tta)
        except LoadError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        resp = outcome.response
        history.insert(
            request_id=resp.request_id,
            source=file.filename or "upload",
            verdict="ai" if resp.is_ai else "authentic",
            verdict_state=resp.verdict_state,
            confidence=resp.confidence,
            fake_probability=resp.fake_probability,
            latency_ms=resp.latency_ms,
            model_version=resp.model_version,
            file_sha256=resp.sha256,
            payload={
                "features": resp.features.model_dump(),
                "families": [f.model_dump() for f in resp.families],
                # FrequencyGuard Phase 4: persist the linguistic summary so the
                # dashboard's narrative feed renders without re-analysis.
                "severity": resp.explanation.severity if resp.explanation else None,
                "narrative": resp.explanation.narrative if resp.explanation else None,
                "provenance": resp.provenance.summary if resp.provenance else None,
                "semantic_suspicious": resp.semantic.suspicious if resp.semantic else False,
            },
        )
        latency_samples.append(resp.latency_ms)
        if len(latency_samples) > 2000:
            del latency_samples[:1000]
        total_analyzed += 1
        return resp

    # ------------------------------------------------------------------
    # batch
    # ------------------------------------------------------------------

    def _run_batch(job: BatchJob, files: list[tuple[str, bytes]]) -> None:
        """Worker: analyze every file sequentially on the thread pool."""
        job.status = "running"
        for idx, (filename, blob) in enumerate(files):
            try:
                outcome = inference.analyze_bytes(blob, source=filename)
                resp = outcome.response
                top_family, _top_prob = max(
                    ((f.family, f.probability) for f in resp.families), key=lambda t: t[1]
                )
                job.results.append(
                    BatchImageResult(
                        filename=filename,
                        index=idx,
                        status="done",
                        is_ai=resp.is_ai,
                        confidence=resp.confidence,
                        fake_probability=resp.fake_probability,
                        family=top_family,
                        latency_ms=resp.latency_ms,
                    )
                )
                history.insert(
                    request_id=resp.request_id,
                    source=filename,
                    verdict="ai" if resp.is_ai else "authentic",
                    verdict_state=resp.verdict_state,
                    confidence=resp.confidence,
                    fake_probability=resp.fake_probability,
                    latency_ms=resp.latency_ms,
                    model_version=resp.model_version,
                    file_sha256=resp.sha256,
                )
                latency_samples.append(resp.latency_ms)
            except (LoadError, ModelNotReadyError, ValueError) as exc:
                job.results.append(
                    BatchImageResult(filename=filename, index=idx, status="error", error=str(exc))
                )
                job.errors += 1
            finally:
                job.completed += 1
        job.status = "completed" if job.errors == 0 else "completed"

    @app.post("/api/v1/batch", response_model=BatchJobResponse)
    async def submit_batch(files: list[UploadFile] = File(...)) -> BatchJobResponse:
        """Queue a batch of images for asynchronous analysis."""
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        payloads: list[tuple[str, bytes]] = []
        for f in files[:64]:
            payloads.append((f.filename or "upload", await f.read()))

        job = BatchJob(id=uuid.uuid4().hex[:12], total=len(payloads))
        jobs[job.id] = job
        executor.submit(_run_batch, job, payloads)
        log.info("batch submitted", extra={"fields": {"job_id": job.id, "n": job.total}})
        return BatchJobResponse(
            job_id=job.id, status=job.status, total_images=job.total, completed=job.completed, results=[]
        )

    @app.get("/api/v1/jobs/{job_id}", response_model=BatchJobResponse)
    def get_job(job_id: str) -> BatchJobResponse:
        """Poll batch-job progress."""
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")
        return BatchJobResponse(
            job_id=job.id,
            status=job.status,
            total_images=job.total,
            completed=job.completed,
            results=list(job.results),
        )

    @app.get("/api/v1/jobs/{job_id}/csv")
    def get_job_csv(job_id: str) -> StreamingResponse:
        """Download batch results as CSV (Masterplan §5.2 export)."""
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "index",
                "filename",
                "status",
                "is_ai",
                "confidence",
                "fake_probability",
                "family",
                "latency_ms",
                "error",
            ]
        )
        for r in sorted(job.results, key=lambda x: x.index):
            writer.writerow(
                [
                    r.index,
                    r.filename,
                    r.status,
                    "" if r.is_ai is None else ("ai" if r.is_ai else "authentic"),
                    r.confidence,
                    r.fake_probability,
                    r.family,
                    r.latency_ms,
                    r.error,
                ]
            )
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="frequency_guard_batch_{job_id}.csv"'},
        )

    # ------------------------------------------------------------------
    # model info + performance
    # ------------------------------------------------------------------

    @app.get("/api/v1/model", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        """Metadata about the currently loaded model."""
        try:
            inference.ensure_model()
        except ModelNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        report = getattr(inference, "_metrics", None)
        ensemble = inference._ensemble
        return ModelInfo(
            version="2.0.0",
            trained_at=inference._trained_at,
            training_mode=inference._training_mode,
            accuracy=(report or {}).get("accuracy"),
            f1=(report or {}).get("f1"),
            roc_auc=(report or {}).get("roc_auc"),
            ece=(report or {}).get("ece"),
            threshold=inference._threshold,
            n_features=len(ensemble.feature_names) if ensemble and ensemble.feature_names else None,
            n_training_samples=int((report or {}).get("n_samples") or 0) or None,
        )

    @app.get("/api/v1/model/performance")
    def model_performance() -> dict[str, object]:
        """Full held-out performance report for the dashboard panel.

        Reads the persisted evaluation report produced at train/eval time;
        returns 409 when no evaluation has been run yet.
        """
        perf_path = settings.model_dir / "performance.json"
        if not perf_path.exists():
            raise HTTPException(status_code=409, detail="No performance report yet. Run evaluate.py first.")
        import json as _json

        return dict(_json.loads(perf_path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # history
    # ------------------------------------------------------------------

    @app.get("/api/v1/history", response_model=list[HistoryEntry])
    def history_rows(limit: int = Query(default=50, ge=1, le=500)) -> list[HistoryEntry]:
        """Recent analysis audit rows, newest first."""
        rows = history.recent(limit=limit)
        return [
            HistoryEntry(
                id=row.id,
                request_id=row.request_id,
                created_at=row.created_at,
                source=row.source,
                verdict=row.verdict,
                verdict_state=row.verdict_state,
                confidence=row.confidence,
                fake_probability=row.fake_probability,
                latency_ms=row.latency_ms,
                model_version=row.model_version,
                file_sha256=row.file_sha256,
                severity=row.severity,
                narrative=row.narrative,
            )
            for row in rows
        ]

    # Exposed for tests / scripts.
    app.state.settings = settings
    app.state.inference = inference
    app.state.history = history
    app.state.jobs = jobs
    app.state.executor = executor

    return app


app = create_app()
