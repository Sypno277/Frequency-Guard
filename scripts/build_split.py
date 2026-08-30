"""Build a train / held-out split that eliminates validation leakage.

The repository's core integrity problem (Diagnosis Report, Layer 3) is that
the benchmark images — the only real photos + the only genuine AI portrait —
were mixed into the training manifest, so every reported metric was in-sample.

This script partitions the data so that:

  * TRAIN   = the procedural demo dataset (``data/demo_dataset/manifest.csv``)
              plus any user-supplied train-only manifest. The model never
              sees the held-out images during training.
  * HOLDOUT = the benchmark set (``data/benchmark/manifest.csv``): 5 real
              camera photos + 4 AI images x 4 degradation states = 36 rows.
              This is the *only* source of genuine AI output in the repo, so
              it must stay out of training to provide an honest measurement.

Outputs (both under ``data/split/``):

  * train_manifest.csv   -> image_path,label[,generator]
  * holdout_manifest.csv -> image_path,label,state,image_id

Run:
    python -m scripts.build_split
"""

from __future__ import annotations

import csv
from pathlib import Path

from python_services.frequency_guard.config import Settings, ensure_dirs, load_settings
from python_services.frequency_guard.logging import configure_logging, get_logger

log = get_logger(__name__)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _normalize_label(raw: str) -> int:
    val = raw.strip().lower()
    if val in {"real", "0", "authentic", "camera"}:
        return 0
    if val in {"ai", "fake", "1", "generated-ai"}:
        return 1
    return int(float(val))


def build_split(settings: Settings, extra_train_manifest: Path | None = None) -> tuple[Path, Path]:
    """Write train/holdout manifests; returns (train, holdout) paths."""
    split_dir = settings.data_dir / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    # --- Held-out: the benchmark set ------------------------------------
    benchmark = settings.data_dir / "benchmark" / "manifest.csv"
    if not benchmark.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {benchmark}")
    bench_rows = _load_rows(benchmark)
    holdout_rows: list[dict[str, str]] = []
    seen_holdout: set[str] = set()
    for row in bench_rows:
        path = row.get("path") or row.get("file")
        if not path or path in seen_holdout:
            continue
        seen_holdout.add(path)
        holdout_rows.append(
            {
                "image_path": path,
                "label": str(_normalize_label(row.get("label", row.get("state", "real")))),
                "state": row.get("state", "pristine"),
                "image_id": row.get("image_id", row.get("file", path)),
            }
        )

    holdout_path = split_dir / "holdout_manifest.csv"
    with holdout_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "label", "state", "image_id"])
        writer.writeheader()
        writer.writerows(holdout_rows)
    log.info(
        "holdout manifest written", extra={"fields": {"path": str(holdout_path), "n": len(holdout_rows)}}
    )

    # --- Train: demo dataset + optional extra train-only manifest -------
    demo = settings.data_dir / "demo_dataset" / "manifest.csv"
    if not demo.exists():
        raise FileNotFoundError(f"Demo manifest not found: {demo} (run scripts/train_demo.py first)")

    train_rows: dict[str, dict[str, str]] = {}
    for row in _load_rows(demo):
        path = row["image_path"]
        label = str(int(_normalize_label(row["label"])))
        # Windows manifest uses absolute paths; relativize for portability.
        rel = Path(path).resolve().relative_to(settings.data_dir.parent.resolve())
        train_rows[str(rel).replace("\\", "/")] = {"image_path": str(rel).replace("\\", "/"), "label": label}

    if extra_train_manifest is not None and extra_train_manifest.exists():
        for row in _load_rows(extra_train_manifest):
            path = row["image_path"]
            label = str(int(_normalize_label(row["label"])))
            norm = str(Path(path)).replace("\\", "/")
            if norm in seen_holdout:
                log.warning("dropping held-out image from train split", extra={"fields": {"path": norm}})
                continue
            train_rows[norm] = {"image_path": norm, "label": label}

    train_path = split_dir / "train_manifest.csv"
    with train_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(train_rows.values())
    log.info("train manifest written", extra={"fields": {"path": str(train_path), "n": len(train_rows)}})

    return train_path, holdout_path


def main() -> None:
    settings: Settings = load_settings()
    configure_logging(level=settings.log_level)
    ensure_dirs(settings)
    train_path, holdout_path = build_split(settings)
    print(f"train manifest  : {train_path}")
    print(f"holdout manifest: {holdout_path}")


if __name__ == "__main__":
    main()
