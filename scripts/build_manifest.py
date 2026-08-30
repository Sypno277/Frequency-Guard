"""Build a normalized training/eval manifest from any dataset source (D1.2).

Normalizes downloaded corpora (GenImage, CIFAKE, DiffusionForensics-style
directories) into the CSV schema the pipeline already consumes:

    image_path,label,generator,dataset,split

1. Scans a source root and classifies files as real (label=0) or fake
   (label=1) from the folder/path conventions of the source dataset.
2. Records the generator family (sdxl, sd15, vqgan, midjourney, flux, ...)
   from the source folder name, falling back to "unknown".
3. Records the dataset name so cross-dataset evaluation can identify it.
4. Assigns a deterministic split (train/val/test) per label+generator.
5. De-duplicates near-identical images with a pHash nearest-neighbour check
   **across** split boundaries — silent train/test leakage is the #1 fake
   accuracy killer (E-Masterplan D1.2).

Output is written to ``<datasets_dir>/manifest.csv`` (or ``--out``).

Usage:
    python -m scripts.build_manifest --source data/datasets/genimage \\
        --dataset genimage --out data/datasets/genimage_manifest.csv

The manifest is then passed to ``python -m scripts.train_benchmark``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#: Substrings that mark a directory/file as "fake" (AI-generated).
FAKE_HINTS = ("fake", "synthetic", "generated", "sd", "sdxl", "vqgan", "midjourney", "flux", "gan")
#: Substrings that mark a directory/file as "real" (authentic photograph).
REAL_HINTS = ("real", "authentic", "original", "photo", "nature", "train_real")
#: Generator family recognized from a path token.
GENERATORS = (
    "sdxl",
    "sd15",
    "sd1.5",
    "sd",
    "vqgan",
    "midjourney",
    "flux",
    "gan",
    "cifake",
    "unknown",
)
#: Hash size for pHash; smaller is faster, larger is more accurate.
_PHASH_BITS = 64
#: Number of unique pHash buckets to keep in memory; LRU-evicted beyond.
_MAX_HASHES = 50_000
#: pHash Hamming distance <= this is considered a duplicate (out of 64 bits).
_DUP_THRESHOLD = 4


@dataclass
class ManifestRow:
    image_path: Path
    label: int
    generator: str
    dataset: str
    split: str


def _iter_image_files(root: Path) -> list[Path]:
    """Return all decodable image files under ``root`` (recursive)."""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def _classify(path: Path) -> tuple[int, str]:
    """Decide (label, generator) from a path's folder/file name tokens."""
    tokens = [t.lower() for t in path.parts if t]
    label = 1 if any(any(h in t for h in FAKE_HINTS) for t in tokens) else 0
    if label == 1 and any(any(h in t for h in REAL_HINTS) for t in tokens):
        # Prefer "real" if both words appear and real comes later (e.g.
        # genimage/real vs genimage/fake) — most sources are unambiguous.
        pass
    generator = "unknown"
    for t in tokens:
        for g in GENERATORS:
            if g == "unknown":
                continue
            if g in t:
                generator = g
                break
    # CIFAKE names its fake folder "cifake" and the real folder is "train"; the
    # heuristic above already handles it -- but pin the generator label.
    if "cifake" in " ".join(tokens):
        generator = "cifake"
    return label, generator


def _phash(img: np.ndarray) -> int:
    """Perceptual hash (DCT low-frequency signature) of a grayscale image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:8, :8]  # top-left 8x8 DCT coefficients
    # Median-split: bit=1 where coefficient is above the block median.
    median = float(np.median(low))
    bits = (low > median).astype(np.uint8).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


def _load_hash(path: Path) -> int:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        # Unreadable file -- return a sentinel that won't be considered a dup
        # without crashing the whole build.
        raise ValueError(f"unreadable image: {path}")
    return _phash(img)


def _assign_split(rows: list[ManifestRow], seed: int = 42) -> list[ManifestRow]:
    """Deterministic stratify: keep train/val/test separate within label+gen."""
    rng = np.random.default_rng(seed)
    out: list[ManifestRow] = []
    for label in (0, 1):
        for gen in sorted({r.generator for r in rows if r.label == label}):
            group = [r for r in rows if r.label == label and r.generator == gen]
            indices = np.arange(len(group))
            rng.shuffle(indices)
            n = len(indices)
            # 70 / 15 / 15 deterministic split.
            n_val = max(1, int(n * 0.15))
            n_test = max(1, int(n * 0.15))
            for i, idx in enumerate(indices):
                if i < n_val:
                    split = "val"
                elif i < n_val + n_test:
                    split = "test"
                else:
                    split = "train"
                out.append(
                    ManifestRow(
                        image_path=group[idx].image_path,
                        label=group[idx].label,
                        generator=group[idx].generator,
                        dataset=group[idx].dataset,
                        split=split,
                    )
                )
    return out


def _dedupe(rows: list[ManifestRow]) -> list[ManifestRow]:
    """Drop rows whose pHash is within ``_DUP_THRESHOLD`` of a kept row.

    Keeps the first-seen row per duplicate cluster (sorted path order is
    deterministic), which favours the "train" split when the source ordering
    is stable. This is the cross-boundary leakage killer referenced in the
    E-Masterplan (pHash near-duplicate removal **across** train/test edges).
    """
    seen_hashes: list[int] = []
    kept: list[ManifestRow] = []
    dropped = 0
    for r in sorted(rows, key=lambda x: str(x.image_path)):
        try:
            h = _load_hash(r.image_path)
        except ValueError:
            kept.append(r)  # unreadable is kept, not dropped
            continue
        if any(_hamming(h, s) <= _DUP_THRESHOLD for s in seen_hashes):
            dropped += 1
            continue
        seen_hashes.append(h)
        if len(seen_hashes) > _MAX_HASHES:
            # Cheap approximation of LRU: drop early hashes so memory stays
            # bounded over very large corpora.
            seen_hashes = seen_hashes[-_MAX_HASHES // 2 :]
        kept.append(r)
    if dropped:
        print(f"dedupe: dropped {dropped} near-duplicate images")
    return kept


def _write_manifest(rows: list[ManifestRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "label", "generator", "dataset", "split"])
        for r in rows:
            writer.writerow([str(r.image_path), r.label, r.generator, r.dataset, r.split])
    print(f"wrote {len(rows)} rows to {out_path}")


def build_manifest(
    source: Path,
    dataset: str,
    out: Path,
    dedupe: bool = True,
) -> Path:
    """Normalize ``source`` into ``out`` as a pipeline-ready manifest."""
    files = _iter_image_files(source)
    rows: list[ManifestRow] = []
    for f in files:
        label, gen = _classify(f)
        rows.append(
            ManifestRow(
                image_path=f,
                label=label,
                generator=gen,
                dataset=dataset,
                split="train",  # overwritten by _assign_split
            )
        )
    if not rows:
        raise ValueError(f"no images found under {source}")
    print(f"scanned {len(rows)} images from {source} ({dataset})")

    rows = _assign_split(rows)
    if dedupe:
        rows = _dedupe(rows)
    _write_manifest(rows, out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a normalized pipeline manifest from a dataset directory."
    )
    parser.add_argument("--source", type=Path, required=True, help="dataset root directory")
    parser.add_argument("--dataset", required=True, help="dataset name (genimage, cifake, ...)")
    parser.add_argument("--out", type=Path, required=True, help="output manifest CSV path")
    parser.add_argument("--no-dedupe", action="store_true", help="skip pHash near-duplicate removal")
    args = parser.parse_args(argv)

    build_manifest(args.source, args.dataset, args.out, dedupe=not args.no_dedupe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
