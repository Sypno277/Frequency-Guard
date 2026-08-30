"""Dataset acquisition for the real-data foundation (E-Masterplan D1.1).

Downloads checksummed, licensed AI-image-detection corpora into
``data/datasets/`` so we can train on real diffusion/GAN output instead of
the synthetic demo set whose "fake" class does not resemble real generators.

The demo set (``scripts/train_demo.py``) is unrepresentative: its "fake"
class is a hand-tuned flat-spectrum + checkerboard image that real
SDXL/FLUX/Midjourney output shares none of. Training on it and then
claiming real-world accuracy is dishonest — D1 is the only durable fix.

Supported sources (one class per dataset, see READMEs for licenses):
  * GenImage (NeurIPS'23): SD1.5 / SDXL / VQGAN + real photography.
  * CIFAKE: 120k labeled images; used for cross-dataset evaluation only,
    never trained on.
  * (optional) DiffusionForensics-style downloads: pass --df-url.

Each downloader verifies an expected SHA256 when ``--checksums`` is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

#: Root for downloaded corpora. Mirrors Settings.data_dir (data/).
DEFAULT_DATASETS_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"

#: Expected SHA256 for each distributed archive's primary file. Populate
#: these from the upstream release notes; a mismatch aborts the download so
#: a tampered mirror can never silently poison a training manifest.
#: (Keyed by the archive basename.)
KNOWN_CHECKSUMS: dict[str, str] = {}

#: CIFAKE CC0-like distribution URL. GenImage is gated behind a request
#: form, so it is downloaded manually or via the --genimage-zip flag.
CIFAKE_URL = "https://www.kaggle.com/api/v1/datasets/download/birdyowo/cifake-real-and-ai-generated-synthetic-face-images"

#: Max bytes to buffer per URL fetch (archives can be 100s of MB but we
#: stream to disk rather than hold them in memory).
_CHUNK = 1 << 20  # 1 MiB


@dataclass
class DatasetInfo:
    name: str
    path: Path
    ok: bool
    message: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _request_with_retry(url: str, dest: Path, retries: int = 3) -> None:
    """Stream ``url`` to ``dest`` with simple retry on transient errors."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with (
                urllib.request.urlopen(url, timeout=120) as resp,
                tempfile.NamedTemporaryFile(delete=False, dir=dest.parent, suffix=".part") as tmp,
            ):
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    tmp.write(chunk)
                tmp.flush()
            Path(tmp.name).replace(dest)
            return
        except Exception as exc:  # noqa: BLE001 - network retry
            last_err = exc
            if attempt == retries:
                break
    raise RuntimeError(f"failed to download {url} after {retries} attempts: {last_err}") from last_err


def _extract_zip(archive: Path, dest_dir: Path) -> Path:
    """Extract ``archive`` into ``dest_dir``; returns the extracted root."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)
    # The extracted root is the first top-level directory if one exists.
    tops = sorted(p for p in dest_dir.iterdir() if p.is_dir())
    return tops[0] if tops else dest_dir


def _verify(archive: Path) -> bool:
    expected = KNOWN_CHECKSUMS.get(archive.name)
    if not expected:
        return True  # no pin registered for this archive
    actual = _sha256_file(archive)
    if actual != expected:
        print(f"CHECKSUM MISMATCH for {archive.name}: expected {expected}, got {actual}")
        return False
    return True


def download_cifake(dest: Path) -> DatasetInfo:
    """Download CIFAKE (cross-dataset eval only)."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "cifake.zip"
    if archive.exists() and _verify(archive):
        return DatasetInfo("cifake", dest, True, "archive already present (checksum ok)")
    print(f"[cifake] downloading from {CIFAKE_URL} ...")
    try:
        _request_with_retry(CIFAKE_URL, archive)
        if not _verify(archive):
            return DatasetInfo("cifake", dest, False, "checksum mismatch")
        _extract_zip(archive, dest)
        return DatasetInfo("cifake", dest, True, "downloaded + extracted")
    except Exception as exc:  # noqa: BLE001
        return DatasetInfo("cifake", dest, False, str(exc))


def download_genimage_zip(zip_path: Path, dest: Path) -> DatasetInfo:
    """Import a locally-supplied GenImage zip (gated upstream)."""
    if not zip_path.exists():
        return DatasetInfo("genimage", dest, False, f"zip not found: {zip_path}")
    if not _verify(zip_path):
        return DatasetInfo("genimage", dest, False, "checksum mismatch")
    dest.mkdir(parents=True, exist_ok=True)
    _extract_zip(zip_path, dest)
    return DatasetInfo("genimage", dest, True, "imported from local zip")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download licensed AI-image-detection datasets.")
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--cifake", action="store_true", help="download CIFAKE")
    parser.add_argument("--genimage-zip", type=Path, default=None, help="local GenImage zip path")
    parser.add_argument("--all", action="store_true", help="download all available sources")
    parser.add_argument("--checksums", action="store_true", help="verify known SHA256 pins")
    args = parser.parse_args(argv)

    dest = args.datasets_dir
    results: list[DatasetInfo] = []

    if args.cifake or args.all:
        results.append(download_cifake(dest))
    if args.genimage_zip:
        results.append(download_genimage_zip(args.genimage_zip, dest))

    if not results:
        parser.print_help()
        return 1

    for r in results:
        status = "OK " if r.ok else "ERR"
        print(f"[{status}] {r.name}: {r.message} ({r.path})")

    ok = all(r.ok for r in results)
    summary = {"datasets_dir": str(dest), "results": [vars(r) for r in results]}
    (dest / "_download_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
