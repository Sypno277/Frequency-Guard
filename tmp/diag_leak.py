"""Prove the validation leak: are the benchmark images also training images?

Steps:
1. Load the ACTUAL training manifest (data/benchmark/train_manifest.csv).
2. Count how many of the 36 benchmark images are in that manifest.
3. Walk data/ for ALL images and find any NOT in the manifest (truly unseen).
4. If unseen images exist, run the deployed model on them.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Running from tmp/ means Python puts tmp/ on sys.path, not the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python_services.frequency_guard.api.inference import InferenceService
from python_services.frequency_guard.config import Settings, ensure_dirs, load_settings

settings: Settings = load_settings()
ensure_dirs(settings)

# --- 1. Load the actual training manifest -------------------------------
train_manifest = Path("data/benchmark/train_manifest.csv")
train_paths: set[str] = set()
with train_manifest.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        # Normalize backslashes/forward slashes & resolve relative to root
        p = Path(row["image_path"]).resolve()
        train_paths.add(str(p))
print(f"Training manifest: {len(train_paths)} unique image paths")

# --- 2. Benchmark overlap -------------------------------------------------
benchmark_manifest = Path("data/benchmark/manifest.csv")
bench_rows = []
with benchmark_manifest.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        bench_rows.append(row)

in_train = 0
not_in_train = []
for row in bench_rows:
    p = Path(row["path"]).resolve()
    if str(p) in train_paths:
        in_train += 1
    else:
        not_in_train.append(row["path"])

print(f"Benchmark images: {len(bench_rows)}")
print(f"  => IN training manifest: {in_train}/{len(bench_rows)}")
print(f"  => NOT in training manifest: {len(not_in_train)}")
if not_in_train:
    print("  Unseen benchmark images:")
    for p in not_in_train:
        print(f"    {p}")

# --- 3. Scan data/ for ALL images, find truly unseen ----------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
all_images: list[Path] = []
for ext in IMAGE_EXTS:
    all_images.extend(Path("data").rglob(f"*{ext}"))

unseen: list[Path] = []
for img in all_images:
    resolved = str(img.resolve())
    if resolved not in train_paths:
        unseen.append(img)

print(f"\nAll image files under data/: {len(all_images)}")
print(f"  => NOT in training manifest (truly unseen): {len(unseen)}")

# --- 4. Run model on unseen images if any ---------------------------------
if unseen:
    print("\n=== Running deployed model on UNSEEN images ===")
    svc = InferenceService(settings)
    svc.ensure_model()
    for img in unseen[:20]:  # cap at 20 to keep runtime sane
        try:
            outcome = svc.analyze_bytes(img.read_bytes(), source=str(img))
            resp = outcome.response
            print(
                f"  {str(img):70s} pred_ai={resp.is_ai} "
                f"fake_p={resp.fake_probability:.4f} conf={resp.confidence:.2f} "
                f"vstate={resp.verdict_state}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {str(img):70s} ERROR: {exc}")
else:
    print("\nNo unseen images exist in data/. The deployed model has only ever been")
    print("evaluated on images it was trained on (in-sample). No held-out test set.")

# --- 5. Also check: is golden manifest subset of training manifest? --------
golden = Path("data/golden/golden_manifest.csv")
if golden.exists():
    golden_paths = []
    with golden.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            golden_paths.append((row["relative_path"], row["label"]))
    demo_dir = Path("data/demo_dataset")
    golden_in_train = 0
    for rel, _label in golden_paths:
        p = (demo_dir / rel).resolve()
        if str(p) in train_paths:
            golden_in_train += 1
    print(f"\nGolden regression samples: {len(golden_paths)}")
    print(f"  => IN training manifest: {golden_in_train}/{len(golden_paths)}")
