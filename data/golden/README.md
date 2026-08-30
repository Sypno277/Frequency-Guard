# Golden dataset (locked expectations)

Masterplan §6 regression layout. This directory pins the **expected outputs**
for a small, deterministic sample of images so any feature/verdict drift fails
the build.

- The image files themselves live in `data/demo_dataset/` (procedurally
  generated with fixed seeds by `scripts/train_demo.py`, so they are
  reproducible bit-for-bit without shipping binaries).
- `golden_manifest.csv` lists each sample: relative path, label, and the
  expected fake-verdict direction (0=authentic, 1=AI).
- Locked feature vectors are cached in `reports/golden_cache/*.npy`
  (hash-keyed per sample) by `tests/python/regression/test_golden.py`.
- Tolerance: L2 feature drift must stay below 2%; verdicts must match exactly.

To regenerate everything deterministically:

    python -m scripts.train_demo
    python -m pytest tests/python/regression -m regression
