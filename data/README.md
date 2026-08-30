# Frequency Guard — Data Provenance & Licenses

This directory holds the datasets the model is trained/evaluated on. The
_demo_ set is **not representative of real AI-generated images** — it is a
procedural smoke-test corpus. The only durable accuracy fix is training on
real generator output (GenImage / CIFAKE / DiffusionForensics-style) via
`scripts/fetch_datasets.py` + `scripts/build_manifest.py`.

## Folders

| Folder | Purpose | Reproducible |
|---|---|---|
| `demo_dataset/` | Synthetic smoke-test set (procedural, fixed seeds). **Do NOT report accuracy from this.** | Yes (`python -m scripts.train_demo`) |
| `golden/` | Locked golden expectations for regression tests. | Yes |
| `datasets/` | Downloaded real corpora (GenImage, CIFAKE, ...). | Depends on source |

## Sources & Licenses

### GenImage (NeurIPS'23)
- **What**: SD1.5 / SDXL / VQGAN + real photography, large scale.
- **License**: Contact the authors; a request form gates the download.
- **Use**: Primary training corpus. Download the zip manually and import it:
  ```
  python -m scripts.fetch_datasets --genimage-zip <path/to/genimage.zip>
  python -m scripts.build_manifest --source data/datasets/genimage \
      --dataset genimage --out data/datasets/genimage_manifest.csv
  python -m scripts.train_benchmark --manifest data/datasets/genimage_manifest.csv
  ```

### CIFAKE
- **What**: 120k labeled face images (real vs AI-generated).
- **License**: Distributed on Kaggle under CC0-like terms (check the page).
- **Use**: **Cross-dataset evaluation only — never trained on.**
  ```
  python -m scripts.fetch_datasets --cifake
  python -m scripts.build_manifest --source data/datasets/cifake \
      --dataset cifake --out data/datasets/cifake_manifest.csv
  python -m scripts.train_benchmark --manifest data/datasets/genimage_manifest.csv \
      --eval-manifest data/datasets/cifake_manifest.csv
  ```

### DiffusionForensics-style (FLUX / Midjourney-class)
- **What**: Additional modern-generator sets when available.
- **License**: Varies by source; always record it here before use.

## Integrity

- `scripts/fetch_datasets.py` pins SHA256 for known archives (see
  `KNOWN_CHECKSUMS`); a mismatch aborts the download.
- `scripts/build_manifest.py` performs pHash near-duplicate removal **across**
  train/test boundaries so evaluation accuracy cannot be inflated by leakage.

## Reporting rules (E-Masterplan D5)

- The **real-benchmark report** (written to `reports/`) is the only place
  accuracy may be claimed.
- Synthetic demo numbers must be explicitly labeled **"unrepresentative"**.
- All new dataset additions must document provenance + license here before
  use.

## History

- **2026-08-27**: Added `scripts/fetch_datasets.py` + `scripts/build_manifest.py`
  skeleton. No real corpora downloaded yet — run the fetch step to populate
  `datasets/`, then run the benchmark protocol to generate the honest report.
