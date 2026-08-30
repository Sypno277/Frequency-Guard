"""Centralized, environment-driven configuration for Frequency Guard.

All paths and tunables live here (or in environment variables). No module in
the package may hardcode a path or parameter — import from this module.

Importing this module has no side effects; call :func:`ensure_dirs` from
service/training entry points that actually need the directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SERVICE_DIR.parent.parent


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Override any value via environment variable FG_*.

    Example:
        export FG_IMAGE_SIZE=256
        export FG_LOG_LEVEL=DEBUG
    """

    # --- ingestion / preprocessing -----------------------------------
    image_size: int = 256
    supported_formats: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    max_upload_bytes: int = 25 * 1024 * 1024

    # --- feature extraction ------------------------------------------
    dct_block_size: int = 8
    wavelet_name: str = "db4"
    wavelet_levels: int = 3
    radial_bins: int = 24
    azimuthal_bins: int = 18
    tiles_per_side: int = 4

    # --- classifier / calibration ------------------------------------
    n_folds: int = 5
    random_state: int = 42
    calibrate_method: str = "isotonic"  # "isotonic" | "sigmoid"
    threshold_fpr_target: float = 0.02

    # --- service ------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    batch_workers: int = 2
    cache_size: int = 256

    # --- logging / persistence ---------------------------------------
    log_level: str = "INFO"
    log_file: str | None = None

    # --- paths (defaults relative to project root) --------------------
    model_dir: Path = PROJECT_ROOT / "checkpoints"
    reports_dir: Path = PROJECT_ROOT / "reports"
    data_dir: Path = PROJECT_ROOT / "data"
    history_db: Path = PROJECT_ROOT / "data" / "frequency_guard.sqlite3"

    @property
    def path_dirs(self) -> tuple[Path, ...]:
        """All directories that service/training code may require."""
        return (self.model_dir, self.reports_dir, self.data_dir)


def load_settings() -> Settings:
    """Build a Settings instance from environment variables (FG_*)."""
    import os

    kwargs: dict[str, object] = {}
    prefix = "FG_"
    for name in dir(Settings):
        if name.startswith("_"):
            continue
        env_key = prefix + name.upper()
        raw = os.environ.get(env_key)
        if raw is None:
            continue
        current = getattr(Settings, name)
        if isinstance(current, bool):
            kwargs[name] = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            kwargs[name] = int(raw)
        elif isinstance(current, float):
            kwargs[name] = float(raw)
        elif isinstance(current, tuple | list):
            kwargs[name] = tuple(item.strip() for item in raw.split(",") if item.strip())
        else:
            kwargs[name] = str(raw)
    return Settings(**kwargs)  # type: ignore[arg-type]


def ensure_dirs(settings: Settings) -> None:
    """Create the directories required by ``settings`` (idempotent)."""
    for path in settings.path_dirs:
        path.mkdir(parents=True, exist_ok=True)
        path.joinpath(".gitkeep").touch(exist_ok=True)
