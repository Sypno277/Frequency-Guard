"""One-command API startup with a health-check guard (Layer 2 process fix).

The 500-error root cause was a server that was not running: nothing listened
on port 8000. This script starts uvicorn in a background thread, polls
``/health`` until it returns 200, and exits non-zero (with a clear message)
if the server fails to come up — so the failure is visible instead of
silently surfacing as "Internal Server Error" in the dashboard.

Run:
    python -m scripts.start_api [--port 8000] [--reload]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Running from scripts/ means Python puts scripts/ on sys.path when invoked
# as `python scripts/start_api.py`. Normalize to project root so uvicorn
# imports the package reliably.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn  # noqa: E402

from python_services.frequency_guard.config import Settings, load_settings  # noqa: E402


def _health_check_url(host: str, port: int) -> str:
    # uvicorn binds 0.0.0.0; probe loopback for the check.
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return f"http://{probe_host}:{port}/health"


def _wait_for_health(url: str, timeout: float, interval: float, on_ready: threading.Event) -> None:
    """Poll /health until 200, or raise on timeout before server is ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    on_ready.set()
                    return
        except Exception:
            time.sleep(interval)
    # Server never became healthy within budget.
    print(f"API failed to become healthy at {url} within {timeout}s.", file=sys.stderr)
    os._exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the Frequency Guard API with a health check")
    parser.add_argument("--port", type=int, default=None, help="Port (default: FG_PORT or 8000)")
    parser.add_argument("--host", type=str, default=None, help="Host (default: FG_HOST or 127.0.0.1)")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (dev only)")
    parser.add_argument(
        "--health-timeout", type=float, default=60.0, help="Seconds to wait for /health before aborting"
    )
    args = parser.parse_args()

    settings: Settings = load_settings()
    port = args.port or settings.port
    host = args.host or settings.host
    health_url = _health_check_url(host, port)

    print(f"Starting Frequency Guard API on http://{host}:{port} ...")
    print(f"Health check: {health_url}")

    ready = threading.Event()
    checker = threading.Thread(
        target=_wait_for_health,
        args=(health_url, args.health_timeout, 1.0, ready),
        daemon=True,
    )
    checker.start()

    try:
        uvicorn.run(
            "python_services.frequency_guard.api.server:app",
            host=host,
            port=port,
            reload=args.reload,
        )
    except KeyboardInterrupt:
        print("\nAPI stopped.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
