"""Launches the FastAPI Control Plane, ZeroMQ KV Event Bus, Prometheus Exporter, and Web UI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def ensure_ui_built() -> None:
    ui_dir = ROOT / "ui"
    dist_dir = ui_dir / "dist"
    npm_bin = shutil.which("npm")
    if ui_dir.exists() and not dist_dir.exists() and npm_bin:
        if not (ui_dir / "node_modules").exists():
            subprocess.run([npm_bin, "install"], cwd=str(ui_dir), check=True)
        subprocess.run([npm_bin, "run", "build"], cwd=str(ui_dir), check=True)


if __name__ == "__main__":
    ensure_ui_built()
    print("Starting server on http://127.0.0.1:8090")
    uvicorn.run("control_plane.app:app", host="127.0.0.1", port=8090)
