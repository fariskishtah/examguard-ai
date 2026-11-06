"""Locating model weights in dev, packaged (PyInstaller) and first-run situations."""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

from examguard.config import default_data_dir

WEIGHTS_NAME = "yolo11n.pt"
# Official Ultralytics asset release. Weights are AGPL-3.0; see docs/LICENSES.md.
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"


def candidate_dirs() -> list[Path]:
    dirs = []
    if getattr(sys, "frozen", False):
        dirs.append(Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "models")
    dirs.append(Path(__file__).resolve().parents[3] / "models")  # source checkout
    dirs.append(default_data_dir() / "models")
    return dirs


def find_weights(configured: str = f"models/{WEIGHTS_NAME}") -> Path | None:
    p = Path(configured).expanduser()
    if p.is_absolute() and p.is_file():
        return p
    for d in candidate_dirs():
        c = d / p.name
        if c.is_file():
            return c
    if p.is_file():
        return p.resolve()
    return None


def download_weights(dest_dir: Path | None = None, url: str = WEIGHTS_URL) -> Path:
    """User-initiated download of the official weights (never automatic)."""
    dest_dir = dest_dir or (default_data_dir() / "models")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / WEIGHTS_NAME
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "examguard-ai"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()
