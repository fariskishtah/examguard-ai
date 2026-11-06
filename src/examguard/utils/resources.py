"""Locating model weights in dev, packaged (PyInstaller) and first-run situations."""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

from examguard.config import default_data_dir

WEIGHTS_NAME = "yolo11n.pt"
AVAILABLE_MODELS = ("yolo11n.pt", "yolo11s.pt")  # n = fast default, s = more accurate (see docs/EVALUATION.md)
# Official Ultralytics asset release (URLs and sizes verified). Weights are AGPL-3.0; see docs/LICENSES.md.
WEIGHTS_URL_TEMPLATE = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"
WEIGHTS_URL = WEIGHTS_URL_TEMPLATE.format(name=WEIGHTS_NAME)
KNOWN_SHA256 = {"yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
                "yolo11s.pt": "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"}


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


def download_weights(dest_dir: Path | None = None, url: str | None = None, name: str = WEIGHTS_NAME) -> Path:
    """User-initiated download of official weights (never automatic)."""
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {AVAILABLE_MODELS}")
    url = url or WEIGHTS_URL_TEMPLATE.format(name=name)
    dest_dir = dest_dir or (default_data_dir() / "models")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
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
