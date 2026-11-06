"""Fetch official Ultralytics YOLO11 weights into ./models (AGPL-3.0, see docs/LICENSES.md).

Usage: python scripts/fetch_model.py [yolo11n|yolo11s ...]   (default: yolo11n)
"""
import sys
from pathlib import Path

from examguard.utils.resources import AVAILABLE_MODELS, KNOWN_SHA256, download_weights, sha256

if __name__ == "__main__":
    dest = Path(__file__).resolve().parents[1] / "models"
    for arg in sys.argv[1:] or ["yolo11n"]:
        name = arg if arg.endswith(".pt") else arg + ".pt"
        if name not in AVAILABLE_MODELS:
            raise SystemExit(f"unknown model {arg!r}; choose from {AVAILABLE_MODELS}")
        p = dest / name
        if not p.exists():
            print("downloading", name, "from the official Ultralytics assets release ...")
            p = download_weights(dest, name=name)
        h = sha256(p)
        print(p, h, "(matches development weights)" if h == KNOWN_SHA256[name] else "(DIFFERENT from development weights - results may differ)")
