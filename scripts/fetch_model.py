"""Fetch the official Ultralytics YOLO11n weights into ./models (AGPL-3.0, see docs/LICENSES.md)."""
from pathlib import Path

from examguard.utils.resources import WEIGHTS_NAME, download_weights, sha256

OBSERVED_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"  # hash of the file used in development

if __name__ == "__main__":
    dest = Path(__file__).resolve().parents[1] / "models"
    p = dest / WEIGHTS_NAME
    if not p.exists():
        print("downloading", WEIGHTS_NAME, "from the official Ultralytics assets release ...")
        p = download_weights(dest)
    h = sha256(p)
    print(p, h, "(matches development weights)" if h == OBSERVED_SHA256 else "(DIFFERENT from development weights - results may differ)")
