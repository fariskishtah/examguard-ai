from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
FIXTURES = ROOT / "data" / "fixtures"
WEIGHTS = ROOT / "models" / "yolo11n.pt"


def write_video(path: Path, n: int = 30, fps: float = 25.0, size=(160, 120), tag="mp4v") -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*tag), fps, size)
    assert w.isOpened()
    for i in range(n):
        img = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.rectangle(img, (10 + i, 20), (50 + i, 100), (200, 200, 200), -1)
        w.write(img)
    w.release()
    return path


@pytest.fixture()
def tiny_video(tmp_path):
    return write_video(tmp_path / "tiny.mp4")


needs_model = pytest.mark.skipif(not WEIGHTS.exists(), reason="models/yolo11n.pt missing (run scripts/fetch_model.py)")
needs_samples = pytest.mark.skipif(not (SAMPLES / "MANIFEST.json").exists(), reason="run scripts/fetch_samples.py")
needs_fixture = pytest.mark.skipif(not (FIXTURES / "synthetic_phone_pedestrian.mp4").exists(),
                                   reason="run scripts/make_synthetic_fixture.py")
