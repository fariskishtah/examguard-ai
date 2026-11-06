import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import write_video
from examguard.config import MAX_VIDEO_BYTES, Paths
from examguard.detection.detector import Detection
from examguard.events.engine import PHONE_EVENT, Incident
from examguard.evidence.capture import atomic_write_jpeg, capture_evidence, safe_delete, sanitize, unique_stem
from examguard.tracking.tracker import TrackedPerson
from examguard.video.annotate import GREEN, RED, draw_frame
from examguard.video.reader import VideoError, iter_frames, probe_video, validate_video_path


# ---- reader -----------------------------------------------------------
def test_probe_metadata_and_timestamps(tiny_video):
    info = probe_video(tiny_video)
    assert (info.width, info.height, info.frame_count) == (160, 120, 30)
    assert info.fps == pytest.approx(25.0)
    ts = [t for _, t, _ in iter_frames(info)]
    assert len(ts) == 30 and ts[0] == 0.0
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert ts[-1] == pytest.approx(29 / 25, abs=0.02)  # no drift


def test_stride_skips_decode_when_asked(tiny_video):
    info = probe_video(tiny_video)
    out = list(iter_frames(info, stride=5, decode_skipped=False))
    assert len(out) == 30
    assert [i for i, _, f in out if f is not None] == [0, 5, 10, 15, 20, 25]


@pytest.mark.parametrize("name,content,msg", [
    ("empty.mp4", b"", "empty"),
    ("junk.mp4", b"this is not a video" * 100, "open|decod"),
    ("clip.txt", b"x", "Unsupported"),
])
def test_bad_inputs_rejected(tmp_path, name, content, msg):
    p = tmp_path / name
    p.write_bytes(content)
    with pytest.raises(VideoError, match=msg):
        probe_video(p)


def test_missing_and_truncated(tmp_path, tiny_video):
    with pytest.raises(VideoError, match="not found"):
        probe_video(tmp_path / "nope.mp4")
    data = Path(tiny_video).read_bytes()
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(data[: len(data) // 3])
    with pytest.raises(VideoError):
        probe_video(cut)


def test_size_limit(tmp_path, monkeypatch):
    p = tmp_path / "big.mp4"
    p.write_bytes(b"x" * 10)
    monkeypatch.setattr("examguard.video.reader.MAX_VIDEO_BYTES", 5)
    with pytest.raises(VideoError, match="limit"):
        validate_video_path(p)
    assert MAX_VIDEO_BYTES > 0


# ---- annotate / evidence --------------------------------------------------
def _tr(tid, box=(10, 10, 60, 100)):
    return TrackedPerson(tid, box, 0.9, 0, 0.0, None)


def _inc(tid=2):
    i = Incident(PHONE_EVENT, tid, 1.5, 37, 1.5, 37, (10, 10, 60, 100), (20, 30, 35, 50))
    i.add_support(37, 1.5, 0.6, 0.9, (10, 10, 60, 100), (20, 30, 35, 50))
    return i


def test_box_colours_green_normal_red_active():
    frame = np.zeros((120, 160, 3), np.uint8)
    inc = _inc(2)
    img = draw_frame(frame, [_tr(1, (5, 20, 40, 100)), _tr(2, (80, 20, 120, 100))], [], {2: [inc]})
    assert tuple(img[60, 5]) == GREEN       # left edge of track 1
    assert tuple(img[60, 80]) == RED        # left edge of track 2
    assert frame.sum() == 0                 # input never mutated


def test_sanitize_and_unique_names():
    assert sanitize("../../etc/passwd") == "etc_passwd"
    assert sanitize("") == "x"
    a, b = unique_stem(1, _inc()), unique_stem(1, _inc())
    assert a != b and "/" not in a


def test_capture_writes_raw_and_annotated(tmp_path):
    gx, gy = np.meshgrid(np.linspace(0, 255, 160), np.linspace(0, 255, 120))
    frame = np.dstack([gx, gy, (gx + gy) / 2]).astype(np.uint8)  # smooth image: JPEG-friendly
    ph = Detection("cell phone", 67, 0.6, (20, 30, 35, 50), 37, 1.5, "t")
    ev = capture_evidence(tmp_path / "ev", 1, _inc(), frame, [_tr(2)], [ph], {})
    raw = cv2.imread(str(ev.raw))
    assert raw.shape == frame.shape
    assert np.abs(raw.astype(int) - frame.astype(int)).mean() < 3  # raw == real frame (JPEG loss only)
    ann = cv2.imread(str(ev.annotated))
    assert ann.shape[0] > frame.shape[0] and ann.shape[1] == frame.shape[1]  # banner added
    assert oct(os.stat(ev.raw).st_mode & 0o777) == "0o600"
    assert not list((tmp_path / "ev").glob("*.tmp"))


def test_no_overwrite(tmp_path):
    p = tmp_path / "a.jpg"
    img = np.zeros((10, 10, 3), np.uint8)
    atomic_write_jpeg(p, img)
    with pytest.raises(FileExistsError):
        atomic_write_jpeg(p, img)
    assert not list(tmp_path.glob("*.tmp"))


def test_safe_delete_stays_inside_root(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    inside, outside = root / "a.jpg", tmp_path / "keep.jpg"
    inside.write_bytes(b"1"); outside.write_bytes(b"1")
    n = safe_delete([str(inside), str(outside), str(root / ".." / "keep.jpg")], root)
    assert n == 1 and not inside.exists() and outside.exists()


def test_paths_are_private_and_independent_of_repo(tmp_path):
    p = Paths(tmp_path / "home").ensure()
    assert p.db.parent == p.root and p.screenshots.is_dir() and p.logs.is_dir()
    assert oct(os.stat(p.root).st_mode & 0o777) == "0o700"
    assert "examguard-ai" not in str(p.root)


def test_model_download_helper_validates_name_and_urls():
    from examguard.utils.resources import AVAILABLE_MODELS, KNOWN_SHA256, WEIGHTS_URL_TEMPLATE, download_weights
    assert set(AVAILABLE_MODELS) == set(KNOWN_SHA256) and all(len(v) == 64 for v in KNOWN_SHA256.values())
    assert WEIGHTS_URL_TEMPLATE.format(name="yolo11s.pt").endswith("/yolo11s.pt")
    with pytest.raises(ValueError):
        download_weights(name="evil.pt")  # only allow-listed official weights can be fetched
