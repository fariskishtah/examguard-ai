"""Video validation, metadata and timestamped frame iteration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from examguard.config import MAX_VIDEO_BYTES, SUPPORTED_VIDEO_EXTENSIONS


class VideoError(Exception):
    """Unsupported, empty, corrupted or oversized video."""


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    size_bytes: int


def validate_video_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_file():
        raise VideoError(f"File not found: {p.name}")
    if p.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise VideoError(f"Unsupported format '{p.suffix}'. Supported: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}")
    size = p.stat().st_size
    if size == 0:
        raise VideoError("File is empty")
    if size > MAX_VIDEO_BYTES:
        raise VideoError(f"File exceeds the {MAX_VIDEO_BYTES // 1024**3} GiB input limit")
    return p


def probe_video(path: str | Path) -> VideoInfo:
    p = validate_video_path(path)
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise VideoError("Could not open video (corrupted or unsupported codec)")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if w <= 0 or h <= 0:
            raise VideoError("Video has invalid dimensions")
        ok, _ = cap.read()  # a corrupted container can open but not decode
        if not ok:
            raise VideoError("Video contains no decodable frames")
        if not (1.0 <= fps <= 240.0) or n <= 0:
            # Container metadata is implausible (e.g. some WebM report a 1000 fps timebase and a
            # bogus frame count). Derive the real values from per-frame timestamps instead.
            n, fps = _scan_timestamps(p)
            if n <= 0:
                raise VideoError("Video contains no decodable frames")
        return VideoInfo(p, w, h, fps, n, n / fps, p.stat().st_size)
    finally:
        cap.release()


def _scan_timestamps(path: Path) -> tuple[int, float]:
    """Count frames and estimate average FPS from decoder timestamps (one full pass, no conversion)."""
    cap = cv2.VideoCapture(str(path))
    try:
        n, first, last = 0, None, 0.0
        while cap.grab():
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            first = ts if first is None else first
            last, n = ts, n + 1
        if n < 2 or last <= (first or 0.0):
            return n, 30.0  # unknowable: assume 30 and say so in docs
        return n, (n - 1) / (last - first)
    finally:
        cap.release()


def iter_frames(info: VideoInfo, stride: int = 1, decode_skipped: bool = True
                ) -> Iterator[tuple[int, float, np.ndarray | None]]:
    """Yield (frame_index, timestamp_s, frame_bgr). Frames are streamed, never buffered.

    Timestamps come from the container (CAP_PROP_POS_MSEC) when monotonic, which handles
    variable-FPS input; otherwise index/fps. Skipped frames yield frame=None when
    decode_skipped is False (cheaper grab()).
    """
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise VideoError("Could not open video")
    idx, last_ts = 0, -1.0
    try:
        while True:
            want = (idx % stride == 0) or decode_skipped
            if want:
                ok, frame = cap.read()
            else:
                ok, frame = cap.grab(), None
            if not ok:
                break
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not (ts > last_ts or idx == 0) or ts < 0:
                ts = idx / info.fps
            if idx == 0 and ts != 0.0 and ts > 1.0:
                ts = 0.0
            last_ts = ts
            yield idx, ts, frame
            idx += 1
    finally:
        cap.release()
