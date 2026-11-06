"""Multi-object tracking: ByteTrack (Ultralytics implementation) over person detections.

Track IDs are temporary, anonymous integers. They are NOT identities and can
switch after long occlusions or when people cross paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from examguard.detection.detector import Detection

TRACKER_VERSION = "bytetrack(ultralytics)"


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    frame_index: int
    timestamp_s: float
    detection: Detection | None  # source detection (None only if tracker output unmapped)


class _Dets:
    """Minimal Results-like view that BYTETracker consumes (supports bool-mask indexing)."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh, self.conf, self.cls = xywh, conf, cls

    def __getitem__(self, idx):
        return _Dets(self.xywh[idx], self.conf[idx], self.cls[idx])

    def __len__(self):
        return len(self.conf)


class PersonTracker:
    def __init__(self, high: float = 0.35, low: float = 0.1, new: float = 0.4,
                 buffer: int = 30, match: float = 0.8):
        self.args = SimpleNamespace(track_high_thresh=high, track_low_thresh=low,
                                    new_track_thresh=new, track_buffer=buffer,
                                    match_thresh=match, fuse_score=True)
        self.version = TRACKER_VERSION
        self._t = BYTETracker(self.args)

    def reset(self) -> None:
        self._t = BYTETracker(self.args)

    def update(self, persons: list[Detection], frame_bgr: np.ndarray | None = None) -> list[TrackedPerson]:
        if persons:
            xyxy = np.array([d.bbox for d in persons], dtype=np.float32)
            xywh = np.empty_like(xyxy)
            xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
            xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
            xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
            xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
            conf = np.array([d.confidence for d in persons], dtype=np.float32)
            cls = np.zeros(len(persons), dtype=np.float32)
        else:
            xywh = np.zeros((0, 4), np.float32)
            conf = np.zeros(0, np.float32)
            cls = np.zeros(0, np.float32)
        out = self._t.update(_Dets(xywh, conf, cls), frame_bgr)
        res: list[TrackedPerson] = []
        ref = persons[0] if persons else None
        for row in out:
            x1, y1, x2, y2, tid, score, _c, idx = row[:8]
            src = persons[int(idx)] if 0 <= int(idx) < len(persons) else None
            fi = src.frame_index if src else (ref.frame_index if ref else -1)
            ts = src.timestamp_s if src else (ref.timestamp_s if ref else 0.0)
            res.append(TrackedPerson(int(tid), (float(x1), float(y1), float(x2), float(y2)),
                                     float(score), fi, ts, src))
        return res
