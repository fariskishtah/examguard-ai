"""Multi-object tracking: ByteTrack (Ultralytics implementation) over person detections.

Track IDs are temporary, anonymous integers. They are NOT identities and can
switch after long occlusions or when people cross paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT
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
    """Minimal Results-like view that BYTETracker/BOTSORT consume (supports bool-mask indexing)."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray, xyxy: np.ndarray | None = None):
        self.xywh, self.conf, self.cls = xywh, conf, cls
        if xyxy is None:  # BoT-SORT's global-motion compensation needs xyxy; without it GMC silently disables itself
            xyxy = np.concatenate([xywh[:, :2] - xywh[:, 2:4] / 2, xywh[:, :2] + xywh[:, 2:4] / 2], axis=1) if len(xywh) else np.zeros((0, 4), np.float32)
        self.xyxy = xyxy

    def __getitem__(self, idx):
        return _Dets(self.xywh[idx], self.conf[idx], self.cls[idx], self.xyxy[idx])

    def __len__(self):
        return len(self.conf)


class PersonTracker:
    """ByteTrack (default) or BoT-SORT over person detections.

    BoT-SORT adds global-motion compensation (`gmc`: none|sparseOptFlow|orb|ecc) and, optionally,
    appearance embeddings from a generic classification network (`reid_model`, e.g. "yolo11n-cls.pt").
    The embeddings are used only to keep tracks apart; no face recognition or identity matching happens.
    """

    def __init__(self, high: float = 0.35, low: float = 0.1, new: float = 0.4, buffer: int = 30, match: float = 0.8,
                 kind: str = "bytetrack", gmc: str = "none", reid_model: str | None = None,
                 appearance_thresh: float = 0.8, proximity_thresh: float = 0.5):
        if kind not in ("bytetrack", "botsort"):
            raise ValueError(f"unknown tracker kind {kind!r}")
        self.args = SimpleNamespace(track_high_thresh=high, track_low_thresh=low, new_track_thresh=new,
                                    track_buffer=buffer, match_thresh=match, fuse_score=True, gmc_method=gmc,
                                    proximity_thresh=proximity_thresh, appearance_thresh=appearance_thresh,
                                    with_reid=bool(reid_model), model=reid_model or "auto", device=None)
        self.kind = kind
        self.version = TRACKER_VERSION if kind == "bytetrack" else f"botsort(ultralytics,gmc={gmc},reid={'on' if reid_model else 'off'})"
        self._cls = BOTSORT if kind == "botsort" else BYTETracker
        self._t = self._cls(self.args)

    def reset(self) -> None:
        self._t = self._cls(self.args)

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
