"""Drawing of boxes / status colours onto frames (BGR)."""
from __future__ import annotations

import cv2
import numpy as np

from examguard.detection.detector import Detection
from examguard.events.engine import EVENT_LABELS, Incident
from examguard.tracking.tracker import TrackedPerson

GREEN, RED, AMBER = (80, 200, 90), (60, 60, 235), (30, 170, 245)
PHONE_COLOR = (255, 160, 40)
STATUS_NORMAL, STATUS_PENDING, STATUS_FLAGGED = "NORMAL", "PENDING", "FLAGGED"



def fmt_ts(ts: float) -> str:
    ms = int(round((ts - int(ts)) * 1000))
    m, s = divmod(int(ts), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}.{ms:03d}"


def _label(img, text, x, y, color, scale):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(max(0, min(x, img.shape[1] - tw - 8)))  # keep the label inside the frame
    y0 = int(min(max(th + 4, y), img.shape[0] - bl - 2))
    cv2.rectangle(img, (x, y0 - th - 4), (x + tw + 6, y0 + bl - 2), color, -1)
    cv2.putText(img, text, (x + 3, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_frame(frame: np.ndarray, tracks: list[TrackedPerson], phones: list[Detection],
               active: dict[int, list[Incident]], pending: set[int] | None = None,
               timestamp_s: float | None = None, only_track: int | None = None) -> np.ndarray:
    """Return an annotated COPY. GREEN = normal, AMBER = unconfirmed evidence, RED = active event."""
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.45, min(w, h) / 1100)
    th = max(1, int(round(min(w, h) / 360)))
    pending = pending or set()
    for tr in tracks:
        if only_track is not None and tr.track_id != only_track:
            color, status = GREEN, STATUS_NORMAL
        elif tr.track_id in active and active[tr.track_id]:
            color, status = RED, STATUS_FLAGGED
        elif tr.track_id in pending:
            color, status = AMBER, STATUS_PENDING
        else:
            color, status = GREEN, STATUS_NORMAL
        x1, y1, x2, y2 = (int(v) for v in tr.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, th + (1 if status == STATUS_FLAGGED else 0))
        text = f"Track #{tr.track_id} {tr.confidence:.2f}"
        if status == STATUS_FLAGGED:
            text += " | REVIEW: " + " + ".join(
                {"PHONE_VISIBLE_NEAR_TRACK": "phone-like object", "MONITORED_REGION_EXIT": "region exit"}[i.event_type]
                for i in active[tr.track_id])
        _label(out, text, x1, y1 - 2, color, scale)
    for ph in phones:
        x1, y1, x2, y2 = (int(v) for v in ph.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), PHONE_COLOR, th)
        _label(out, f"phone {ph.confidence:.2f}", x1, y2 + int(18 * scale * 1.5), PHONE_COLOR, scale * 0.9)
    if timestamp_s is not None:
        _label(out, f"t={fmt_ts(timestamp_s)}", 6, int(22 * scale * 1.5), (60, 60, 60), scale)
    return out


def draw_region(frame: np.ndarray, polygon_norm: list[list[float]], color=(255, 200, 0)) -> None:
    if not polygon_norm:
        return
    h, w = frame.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in polygon_norm], np.int32)
    cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)


def draw_evidence_banner(img: np.ndarray, inc: Incident) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(0.5, min(w, h) / 900)
    bar = int(34 * scale * 1.5)
    out = cv2.copyMakeBorder(img, bar, bar, 0, 0, cv2.BORDER_CONSTANT, value=(40, 30, 20))
    cv2.putText(out, f"{EVENT_LABELS[inc.event_type]} #{inc.track_id} | t={fmt_ts(inc.started_ts)} | "
                     f"conf={inc.confidence:.2f}", (8, int(bar * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.75,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "Automated observation requiring human review - not proof of misconduct.",
                (8, h + bar + int(bar * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, (200, 200, 255), 1, cv2.LINE_AA)
    return out
