"""Evidence screenshot capture: genuine frames, unique names, atomic writes."""
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from examguard.detection.detector import Detection
from examguard.events.engine import Incident
from examguard.tracking.tracker import TrackedPerson
from examguard.video.annotate import draw_evidence_banner, draw_frame, draw_region


@dataclass(frozen=True)
class EvidencePaths:
    annotated: Path
    raw: Path


def sanitize(text: str, maxlen: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:maxlen] or "x"


def unique_stem(session_id: int, inc: Incident) -> str:
    ms = int(inc.started_ts * 1000)
    return f"s{session_id}_t{inc.track_id}_{sanitize(inc.event_type, 24).lower()}_{ms:08d}ms_{uuid.uuid4().hex[:8]}"


def atomic_write_jpeg(path: Path, img: np.ndarray, quality: int = 92) -> None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError("JPEG encoding failed")
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:6]}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path.name}")
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def capture_evidence(out_dir: Path, session_id: int, inc: Incident, frame: np.ndarray,
                     tracks: list[TrackedPerson], phones: list[Detection],
                     active: dict[int, list[Incident]], region_polygon: list[list[float]] | None = None
                     ) -> EvidencePaths:
    """Save the unmodified frame plus an annotated review copy of the SAME frame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = unique_stem(session_id, inc)
    raw, annotated = out_dir / f"{stem}_raw.jpg", out_dir / f"{stem}_annotated.jpg"
    relevant_phones = [p for p in phones if inc.phone_bbox is not None and p.bbox == inc.phone_bbox] \
        if inc.phone_bbox else []
    img = draw_frame(frame, tracks, relevant_phones, active, timestamp_s=None, only_track=inc.track_id)
    if inc.region_id and region_polygon:
        draw_region(img, region_polygon)
    img = draw_evidence_banner(img, inc)
    atomic_write_jpeg(raw, frame)
    atomic_write_jpeg(annotated, img)
    return EvidencePaths(annotated, raw)


def safe_delete(paths: list[str], root: Path) -> int:
    """Delete evidence files, but only inside `root` (never follow arbitrary DB paths)."""
    root = root.resolve()
    n = 0
    for p in paths:
        try:
            rp = Path(p).resolve()
            if root in rp.parents and rp.is_file():
                rp.unlink()
                n += 1
        except OSError:
            continue
    return n
