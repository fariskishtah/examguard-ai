"""End-to-end video processing: decode -> detect -> track -> events -> evidence -> DB -> output.

The processor is UI-agnostic. It reports progress through an `emit(kind, payload)` callback and
honours a `ProcessingControl` (pause / resume / stop). Frames are streamed (constant memory).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from examguard.config import Paths, Settings
from examguard.database.repository import Repository
from examguard.detection.detector import PERSON, PERSON_TRACKER_FLOOR, PHONE, YoloDetector, file_sha256, merge_detections
from examguard.events.engine import EventEngine, Incident
from examguard.evidence.capture import capture_evidence
from examguard.notifications.notifier import Alert, native_notify
from examguard.tracking.tracker import TRACKER_VERSION, PersonTracker, TrackedPerson
from examguard.video.annotate import draw_frame, draw_region
from examguard.video.reader import VideoInfo, iter_frames, probe_video

log = logging.getLogger(__name__)
Emit = Callable[[str, object], None]


class ProcessingControl:
    """Thread-safe pause/stop flags shared with the UI thread."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._run = threading.Event()
        self._run.set()

    def pause(self) -> None: self._run.clear()
    def resume(self) -> None: self._run.set()
    def stop(self) -> None:
        self._stop.set()
        self._run.set()  # release a paused worker so it can exit

    @property
    def stopped(self) -> bool: return self._stop.is_set()
    @property
    def paused(self) -> bool: return not self._run.is_set()

    def wait_if_paused(self) -> None:
        while not self._run.wait(timeout=0.1):
            if self._stop.is_set():
                return


@dataclass
class ProcessingSummary:
    session_id: int
    status: str
    frames_read: int = 0
    frames_inferred: int = 0
    wall_seconds: float = 0.0
    n_tracks: int = 0
    n_incidents: int = 0
    avg_fps: float = 0.0
    avg_infer_ms: float = 0.0
    annotated_video: str | None = None
    error: str | None = None
    incident_ids: list[int] = field(default_factory=list)


@dataclass
class FrameUpdate:
    frame_index: int
    timestamp_s: float
    total_frames: int
    fps_proc: float
    image: np.ndarray | None  # annotated, full-frame-resolution (UI rescales it)
    n_tracks: int


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter | None:
    for tag in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*tag), fps, size)
        if w.isOpened():
            return w
        w.release()
    return None


class VideoProcessor:
    def __init__(self, settings: Settings, paths: Paths, repo: Repository,
                 detector: YoloDetector | None = None, emit: Emit | None = None,
                 preview_interval_s: float = 1 / 15):
        self.s = settings.validate()
        self.paths = paths.ensure()
        self.repo = repo
        self._detector = detector
        self.emit: Emit = emit or (lambda k, p: None)
        self.preview_interval_s = preview_interval_s

    @property
    def detector(self) -> YoloDetector:
        if self._detector is None:
            self._detector = YoloDetector(self.s.weights_path, PERSON_TRACKER_FLOOR, self.s.phone_conf,
                                          self.s.nms_iou, self.s.imgsz, self.s.device)
        return self._detector

    # ------------------------------------------------------------------
    def process(self, video_path: str | Path, control: ProcessingControl | None = None) -> ProcessingSummary:
        control = control or ProcessingControl()
        info = probe_video(video_path)  # raises VideoError for bad input BEFORE any session is created
        det = self.detector
        self.emit("status", "Hashing video...")
        fhash = file_sha256(info.path)
        sid = self.repo.create_session(
            info.path.name, fhash, duration_seconds=info.duration_s, frame_count=info.frame_count,
            fps=info.fps, width=info.width, height=info.height, processing_status="PROCESSING",
            processing_started_at=datetime.now(timezone.utc).replace(tzinfo=None),
            detector_version=det.version, tracker_version=TRACKER_VERSION,
            settings_json=json.dumps(self.s.to_dict()))
        summary = ProcessingSummary(sid, "PROCESSING")
        self.emit("session", {"session_id": sid, "info": info})
        writer: cv2.VideoWriter | None = None
        out_path: Path | None = None
        spans: dict[int, list[float]] = {}
        infer_ms_total = 0.0
        t0 = time.perf_counter()
        last_ts = 0.0
        try:
            stride = self.s.frame_stride
            eff_fps = info.fps / stride
            tracker = PersonTracker(self.s.person_conf, self.s.track_low_thresh,
                                    self.s.person_conf, self.s.track_buffer, self.s.match_thresh)
            engine = EventEngine(self.s, eff_fps)
            engine.frame_size = (info.width, info.height)
            if self.s.save_annotated_video:
                out_path = self.paths.processed / f"session_{sid}_annotated.mp4"
                writer = _open_writer(out_path, info.fps, (info.width, info.height))
                if writer is None:
                    log.warning("Could not open any video encoder; skipping annotated video")
                    out_path = None
            tracks: list[TrackedPerson] = []
            phones: list = []
            active: dict[int, list[Incident]] = {}
            pending: set[int] = set()
            last_preview = 0.0
            need_all = writer is not None
            for idx, ts, frame in iter_frames(info, stride, decode_skipped=need_all):
                control.wait_if_paused()
                if control.stopped:
                    summary.status = "STOPPED"
                    break
                last_ts = ts
                summary.frames_read = idx + 1
                if idx % stride == 0 and frame is not None:
                    ti = time.perf_counter()
                    dets = det.detect(frame, idx, ts)
                    infer_ms_total += (time.perf_counter() - ti) * 1000
                    summary.frames_inferred += 1
                    persons = [d for d in dets if d.label == PERSON]
                    phones = [d for d in dets if d.label == PHONE]
                    tracks = tracker.update(persons, frame)
                    assist = getattr(det, "detect_phones_in_crops", None)
                    if self.s.phone_crop_assist and assist is not None and tracks:
                        ci = time.perf_counter()
                        extra = assist(frame, [t.bbox for t in tracks], idx, ts, imgsz=self.s.phone_crop_imgsz)
                        infer_ms_total += (time.perf_counter() - ci) * 1000
                        phones = merge_detections(phones, extra)
                    for tr in tracks:
                        sp = spans.setdefault(tr.track_id, [ts, ts])
                        sp[1] = ts
                    out = engine.process(idx, ts, tracks, phones)
                    active, pending = out.active, engine.pending_tracks()
                    for inc in out.started:
                        self._on_incident_started(sid, inc, frame, tracks, phones, active, summary)
                    for inc in out.resolved:
                        self._persist_resolved(inc)
                if writer is not None and frame is not None:
                    ann = draw_frame(frame, tracks, phones, active, pending, ts)
                    draw_region(ann, self.s.region_polygon)
                    writer.write(ann)
                now = time.perf_counter()
                if frame is not None and now - last_preview >= self.preview_interval_s:
                    last_preview = now
                    img = ann if writer is not None else draw_frame(frame, tracks, phones, active, pending, ts)
                    if writer is None:
                        draw_region(img, self.s.region_polygon)
                    self.emit("frame", FrameUpdate(idx, ts, info.frame_count, summary.frames_read / max(now - t0, 1e-6),
                                                   img, len(tracks)))
            if summary.status == "PROCESSING":
                summary.status = "COMPLETED"
            for inc in engine.finalize(last_ts).resolved:
                self._persist_resolved(inc)
        except Exception as e:  # noqa: BLE001 - surfaced to the UI and persisted
            log.exception("processing failed")
            summary.status, summary.error = "FAILED", f"{type(e).__name__}: {e}"
        finally:
            if writer is not None:
                writer.release()
            summary.wall_seconds = time.perf_counter() - t0
            summary.n_tracks = len(spans)
            summary.avg_fps = summary.frames_read / summary.wall_seconds if summary.wall_seconds else 0.0
            summary.avg_infer_ms = infer_ms_total / summary.frames_inferred if summary.frames_inferred else 0.0
            summary.annotated_video = str(out_path) if out_path and out_path.exists() else None
            try:
                self.repo.save_tracks(sid, {k: (v[0], v[1]) for k, v in spans.items()})
                self.repo.update_session(
                    sid, processing_status=summary.status, annotated_video_path=summary.annotated_video,
                    processing_completed_at=datetime.now(timezone.utc).replace(tzinfo=None),
                    frame_count=summary.frames_read or info.frame_count)
            except Exception:  # noqa: BLE001
                log.exception("failed to persist session end")
            self.emit("done", summary)
        return summary

    # ------------------------------------------------------------------
    def _on_incident_started(self, sid: int, inc: Incident, frame: np.ndarray, tracks, phones, active,
                             summary: ProcessingSummary) -> None:
        ev = capture_evidence(self.paths.screenshots, sid, inc, frame, tracks, phones, active,
                              self.s.region_polygon)
        inc.db_id = self.repo.create_incident(
            sid, inc.track_id, inc.event_type, inc.started_ts, inc.started_frame, inc.confidence,
            inc.person_confidence, str(ev.annotated), str(ev.raw), inc.region_id,
            {"supporting_frames": inc.evidence_frames, "phone_bbox": inc.phone_bbox,
             "person_bbox": inc.person_bbox})
        summary.n_incidents += 1
        summary.incident_ids.append(inc.db_id)
        alert = Alert(inc.db_id, sid, inc.track_id, inc.event_type, inc.started_ts, inc.confidence,
                      inc.message(), str(ev.annotated))
        native_notify("ExamGuard AI - Review Required", inc.message(), self.s.notifications_enabled,
                      self.s.notification_sound)
        self.emit("alert", alert)

    def _persist_resolved(self, inc: Incident) -> None:
        if inc.db_id is None:
            return
        self.repo.update_incident(
            inc.db_id, last_detected_at=inc.ended_ts or inc.last_ts, lifecycle_state="RESOLVED",
            confidence=inc.confidence, person_confidence=inc.person_confidence,
            evidence={"supporting_frames": inc.evidence_frames, "phone_bbox": inc.phone_bbox,
                      "person_bbox": inc.person_bbox, "peak_confidence": inc.peak_conf})
        self.emit("incident_resolved", inc.db_id)
