"""Application settings with explicit validation.

All thresholds here are *engineering defaults*, not scientifically calibrated
values. See docs/EVENT_RULES.md for how each one affects alerts.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

APP_NAME = "ExamGuard AI"
APP_SUBTITLE = "Intelligent Exam Monitoring & Suspicious Behavior Analysis Platform"
DISCLAIMER = (
    "Automated alerts identify observable events requiring human review. "
    "They are not proof of academic misconduct."
)

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")
MAX_VIDEO_BYTES = 4 * 1024**3  # 4 GiB input-size limit


class ConfigError(ValueError):
    """Raised when a setting is out of its valid range."""


def effective_imgsz(width: int, height: int, requested: int = 0, cap: int = 1920) -> int:
    """YOLO input size. 0 = auto: native long side (so small objects are not shrunk away), capped, multiple of 32.

    Measured on real labelled images: running at native resolution more than doubled small-phone AP50 compared
    with a fixed 640 input on 1080p-like frames.
    """
    if requested:
        return int(requested)
    long_side = max(width, height)
    return int(min(cap, max(320, -(-long_side // 32) * 32)))


def default_data_dir() -> Path:
    """Per-user runtime directory, kept outside the repository/app bundle."""
    override = os.environ.get("EXAMGUARD_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ExamGuardAI"
    if sys.platform.startswith("win"):
        return Path(os.environ.get("APPDATA", Path.home())) / "ExamGuardAI"
    return Path.home() / ".local" / "share" / "examguard-ai"


@dataclass
class Settings:
    # detection
    # Persons are always detected down to a low floor (detector.PERSON_TRACKER_FLOOR) so ByteTrack can use
    # low-score boxes for its second association; `person_conf` is ByteTrack's HIGH threshold.
    person_conf: float = 0.5              # tracker high-confidence threshold (also gates new tracks)
    phone_conf: float = 0.25              # phones are small -> lower floor
    nms_iou: float = 0.5
    # Second stage: re-detect phones inside each tracked person's crop and merge with the full-frame result. On tiny phones
    # in small frames (synthetic held-out fixtures) it is what makes detection possible at all (frame recall 0% -> ~55%);
    # on real labelled images it trades some precision for recall. It costs ~3-5x inference time. See docs/EVALUATION.md.
    phone_crop_assist: bool = True
    phone_crop_imgsz: int = 640           # crop input size (640 beat 416 and 320 in the benchmark)
    imgsz: int = 0                        # YOLO input size; 0 = auto: native video size, capped at 1920 (see effective_imgsz)
    tracker_kind: str = "botsort"         # botsort (no appearance/GMC) | bytetrack
    device: str = "auto"                  # auto | mps | cpu
    weights_path: str = "models/yolo11n.pt"
    # frame sampling: run inference on every Nth frame (tracker is fed only those)
    frame_stride: int = 1
    # tracking (ByteTrack)
    track_low_thresh: float = 0.1
    track_buffer: int = 60                # frames a lost track is kept alive (at the processed frame rate)
    match_thresh: float = 0.8
    # event A: phone near person
    # A phone seen again on a DIFFERENT track soon after (an ID switch) is merged into the earlier incident instead of
    # raising a duplicate alert: within phone_merge_gap_seconds and within phone_merge_distance phone-sizes (0 = off).
    phone_merge_gap_seconds: float = 6.0
    phone_merge_distance: float = 2.5
    phone_min_support_frames: int = 5     # frames with association inside window
    # The supporting detections must also span at least this many SECONDS (0 = off). A frame count alone confirms a 0.1 s
    # blip on 60 fps video (found on a real clip: a face mask flagged as a phone). The window is widened for fast video.
    phone_min_duration_seconds: float = 0.5
    phone_window_frames: int = 15
    phone_assoc_min_overlap: float = 0.15  # fraction of phone box inside expanded person box
    phone_assoc_expand: float = 0.15       # expand person box by this fraction each side
    # event B: monitored region exit
    region_polygon: list[list[float]] = field(default_factory=list)  # normalized [[x,y],...]; empty = disabled
    region_exit_min_seconds: float = 1.5
    region_id: str = "region-1"
    # lifecycle
    event_cooldown_seconds: float = 10.0   # after resolve, ignore same (track,event) re-trigger
    resolve_after_seconds: float = 2.0     # condition must be absent this long to resolve
    # storage
    output_dir: str = ""
    retention_days: int = 30
    notifications_enabled: bool = True
    notification_sound: bool = False
    save_annotated_video: bool = True

    def validate(self) -> "Settings":
        def rng(name: str, lo: float, hi: float) -> None:
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not (lo <= v <= hi):
                raise ConfigError(f"{name} must be between {lo} and {hi}, got {v!r}")

        rng("person_conf", 0.05, 0.99)
        rng("phone_conf", 0.01, 0.99)
        rng("nms_iou", 0.05, 0.95)
        if self.imgsz != 0:
            rng("imgsz", 160, 1920)
        if self.tracker_kind not in ("botsort", "bytetrack"):
            raise ConfigError(f"tracker_kind must be botsort|bytetrack, got {self.tracker_kind!r}")
        rng("phone_crop_imgsz", 160, 1280)
        if not isinstance(self.phone_crop_assist, bool):
            raise ConfigError("phone_crop_assist must be true or false")
        rng("frame_stride", 1, 30)
        rng("track_low_thresh", 0.01, 0.98)
        rng("track_buffer", 1, 300)
        rng("match_thresh", 0.1, 1.0)
        rng("phone_min_duration_seconds", 0.0, 30.0)
        rng("phone_merge_gap_seconds", 0.0, 60.0)
        rng("phone_merge_distance", 0.5, 10.0)
        rng("phone_min_support_frames", 1, 300)
        rng("phone_window_frames", 1, 600)
        rng("phone_assoc_min_overlap", 0.01, 1.0)
        rng("phone_assoc_expand", 0.0, 1.0)
        rng("region_exit_min_seconds", 0.0, 600)
        rng("event_cooldown_seconds", 0.0, 3600)
        rng("resolve_after_seconds", 0.0, 600)
        rng("retention_days", 0, 3650)
        if self.track_low_thresh >= self.person_conf:
            raise ConfigError("track_low_thresh must be below person_conf")
        if self.phone_min_support_frames > self.phone_window_frames:
            raise ConfigError("phone_min_support_frames cannot exceed phone_window_frames")
        if self.device not in ("auto", "mps", "cpu"):
            raise ConfigError(f"device must be auto|mps|cpu, got {self.device!r}")
        if self.region_polygon:
            if len(self.region_polygon) < 3:
                raise ConfigError("region_polygon needs at least 3 points")
            for p in self.region_polygon:
                if len(p) != 2 or not all(0.0 <= float(c) <= 1.0 for c in p):
                    raise ConfigError("region_polygon points must be normalized [x, y] in 0..1")
        return self

    # --- persistence -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown settings: {sorted(unknown)}")
        return cls(**data).validate()

    @classmethod
    def load(cls, path: Path) -> "Settings":
        if not path.exists():
            return cls().validate()
        return cls.from_dict(json.loads(path.read_text()))

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)


@dataclass(frozen=True)
class Paths:
    """Resolved runtime locations. Nothing here depends on the repo path."""
    root: Path

    @property
    def db(self) -> Path: return self.root / "examguard.db"
    @property
    def screenshots(self) -> Path: return self.root / "incidents" / "screenshots"
    @property
    def videos(self) -> Path: return self.root / "videos"
    @property
    def processed(self) -> Path: return self.root / "processed"
    @property
    def logs(self) -> Path: return self.root / "logs"
    @property
    def exports(self) -> Path: return self.root / "exports"
    @property
    def settings_file(self) -> Path: return self.root / "settings.json"

    def ensure(self) -> "Paths":
        for p in (self.screenshots, self.videos, self.processed, self.logs, self.exports):
            p.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, 0o700)  # evidence is sensitive: owner-only
        except OSError:
            pass
        return self
