"""UI-independent application controller (testable without a display)."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from examguard.config import Paths, Settings, default_data_dir
from examguard.database.db import Database
from examguard.database.repository import Repository
from examguard.detection.detector import PERSON_TRACKER_FLOOR, YoloDetector
from examguard.evidence.capture import safe_delete
from examguard.ui.worker import ProcessingWorker
from examguard.utils.resources import find_weights

log = logging.getLogger(__name__)


class ModelMissing(RuntimeError):
    pass


class AppController:
    def __init__(self, home: Path | None = None):
        self.bootstrap = Path(home) if home else default_data_dir()
        self.bootstrap.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.bootstrap / "settings.json"
        self.settings = Settings.load(self.settings_path)
        root = Path(self.settings.output_dir).expanduser() if self.settings.output_dir else self.bootstrap
        self.paths = Paths(root).ensure()
        self.repo = Repository(Database(self.paths.db))
        self._detector: YoloDetector | None = None
        self._det_lock = threading.Lock()
        self.worker: ProcessingWorker | None = None
        self.current_session_id: int | None = None

    # -- settings ----------------------------------------------------------
    def save_settings(self, s: Settings) -> None:
        s.validate()
        s.save(self.settings_path)
        self.settings = s

    # -- detector ----------------------------------------------------------
    def _get_detector(self) -> YoloDetector:
        with self._det_lock:
            w = find_weights(self.settings.weights_path)
            if w is None:
                raise ModelMissing("Model weights not found. Run `python scripts/fetch_model.py` "
                                   "or use Settings > Download model.")
            s = self.settings
            if self._detector is None or Path(self._detector.weights) != w or self._detector.device != self._wanted_device():
                self._detector = YoloDetector(w, PERSON_TRACKER_FLOOR, s.phone_conf, s.nms_iou, s.imgsz, s.device)
            d = self._detector  # thresholds may have changed in Settings since last run
            d.phone_conf, d.nms_iou, d.imgsz = s.phone_conf, s.nms_iou, s.imgsz
            return d

    def _wanted_device(self) -> str:
        from examguard.detection.detector import resolve_device
        return resolve_device(self.settings.device)

    # -- processing ----------------------------------------------------------
    @property
    def running(self) -> bool:
        return bool(self.worker and self.worker.alive)

    def start_processing(self, video: Path) -> ProcessingWorker:
        if self.running:
            raise RuntimeError("A video is already being processed")
        self.worker = ProcessingWorker(Settings(**self.settings.to_dict()).validate(), self.paths, self.repo,
                                       Path(video), self._get_detector)
        self.worker.start()
        return self.worker

    def stop_processing(self, timeout: float = 15.0) -> bool:
        if self.worker and self.worker.alive:
            self.worker.stop()
            return self.worker.join(timeout)
        return True

    # -- retention -------------------------------------------------------------
    def apply_retention(self) -> int:
        days = self.settings.retention_days
        if days <= 0:
            return 0
        return safe_delete(self.repo.delete_older_than(days), self.paths.root)

    def delete_session(self, session_id: int) -> int:
        return safe_delete(self.repo.delete_session(session_id), self.paths.root)

    # -- lifecycle ---------------------------------------------------------------
    def shutdown(self) -> bool:
        ok = self.stop_processing()
        try:
            self.repo.db.close()
        except Exception:  # noqa: BLE001
            log.exception("db close failed")
        return ok
