"""Background processing thread. Talks to the UI ONLY through a thread-safe queue + latest-frame slot."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from examguard.config import Paths, Settings
from examguard.database.repository import Repository
from examguard.video.processor import ProcessingControl, VideoProcessor
from examguard.video.reader import VideoError


class ProcessingWorker:
    def __init__(self, settings: Settings, paths: Paths, repo: Repository, video: Path, detector_factory):
        self.settings, self.paths, self.repo, self.video = settings, paths, repo, Path(video)
        self.detector_factory = detector_factory
        self.control = ProcessingControl()
        self.messages: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._latest = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="examguard-worker", daemon=True)
        self.finished = threading.Event()

    # -- UI-side API -----------------------------------------------------
    def start(self) -> None: self._thread.start()
    def pause(self) -> None: self.control.pause()
    def resume(self) -> None: self.control.resume()
    def stop(self) -> None: self.control.stop()
    @property
    def alive(self) -> bool: return self._thread.is_alive()

    def join(self, timeout: float = 10.0) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def drain(self, limit: int = 200) -> list[tuple[str, object]]:
        out = []
        for _ in range(limit):
            try:
                out.append(self.messages.get_nowait())
            except queue.Empty:
                break
        return out

    def latest_frame(self):
        with self._lock:
            f, self._latest = self._latest, None
        return f

    # -- worker thread ---------------------------------------------------
    def _emit(self, kind: str, payload: object) -> None:
        if kind == "frame":  # never queue frames: keep only the newest (drops stale previews)
            with self._lock:
                self._latest = payload
        else:
            self.messages.put((kind, payload))

    def _run(self) -> None:
        try:
            self._emit("status", "Loading detection model...")
            det = self.detector_factory()
            proc = VideoProcessor(self.settings, self.paths, self.repo, detector=det, emit=self._emit)
            proc.process(self.video, self.control)
        except VideoError as e:
            self.messages.put(("error", str(e)))
            self.messages.put(("done", None))
        except Exception as e:  # noqa: BLE001
            self.messages.put(("error", f"{type(e).__name__}: {e}"))
            self.messages.put(("done", None))
        finally:
            self.finished.set()
