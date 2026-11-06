"""Controller / worker / GUI tests. Detector is a scripted fake (software behaviour, not model accuracy).
GUI tests need a display and are skipped (and reported as skipped) when none is available."""
import time
import tkinter as tk
from pathlib import Path

import pytest

from conftest import write_video
from examguard.config import Settings
from examguard.ui.controller import AppController
from test_processor import FakeDetector


def wait_done(ctl, timeout=60):
    kinds, t0 = [], time.time()
    while time.time() - t0 < timeout:
        for k, p in ctl.worker.drain():
            kinds.append((k, p))
        if ctl.worker.finished.is_set() and ctl.worker.messages.empty():
            return kinds
        time.sleep(0.02)
    raise AssertionError("worker did not finish")


@pytest.fixture()
def ctl(tmp_path):
    c = AppController(tmp_path / "home")
    c.settings = Settings(**{**c.settings.to_dict(), "notifications_enabled": False, "phone_min_support_frames": 4, "phone_min_duration_seconds": 0.0,
                             "phone_window_frames": 8, "resolve_after_seconds": 0.5})
    c._get_detector = lambda: FakeDetector()
    yield c
    c.shutdown()


def test_worker_to_ui_message_contract(ctl, tmp_path):
    ctl.start_processing(write_video(tmp_path / "v.mp4", n=80))
    kinds = wait_done(ctl)
    names = [k for k, _ in kinds]
    assert names[0] == "status" and names[-1] == "done"
    assert names.count("alert") == 1 and "session" in names
    assert not any(k == "frame" for k in names)  # frames use the latest-frame slot, never the queue
    assert ctl.repo.stats()["incidents"] == 1


def test_latest_frame_slot_overwrites(ctl, tmp_path):
    ctl.start_processing(write_video(tmp_path / "v.mp4", n=60))
    seen = None
    t0 = time.time()
    while time.time() - t0 < 30 and not ctl.worker.finished.is_set():
        f = ctl.worker.latest_frame()
        seen = f or seen
        time.sleep(0.01)
    assert seen is not None and seen.image.shape == (120, 160, 3) and seen.image.dtype.name == "uint8"


def test_second_start_rejected_while_running(ctl, tmp_path):
    ctl._get_detector = lambda: FakeDetector(phone_frames=[], delay=0.01)
    v = write_video(tmp_path / "v.mp4", n=150)
    ctl.start_processing(v)
    with pytest.raises(RuntimeError):
        ctl.start_processing(v)
    assert ctl.stop_processing()


def test_shutdown_stops_worker_and_closes_db(ctl, tmp_path):
    ctl._get_detector = lambda: FakeDetector(phone_frames=[], delay=0.02)
    w = ctl.start_processing(write_video(tmp_path / "v.mp4", n=400))
    time.sleep(0.5)
    assert w.alive
    assert ctl.shutdown() is True and not w.alive
    assert AppController(tmp_path / "home").repo.list_sessions()[0].processing_status == "STOPPED"


def test_bad_video_reports_error_message_not_crash(ctl, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"nope" * 100)
    ctl.start_processing(bad)
    kinds = [k for k, _ in wait_done(ctl)]
    assert "error" in kinds and kinds[-1] == "done"
    assert ctl.repo.list_sessions() == []


def test_missing_model_is_reported_cleanly(tmp_path, monkeypatch):
    c = AppController(tmp_path / "h")
    monkeypatch.setattr("examguard.ui.controller.find_weights", lambda p: None)
    c.start_processing(write_video(tmp_path / "v.mp4", n=5))
    kinds = wait_done(c)
    assert any(k == "error" and "weights" in p for k, p in kinds)
    c.shutdown()


def test_settings_persist_across_controllers(tmp_path):
    c = AppController(tmp_path / "h")
    s = Settings(**{**c.settings.to_dict(), "imgsz": 800, "retention_days": 7})
    c.save_settings(s)
    c.shutdown()
    assert AppController(tmp_path / "h").settings.imgsz == 800


def test_invalid_settings_not_saved(tmp_path):
    c = AppController(tmp_path / "h")
    with pytest.raises(ValueError):
        c.save_settings(Settings(imgsz=10))
    assert not c.settings_path.exists()


def test_retention_deletes_old_evidence_only_inside_root(tmp_path):
    from datetime import timedelta
    from examguard.database.models import utcnow
    c = AppController(tmp_path / "h")
    inside = c.paths.screenshots / "old.jpg"
    inside.write_bytes(b"x")
    outside = tmp_path / "precious.jpg"
    outside.write_bytes(b"x")
    sid = c.repo.create_session("a.mp4", "a" * 64)
    c.repo.create_incident(sid, 1, "PHONE_VISIBLE_NEAR_TRACK", 1.0, 5, 0.5, None, str(inside), str(outside))
    c.repo.update_session(sid, created_at=utcnow() - timedelta(days=90))
    assert c.apply_retention() == 1
    assert not inside.exists() and outside.exists()  # a tampered DB path cannot delete outside the data root
    c.shutdown()


# ---------------------------------------------------------------------------- GUI
# Each GUI scenario runs in its own process: Tk 9.0.4/macOS crashes when a second Tk root is created in a
# process that already destroyed one (see tests/gui_scenarios.py). The pytest process never creates a root.
import json
import subprocess
import sys

SCENARIOS = Path(__file__).parent / "gui_scenarios.py"


def _display_ok():
    r = subprocess.run([sys.executable, "-c", "import tkinter; tkinter.Tk().destroy()"], capture_output=True, timeout=30)
    return r.returncode == 0


gui = pytest.mark.skipif(not _display_ok(), reason="no display available for Tk")


def run_scenario(name, home, *args):
    r = subprocess.run([sys.executable, str(SCENARIOS), name, str(home), *map(str, args)], capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, f"{name} failed:\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}"
    line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")][-1]
    return json.loads(line[7:])


@gui
def test_gui_end_to_end_review_then_real_restart(tmp_path):
    home = tmp_path / "home"
    res = run_scenario("e2e", home)
    run_scenario("restart", home, res["incident_id"])        # a genuinely separate process = application restart


@gui
def test_gui_validation_missing_evidence_settings_and_region(tmp_path):
    run_scenario("misc", tmp_path / "home2")
