"""Pipeline tests using a SCRIPTED FAKE detector (software behaviour only - not model accuracy)."""
import threading
import time
from pathlib import Path

import cv2
import pytest

from conftest import write_video
from examguard.config import Paths, Settings
from examguard.database.db import Database
from examguard.database.repository import Repository
from examguard.detection.detector import Detection
from examguard.video.processor import ProcessingControl, VideoProcessor
from examguard.video.reader import VideoError


class FakeDetector:
    version = "fake-detector-v0"

    def __init__(self, phone_frames=range(10, 40), person_box=(10, 20, 70, 110), phone_box=(30, 40, 42, 58),
                 delay=0.0):
        self.phone_frames, self.person_box, self.phone_box, self.delay = set(phone_frames), person_box, phone_box, delay
        self.calls = 0

    def detect(self, frame, idx, ts):
        self.calls += 1
        time.sleep(self.delay)
        out = [Detection("person", 0, 0.9, self.person_box, idx, ts, self.version)]
        if idx in self.phone_frames:
            out.append(Detection("cell phone", 67, 0.7, self.phone_box, idx, ts, self.version))
        return out


@pytest.fixture()
def env(tmp_path):
    paths = Paths(tmp_path / "home").ensure()
    repo = Repository(Database(paths.db))
    s = Settings(notifications_enabled=False, phone_min_support_frames=4, phone_window_frames=8, phone_min_duration_seconds=0.0,
                 resolve_after_seconds=0.5, save_annotated_video=True)
    return paths, repo, s


def run(env, video, det, control=None, settings=None, events=None):
    paths, repo, s = env
    ev = events if events is not None else []
    proc = VideoProcessor(settings or s, paths, repo, detector=det, emit=lambda k, p: ev.append((k, p)))
    return proc.process(video, control), repo, ev


def test_full_pipeline_creates_incident_evidence_db_alert(env, tmp_path):
    video = write_video(tmp_path / "v.mp4", n=80, fps=25)
    summ, repo, ev = run(env, video, FakeDetector())
    assert summ.status == "COMPLETED" and summ.frames_read == 80 and summ.n_incidents == 1
    inc = repo.list_incidents()[0]
    assert inc.event_type == "PHONE_VISIBLE_NEAR_TRACK" and inc.lifecycle_state == "RESOLVED"
    assert inc.review_status == "PENDING_REVIEW" and inc.confidence == pytest.approx(0.7)
    assert inc.session.detector_version.startswith("fake-detector-v0|imgsz=") and inc.session.fps == pytest.approx(25)
    assert "botsort" in inc.session.tracker_version
    assert Path(inc.screenshot_path).is_file() and Path(inc.raw_screenshot_path).is_file()
    assert Path(inc.screenshot_path).parent == env[0].screenshots
    kinds = [k for k, _ in ev]
    assert kinds.count("alert") == 1 and "frame" in kinds and kinds[-1] == "done" and "incident_resolved" in kinds
    alert = next(p for k, p in ev if k == "alert")
    assert alert.incident_db_id == inc.id and "Review Required" in alert.message
    assert f"Track #{inc.anonymous_track_id}" in alert.message
    assert repo.count_tracks(summ.session_id) == 1
    assert repo.get_session(summ.session_id).processing_status == "COMPLETED"


def test_continuous_event_is_one_incident(env, tmp_path):
    video = write_video(tmp_path / "v.mp4", n=120)
    summ, repo, _ = run(env, video, FakeDetector(phone_frames=range(5, 115)))
    assert summ.n_incidents == 1 and len(repo.list_incidents()) == 1


def test_normal_video_no_incidents(env, tmp_path):
    video = write_video(tmp_path / "v.mp4", n=60)
    summ, repo, ev = run(env, video, FakeDetector(phone_frames=[]))
    assert summ.n_incidents == 0 and repo.list_incidents() == [] and not [k for k, _ in ev if k == "alert"]


def test_annotated_video_is_playable_and_consistent(env, tmp_path):
    video = write_video(tmp_path / "v.mp4", n=50, fps=25)
    summ, *_ = run(env, video, FakeDetector())
    cap = cv2.VideoCapture(summ.annotated_video)
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 50
    assert (int(cap.get(3)), int(cap.get(4))) == (160, 120)
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(25, abs=0.5)
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == 50


def test_stride_reduces_inference(env, tmp_path):
    _, _, s = env
    s2 = Settings(**{**s.to_dict(), "frame_stride": 4, "save_annotated_video": False, "phone_min_support_frames": 2,
                     "phone_window_frames": 4})
    det = FakeDetector(phone_frames=range(0, 100))
    summ, *_ = run(env, write_video(tmp_path / "v.mp4", n=80), det, settings=s2)
    assert summ.frames_read == 80 and summ.frames_inferred == 20 == det.calls
    assert summ.annotated_video is None


def test_stop_cancels_cleanly_and_persists_state(env, tmp_path):
    video = write_video(tmp_path / "v.mp4", n=200)
    ctl = ProcessingControl()
    det = FakeDetector(phone_frames=range(0, 200), delay=0.01)
    threading.Timer(0.5, ctl.stop).start()
    summ, repo, ev = run(env, video, det, control=ctl)
    assert summ.status == "STOPPED" and summ.frames_read < 200
    assert repo.get_session(summ.session_id).processing_status == "STOPPED"
    assert ev[-1][0] == "done"
    for inc in repo.list_incidents():  # active incidents are closed on stop
        assert inc.lifecycle_state == "RESOLVED"


def test_pause_blocks_then_resume_completes(env, tmp_path):
    video = write_video(tmp_path / "v.mp4", n=60)
    ctl = ProcessingControl()
    det = FakeDetector(phone_frames=[], delay=0.01)
    ctl.pause()
    done = {}
    t = threading.Thread(target=lambda: done.update(r=run(env, video, det, control=ctl)[0]))
    t.start()
    time.sleep(0.6)
    assert det.calls <= 1 and t.is_alive()  # paused: no progress
    ctl.resume()
    t.join(timeout=30)
    assert done["r"].status == "COMPLETED" and det.calls == 60


def test_stop_while_paused_does_not_hang(env, tmp_path):
    ctl = ProcessingControl()
    ctl.pause()
    t = threading.Thread(target=lambda: run(env, write_video(tmp_path / "v.mp4", n=30),
                                            FakeDetector(phone_frames=[]), control=ctl))
    t.start()
    time.sleep(0.3)
    ctl.stop()
    t.join(timeout=10)
    assert not t.is_alive()


def test_detector_failure_marks_session_failed(env, tmp_path):
    class Boom(FakeDetector):
        def detect(self, *a):
            raise RuntimeError("gpu exploded")
    summ, repo, ev = run(env, write_video(tmp_path / "v.mp4", n=10), Boom())
    assert summ.status == "FAILED" and "gpu exploded" in summ.error
    assert repo.get_session(summ.session_id).processing_status == "FAILED" and ev[-1][0] == "done"


@pytest.mark.parametrize("content", [b"", b"garbage" * 50])
def test_bad_video_creates_no_session(env, tmp_path, content):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(content)
    with pytest.raises(VideoError):
        run(env, bad, FakeDetector())
    assert env[1].list_sessions() == []


def test_unsupported_extension(env, tmp_path):
    p = tmp_path / "a.gif"
    p.write_bytes(b"GIF89a")
    with pytest.raises(VideoError, match="Unsupported"):
        run(env, p, FakeDetector())


def test_region_exit_through_pipeline(env, tmp_path):
    _, _, s = env
    s2 = Settings(**{**s.to_dict(), "region_polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]],
                     "region_exit_min_seconds": 0.4, "save_annotated_video": False})

    class Walker(FakeDetector):
        def detect(self, frame, idx, ts):
            x = min(110, 10 + max(0, idx - 20) * 4)  # walks smoothly from the left half to the right half
            return [Detection("person", 0, 0.9, (x, 20, x + 40, 110), idx, ts, self.version)]
    _, repo, _ = run(env, write_video(tmp_path / "v.mp4", n=70), Walker(), settings=s2)
    incs = repo.list_incidents()
    assert len(incs) == 1 and incs[0].event_type == "MONITORED_REGION_EXIT" and incs[0].region_id == "region-1"


# ---- cross-track incident merging through the whole pipeline (scripted detector; software behaviour only) -----------
class SwitchDetector(FakeDetector):
    """Same person + same phone, but the person vanishes for 10 frames; with a short track buffer the tracker drops the old
    track and issues a NEW id when the person reappears (an ID switch: the old id disappears)."""

    def detect(self, frame, idx, ts):
        self.calls += 1
        out = []
        if not (30 <= idx < 40):
            out.append(Detection("person", 0, 0.9, (10, 20, 70, 110), idx, ts, self.version))
            if 5 <= idx < 75:
                out.append(Detection("cell phone", 67, 0.7, (30, 40, 42, 58), idx, ts, self.version))
        return out


def test_id_switch_yields_one_consistent_incident_with_review_intact(env, tmp_path):
    _, repo, s = env
    reviewed = []

    class Reviewer(list):  # emulate a reviewer clicking "needs more evidence" as soon as the alert arrives
        def append(self, item):
            kind, payload = item
            if kind == "alert" and not reviewed:
                repo.review(payload.incident_db_id, "NEEDS_MORE_EVIDENCE", "check the context")
                reviewed.append(payload.incident_db_id)
            super().append(item)

    s2 = Settings(**{**s.to_dict(), "track_buffer": 5, "save_annotated_video": False})  # lost tracks are dropped after 5 frames
    summ, repo, ev = run(env, write_video(tmp_path / "v.mp4", n=100, fps=25), SwitchDetector(), events=Reviewer(), settings=s2)
    incs = repo.list_incidents()
    assert len(incs) == 1 and summ.n_incidents == 1 and [k for k, _ in ev].count("alert") == 1   # no duplicate alert
    inc = repo.get_incident(incs[0].id)
    assert repo.count_tracks(summ.session_id) >= 2                                              # the tracker really did switch ids
    assert inc.lifecycle_state == "RESOLVED" and inc.last_detected_at >= 74 / 25 - 0.1           # extended across the switch
    assert inc.first_detected_at < 40 / 25                                                       # start time stays the FIRST confirmation
    assert inc.review_status == "NEEDS_MORE_EVIDENCE" and [r.reviewer_note for r in repo.reviews_for(inc.id)] == ["check the context"]
    assert inc.track_id is not None and inc.session.id == summ.session_id                        # references still consistent


def test_neighbour_with_visible_track_keeps_separate_incidents_in_pipeline(env, tmp_path):
    """Regression for the false-merge bug: A's phone ends, A stays visible, neighbour B shows a phone nearby -> 2 incidents."""
    class Neighbours(FakeDetector):
        def detect(self, frame, idx, ts):
            self.calls += 1
            out = [Detection("person", 0, 0.9, (5, 20, 55, 110), idx, ts, self.version),
                   Detection("person", 0, 0.9, (65, 20, 115, 110), idx, ts, self.version)]
            if 5 <= idx < 40:
                out.append(Detection("cell phone", 67, 0.7, (25, 45, 37, 63), idx, ts, self.version))      # A's phone
            if 80 <= idx < 115:
                out.append(Detection("cell phone", 67, 0.7, (85, 45, 97, 63), idx, ts, self.version))      # B's phone (~60 px away)
            return out
    _, repo, s = env
    s2 = Settings(**{**s.to_dict(), "phone_merge_gap_seconds": 6.0, "phone_merge_distance": 5.0, "save_annotated_video": False})
    summ, repo, _ = run(env, write_video(tmp_path / "v.mp4", n=130, fps=25), Neighbours(), settings=s2)
    incs = sorted(repo.list_incidents(), key=lambda i: i.first_detected_at)
    assert len(incs) == 2 and incs[0].anonymous_track_id != incs[1].anonymous_track_id
    assert incs[0].last_detected_at < incs[1].first_detected_at                                  # timestamps not stretched/merged


def test_id_switch_test_has_teeth_without_merge_it_duplicates(env, tmp_path):
    """Guard for the regression test above: with merging disabled the same scenario DOES create two incidents."""
    _, _, s = env
    s2 = Settings(**{**s.to_dict(), "track_buffer": 5, "save_annotated_video": False, "phone_merge_gap_seconds": 0})
    summ, repo, _ = run(env, write_video(tmp_path / "v.mp4", n=100, fps=25), SwitchDetector(), settings=s2)
    assert repo.count_tracks(summ.session_id) >= 2 and len(repo.list_incidents()) == 2
