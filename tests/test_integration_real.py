"""Integration tests with the REAL YOLO model and REAL (openly licensed) videos.

Scenario 2 uses a clearly SYNTHETIC composite fixture (see scripts/make_synthetic_fixture.py): it checks
the real-model plumbing end to end, not real-world phone-use accuracy.
Tests skip (and are reported as skipped) when the weights/samples/fixture are not present locally.
"""
import json
from pathlib import Path

import cv2
import pytest

from conftest import FIXTURES, SAMPLES, WEIGHTS, needs_fixture, needs_model, needs_samples
from examguard.config import Paths, Settings
from examguard.database.db import Database
from examguard.database.repository import Repository
from examguard.detection.detector import PERSON, PHONE, YoloDetector
from examguard.tracking.tracker import PersonTracker
from examguard.video.processor import VideoProcessor
from examguard.video.reader import iter_frames, probe_video

pytestmark = [pytest.mark.integration, needs_model]
CLASSROOM = SAMPLES / "Student_studies_in_classroom_during_face_to_face_learning_in_Padang_Indonesia.webm"
STREET = SAMPLES / "Video_Codec_Test_pedestrian_area_1080p25.y4m.webm"
FIXTURE = FIXTURES / "synthetic_phone_pedestrian.mp4"


@pytest.fixture(scope="module")
def detector():
    return YoloDetector(WEIGHTS)


def process(tmp_path, video, detector, **kw):
    paths = Paths(tmp_path / "home").ensure()
    repo = Repository(Database(paths.db))
    s = Settings(notifications_enabled=False, save_annotated_video=kw.pop("save_video", False), **kw)
    events = []
    summ = VideoProcessor(s, paths, repo, detector=detector, emit=lambda k, p: events.append((k, p))).process(video)
    return summ, repo, paths, events


def test_real_model_loads_and_detects_people():
    from ultralytics.utils import ASSETS
    det = YoloDetector(WEIGHTS)
    img = cv2.imread(str(ASSETS / "bus.jpg"))
    dets = det.detect(img, 0, 0.0)
    persons = [d for d in dets if d.label == PERSON]
    assert len(persons) >= 3 and det.device in ("mps", "cpu")
    assert all(d.validate(img.shape[1], img.shape[0]) and 0 < d.confidence <= 1 and d.detector_version == det.version for d in dets)
    assert "sha256" in det.version and "ultralytics" in det.version


@needs_samples
def test_scenario1_normal_classroom_no_alerts(tmp_path, detector):
    summ, repo, _, ev = process(tmp_path, CLASSROOM, detector)
    assert summ.status == "COMPLETED" and summ.frames_read >= 600
    assert summ.n_tracks >= 8                      # seated students are tracked
    assert summ.n_incidents == 0 and not [k for k, _ in ev if k == "alert"]


@needs_samples
def test_scenario1b_normal_street_no_alerts(tmp_path, detector):
    summ, repo, *_ = process(tmp_path, STREET, detector)
    assert summ.status == "COMPLETED" and summ.frames_read == 375 and summ.n_incidents == 0


@needs_samples
def test_scenario3_multiple_people_unique_ids_per_frame(detector):
    info = probe_video(STREET)
    tr = PersonTracker(0.5, 0.1, 0.5, 60, 0.8)
    max_conc = 0
    for idx, ts, fr in iter_frames(info):
        if idx > 150:
            break
        res = tr.update([d for d in detector.detect(fr, idx, ts) if d.label == PERSON], fr)
        ids = [t.track_id for t in res]
        assert len(ids) == len(set(ids))           # no two people share a track ID in one frame
        max_conc = max(max_conc, len(ids))
    assert max_conc >= 5


@needs_samples
def test_timestamps_and_metadata_on_real_container(tmp_path):
    info = probe_video(STREET)                     # container claims 1000 fps / 15125 frames; reality is 25 fps / 375
    assert info.fps == pytest.approx(25.0, abs=0.1) and info.frame_count == 375
    ts = [t for _, t, _ in iter_frames(info, decode_skipped=True)]
    assert len(ts) == 375 and ts[-1] == pytest.approx(14.96, abs=0.05)


@needs_fixture
def test_scenario2_synthetic_phone_event_full_chain(tmp_path, detector):
    meta = json.loads(Path(str(FIXTURE) + ".fixture.json").read_text())
    assert meta["SYNTHETIC_COMPOSITE"] is True
    summ, repo, paths, ev = process(tmp_path, FIXTURE, detector, save_video=True)
    assert summ.status == "COMPLETED" and summ.n_incidents >= 1
    incs = repo.list_incidents()
    inc = incs[-1]
    assert inc.event_type == "PHONE_VISIBLE_NEAR_TRACK" and inc.review_status == "PENDING_REVIEW"
    t0, t1 = meta["phone_visible_from_s"], meta["phone_visible_to_s"]
    assert t0 <= inc.first_detected_at <= t1 + 1.0          # alert falls inside the staged window
    shot = cv2.imread(inc.screenshot_path)
    assert shot is not None and shot.shape[1] == 1920       # real annotated evidence frame
    assert cv2.imread(inc.raw_screenshot_path).shape[:2] == (1080, 1920)
    assert [k for k, _ in ev].count("alert") == summ.n_incidents
    # scenario 6: annotated video playable, same frame count / size as the source
    cap = cv2.VideoCapture(summ.annotated_video)
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 375
    assert (int(cap.get(3)), int(cap.get(4))) == (1920, 1080)
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == 375
