"""Event engine tests with deterministic synthetic detections (software behaviour only)."""
import pytest

from examguard.config import Settings
from examguard.detection.detector import Detection
from examguard.events.engine import PHONE_EVENT, REGION_EVENT, EventEngine
from examguard.tracking.tracker import TrackedPerson

FPS = 10.0


def trk(tid, box, fi, conf=0.9):
    return TrackedPerson(tid, box, conf, fi, fi / FPS, None)


def phone(box, fi, conf=0.6):
    return Detection("cell phone", 67, conf, box, fi, fi / FPS, "t")


def mk(**kw):
    base = dict(phone_min_support_frames=3, phone_window_frames=5, resolve_after_seconds=1.0,
                event_cooldown_seconds=5.0, region_exit_min_seconds=1.0)
    base.update(kw)
    return EventEngine(Settings(**base).validate(), fps=FPS)


A = (100, 100, 200, 300)
PHONE_ON_A = (140, 180, 160, 215)


def test_phone_needs_temporal_support():
    e = mk()
    started = []
    for i in range(2):  # only 2 supporting frames < 3
        started += e.process(i, i / FPS, [trk(1, A, i)], [phone(PHONE_ON_A, i)]).started
    assert started == []


def test_phone_event_starts_once_and_marks_track_active():
    e = mk()
    starts, last = [], None
    for i in range(30):
        last = e.process(i, i / FPS, [trk(1, A, i)], [phone(PHONE_ON_A, i)])
        starts += last.started
    assert len(starts) == 1
    inc = starts[0]
    assert inc.event_type == PHONE_EVENT and inc.track_id == 1
    assert 1 in last.active and last.active[1][0] is inc
    assert inc.conf_n >= 3 and 0 < inc.confidence <= 1


def test_phone_resolves_when_absent_then_cooldown_suppresses_retrigger():
    e = mk()
    starts, resolved = [], []
    for i in range(10):
        o = e.process(i, i / FPS, [trk(1, A, i)], [phone(PHONE_ON_A, i)]); starts += o.started
    for i in range(10, 25):  # phone gone 1.5 s -> resolve
        o = e.process(i, i / FPS, [trk(1, A, i)], []); resolved += o.resolved
    assert len(resolved) == 1 and resolved[0].state == "RESOLVED"
    last = None
    for i in range(25, 30):  # reappears inside cooldown (5 s) -> no new incident
        last = e.process(i, i / FPS, [trk(1, A, i)], [phone(PHONE_ON_A, i)]); starts += last.started
    assert len(starts) == 1
    assert 1 not in last.active  # box returns to normal (green)


def test_retrigger_after_cooldown():
    e = mk(event_cooldown_seconds=1.0)
    starts = []
    for i in range(10):
        starts += e.process(i, i / FPS, [trk(1, A, i)], [phone(PHONE_ON_A, i)]).started
    for i in range(10, 40):
        e.process(i, i / FPS, [trk(1, A, i)], [])
    for i in range(40, 50):
        starts += e.process(i, i / FPS, [trk(1, A, i)], [phone(PHONE_ON_A, i)]).started
    assert len(starts) == 2


def test_phone_not_attributed_when_far_from_person():
    e = mk()
    far = (500, 100, 520, 130)
    starts = []
    for i in range(20):
        starts += e.process(i, i / FPS, [trk(1, A, i)], [phone(far, i)]).started
    assert starts == []


def test_ambiguous_phone_between_two_people_is_skipped():
    e = mk()
    p1, p2 = (100, 100, 200, 300), (190, 100, 290, 300)
    between = (185, 180, 205, 215)  # overlaps both nearly equally
    starts = []
    for i in range(20):
        starts += e.process(i, i / FPS, [trk(1, p1, i), trk(2, p2, i)], [phone(between, i)]).started
    assert starts == []


def test_phone_attributed_to_correct_person_among_many():
    e = mk()
    p1, p2, p3 = (100, 100, 200, 300), (400, 100, 500, 300), (700, 100, 800, 300)
    ph = (440, 180, 460, 215)  # on person 2 only
    starts = []
    for i in range(15):
        starts += e.process(i, i / FPS, [trk(1, p1, i), trk(2, p2, i), trk(3, p3, i)], [phone(ph, i)]).started
    assert [s.track_id for s in starts] == [2]


def test_finalize_closes_active_incident():
    e = mk()
    for i in range(10):
        e.process(i, i / FPS, [trk(1, A, i)], [phone(PHONE_ON_A, i)])
    out = e.finalize(1.0)
    assert len(out.resolved) == 1 and out.resolved[0].ended_ts is not None


# ---- region exit ---------------------------------------------------------
REGION = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]  # left half of a 1000x500 frame


def region_engine(**kw):
    e = mk(region_polygon=REGION, **kw)
    e.frame_size = (1000, 500)
    return e


def test_region_exit_requires_prior_presence_and_duration():
    e = region_engine()
    inside, outside = (100, 100, 200, 300), (700, 100, 800, 300)
    starts = []
    for i in range(5):
        starts += e.process(i, i / FPS, [trk(1, inside, i)], []).started
    for i in range(5, 9):  # outside 0.4 s < 1.0 s
        starts += e.process(i, i / FPS, [trk(1, outside, i)], []).started
    assert starts == []
    for i in range(9, 25):
        starts += e.process(i, i / FPS, [trk(1, outside, i)], []).started
    assert len(starts) == 1 and starts[0].event_type == REGION_EVENT
    assert starts[0].region_id == "region-1"


def test_track_that_never_was_inside_does_not_alert():
    e = region_engine()
    starts = []
    for i in range(30):
        starts += e.process(i, i / FPS, [trk(1, (700, 100, 800, 300), i)], []).started
    assert starts == []


def test_brief_detector_dropout_does_not_alert():
    e = region_engine()
    inside = (100, 100, 200, 300)
    starts = []
    for i in range(40):
        tracks = [] if 10 <= i < 14 else [trk(1, inside, i)]
        starts += e.process(i, i / FPS, tracks, []).started
    assert starts == []


def test_region_incident_resolves_on_return():
    e = region_engine()
    inside, outside = (100, 100, 200, 300), (700, 100, 800, 300)
    resolved = []
    for i in range(3):
        e.process(i, i / FPS, [trk(1, inside, i)], [])
    for i in range(3, 20):
        e.process(i, i / FPS, [trk(1, outside, i)], [])
    for i in range(20, 40):
        resolved += e.process(i, i / FPS, [trk(1, inside, i)], []).resolved
    assert len(resolved) == 1


@pytest.mark.parametrize("bad", [
    dict(person_conf=0), dict(phone_min_support_frames=20, phone_window_frames=5),
    dict(track_low_thresh=0.9, person_conf=0.5), dict(region_polygon=[[0, 0], [1, 1]]),
    dict(device="cuda"), dict(frame_stride=0),
])
def test_settings_validation_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        Settings(**bad).validate()
