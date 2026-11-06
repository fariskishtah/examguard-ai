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
    base = dict(phone_min_support_frames=3, phone_window_frames=5, phone_min_duration_seconds=0.0, resolve_after_seconds=1.0,
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


# ---- resolution / tracker settings (new in the improvement pass) ----------------------------------
from examguard.config import effective_imgsz  # noqa: E402


@pytest.mark.parametrize("w,h,req,expected", [
    (1920, 1080, 0, 1920), (3840, 2160, 0, 1920), (1280, 720, 0, 1280), (640, 480, 0, 640), (768, 576, 0, 768),
    (160, 120, 0, 320), (1920, 1080, 960, 960), (1000, 500, 0, 1024)])
def test_effective_imgsz_auto_is_native_capped_and_multiple_of_32(w, h, req, expected):
    out = effective_imgsz(w, h, req)
    assert out == expected and (req or out % 32 == 0)


def test_new_defaults_are_the_evidence_based_ones():
    s = Settings().validate()
    assert s.imgsz == 0 and s.phone_crop_assist is True and s.phone_crop_imgsz == 640 and s.tracker_kind == "botsort"


@pytest.mark.parametrize("bad", [dict(imgsz=100), dict(imgsz=5000), dict(tracker_kind="deepsort"), dict(phone_crop_imgsz=50)])
def test_new_settings_validation(bad):
    with pytest.raises(ValueError):
        Settings(**bad).validate()


def test_imgsz_zero_means_auto_and_is_valid():
    assert Settings(imgsz=0).validate().imgsz == 0


# ---- merging a phone incident across a track-ID switch (software-level, synthetic boxes) ----------------------
def _phone_at(cx, cy, fi, w=20, h=30):
    return phone((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), fi)


def _person_around(cx, cy, tid, fi):
    return trk(tid, (cx - 50, cy - 130, cx + 50, cy + 130), fi)


def test_id_switch_is_one_incident_and_box_follows_new_track():
    e = mk()
    started, last = [], None
    for i in range(0, 20):  # track 1 carries the phone, moving right ~3 px/frame
        last = e.process(i, i / FPS, [_person_around(200 + 3 * i, 300, 1, i)], [_phone_at(200 + 3 * i, 280, i)])
        started += last.started
    for i in range(20, 50):  # ID switch: same person/phone, now track 2
        last = e.process(i, i / FPS, [_person_around(200 + 3 * i, 300, 2, i)], [_phone_at(200 + 3 * i, 280, i)])
        started += last.started
    assert len(started) == 1
    assert 2 in last.active and last.active[2][0] is started[0]  # RED box follows the person's current track


def test_resolved_incident_is_reopened_not_duplicated():
    e = mk(resolve_after_seconds=0.5)
    started, reopened, resolved = [], [], []
    for i in range(10):
        started += e.process(i, i / FPS, [_person_around(300, 300, 1, i)], [_phone_at(300, 280, i)]).started
    for i in range(10, 25):  # nothing for 1.5 s -> resolves
        resolved += e.process(i, i / FPS, [], []).resolved
    assert len(resolved) == 1
    for i in range(25, 40):  # same phone place, new track id, 2.5 s later (< 6 s gap)
        o = e.process(i, i / FPS, [_person_around(310, 300, 7, i)], [_phone_at(310, 280, i)])
        started += o.started; reopened += o.reopened
    assert len(started) == 1 and len(reopened) == 1 and reopened[0] is started[0] and reopened[0].state == "ACTIVE"


def test_different_place_is_a_separate_incident():
    e = mk()
    started = []
    for i in range(10):
        started += e.process(i, i / FPS, [_person_around(200, 300, 1, i), _person_around(900, 300, 2, i)],
                             [_phone_at(200, 280, i), _phone_at(900, 280, i)]).started
    assert sorted(s.track_id for s in started) == [1, 2]  # two people with phones far apart: two incidents


def test_gap_longer_than_limit_starts_new_incident():
    e = mk(resolve_after_seconds=0.5, phone_merge_gap_seconds=2.0, event_cooldown_seconds=0.0)
    started = []
    for i in range(10):
        started += e.process(i, i / FPS, [_person_around(300, 300, 1, i)], [_phone_at(300, 280, i)]).started
    for i in range(10, 60):
        e.process(i, i / FPS, [], [])
    for i in range(60, 75):  # 5 s later: beyond the 2 s merge gap
        started += e.process(i, i / FPS, [_person_around(300, 300, 9, i)], [_phone_at(300, 280, i)]).started
    assert len(started) == 2


def test_merge_can_be_disabled():
    e = mk(phone_merge_gap_seconds=0.0)
    started = []
    for i in range(20):
        started += e.process(i, i / FPS, [_person_around(200 + 3 * i, 300, 1, i)], [_phone_at(200 + 3 * i, 280, i)]).started
    for i in range(20, 50):
        started += e.process(i, i / FPS, [_person_around(200 + 3 * i, 300, 2, i)], [_phone_at(200 + 3 * i, 280, i)]).started
    assert len(started) == 2


def test_neighbour_with_visible_track_is_never_merged():
    """Two people side by side, both visible: B's phone near A's old phone must be a NEW incident, not a merge."""
    e = mk(resolve_after_seconds=0.5)
    started = []
    for i in range(10):  # A (track 1) has a phone
        started += e.process(i, i / FPS, [_person_around(300, 300, 1, i), _person_around(380, 300, 2, i)], [_phone_at(300, 280, i)]).started
    for i in range(10, 25):  # A's phone disappears, track 1 STAYS visible
        e.process(i, i / FPS, [_person_around(300, 300, 1, i), _person_around(380, 300, 2, i)], [])
    for i in range(25, 40):  # B (track 2, 80 px away = ~2.7 phone widths... within merge distance) now shows a phone
        started += e.process(i, i / FPS, [_person_around(300, 300, 1, i), _person_around(380, 300, 2, i)], [_phone_at(380, 280, i, 40, 60)]).started
    assert sorted(s.track_id for s in started) == [1, 2]


# ---- minimum event duration (real false alert: a 0.13 s blip on 60 fps video was confirmed by a frame count alone) ----
def _run_blip(fps, n_hit_frames, min_dur, support=5, window=15):
    e = EventEngine(Settings(phone_min_support_frames=support, phone_window_frames=window, phone_min_duration_seconds=min_dur,
                             resolve_after_seconds=1.0).validate(), fps=fps)
    started = []
    for i in range(int(fps * 4)):
        hit = i < n_hit_frames
        started += e.process(i, i / fps, [trk(1, A, i)], [phone(PHONE_ON_A, i)] if hit else []).started
    return started


def test_short_blip_on_60fps_video_does_not_alert():
    assert _run_blip(60, 8, 0.5) == []                      # 8 frames = 0.13 s: the real false alert
    assert len(_run_blip(60, 8, 0.0)) == 1                  # ...which a frame count alone WOULD have confirmed


def test_sustained_detection_still_alerts_and_alert_is_not_delayed_beyond_min_duration():
    started = _run_blip(60, 120, 0.5)                        # 2 s of continuous detection
    assert len(started) == 1 and 0.5 <= started[0].started_ts <= 0.6


@pytest.mark.parametrize("fps", [7, 14, 25, 30, 60])
def test_min_duration_is_in_seconds_not_frames(fps):
    # a 0.3 s burst never alerts at any frame rate; a 1.0 s burst always does
    assert _run_blip(fps, max(1, int(0.3 * fps)), 0.5) == []
    assert len(_run_blip(fps, int(1.0 * fps), 0.5)) == 1


def test_window_is_widened_for_fast_video_so_duration_is_satisfiable():
    e = EventEngine(Settings(phone_window_frames=15, phone_min_duration_seconds=0.5).validate(), fps=60)
    assert e._window_frames() >= 45
    assert EventEngine(Settings(phone_window_frames=15, phone_min_duration_seconds=0.5).validate(), fps=7)._window_frames() == 15
