"""Software-level tracker tests on SYNTHETIC boxes (not real-world validation)."""
from examguard.detection.detector import Detection
from examguard.tracking.tracker import PersonTracker


def person(x1, y1, x2, y2, fi, conf=0.9):
    return Detection("person", 0, conf, (x1, y1, x2, y2), fi, fi / 30, "test")


def run(frames):
    t = PersonTracker()
    return [t.update(f) for f in frames]


def test_ids_persist_for_moving_people_and_stay_distinct():
    frames = []
    for i in range(40):
        frames.append([person(100 + 3 * i, 100, 200 + 3 * i, 300, i),
                       person(600 - 3 * i, 120, 700 - 3 * i, 320, i)])
    out = run(frames)
    ids_a = {tr.track_id for f in out[3:] for tr in f if tr.bbox[0] < 400}
    ids_b = {tr.track_id for f in out[3:] for tr in f if tr.bbox[0] >= 400}
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_short_occlusion_keeps_id():
    frames = []
    for i in range(50):
        dets = [person(100 + 2 * i, 100, 200 + 2 * i, 300, i)]
        if 20 <= i < 26:  # 6 frames missing
            dets = []
        frames.append(dets)
    out = run(frames)
    ids = {tr.track_id for f in out for tr in f}
    assert len(ids) == 1


def test_track_dropped_after_buffer_and_new_id_on_return():
    frames = []
    for i in range(160):
        dets = [person(100, 100, 200, 300, i)] if (i < 20 or i >= 130) else []
        frames.append(dets)
    out = run(frames)
    early = {tr.track_id for f in out[:20] for tr in f}
    late = {tr.track_id for f in out[130:] for tr in f}
    assert early and late and early != late  # documented limitation: long gaps => new ID


def test_empty_frames_ok():
    assert run([[], [], []]) == [[], [], []]
