"""Phone detection + event metrics on the SYNTHETIC composite fixture.

!! These numbers describe behaviour on a synthetic composite (a real CC0 phone photo pasted into real
!! CC0 street footage with known boxes). They are NOT real-world accuracy for exam-room phone use.
!! Thresholds below were explored on THIS fixture only; they are not independently validated.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

from examguard.config import Settings
from examguard.detection.detector import PERSON, PHONE, YoloDetector, merge_detections
from examguard.events.engine import EventEngine
from examguard.events.geometry import iou
from examguard.tracking.tracker import PersonTracker
from examguard.video.reader import iter_frames, probe_video

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "data/fixtures/synthetic_phone_pedestrian.mp4"
IOU_T = 0.3


def build_cache(det, info, imgsz_crop):
    """Per frame: full-frame detections, tracker output, optional crop-assist phones (computed once)."""
    s = Settings()
    tracker = PersonTracker(s.person_conf, s.track_low_thresh, s.person_conf, s.track_buffer, s.match_thresh)
    rows, t_full, t_crop = [], 0.0, 0.0
    for idx, ts, fr in iter_frames(info):
        a = time.perf_counter(); d = det.detect(fr, idx, ts); t_full += time.perf_counter() - a
        tracks = tracker.update([x for x in d if x.label == PERSON], fr)
        extra = []
        if imgsz_crop:
            a = time.perf_counter()
            extra = det.detect_phones_in_crops(fr, [t.bbox for t in tracks], idx, ts, imgsz=imgsz_crop)
            t_crop += time.perf_counter() - a
        rows.append((idx, ts, d, tracks, extra))
    n = len(rows)
    return rows, t_full / n * 1000, t_crop / n * 1000


def evaluate(rows, gt, t0, t1, fps, conf, sup, resolve, assist):
    tp_frames = tp_dets = fp_dets = 0
    per_frame_phones = []
    for idx, ts, d, tracks, extra in rows:
        ph = [x for x in d if x.label == PHONE]
        if assist:
            ph = merge_detections(ph, extra)
        ph = [x for x in ph if x.confidence >= conf]
        per_frame_phones.append(ph)
        hit = [x for x in ph if idx in gt and iou(x.bbox, gt[idx]) >= IOU_T]
        tp_dets += len(hit); fp_dets += len(ph) - len(hit); tp_frames += bool(hit)
    prec = tp_dets / (tp_dets + fp_dets) if (tp_dets + fp_dets) else None
    s = Settings(phone_conf=conf, phone_min_support_frames=sup, phone_window_frames=max(15, sup),
                 resolve_after_seconds=resolve, region_polygon=[]).validate()
    eng = EventEngine(s, fps)
    incs = []
    for (idx, ts, d, tracks, extra), ph in zip(rows, per_frame_phones):
        incs += eng.process(idx, ts, tracks, ph).started
    eng.finalize(rows[-1][1])
    ok, bad = [], []
    for i in incs:
        in_win = i.started_ts <= t1 and (i.ended_ts or i.last_ts) >= t0
        right = i.last_frame in gt and i.phone_bbox is not None and iou(i.phone_bbox, gt[i.last_frame]) >= IOU_T
        (ok if (in_win and right) else bad).append(i)
    cover = sum(max(0.0, min(i.ended_ts or i.last_ts, t1) - max(i.started_ts, t0)) for i in ok) / (t1 - t0)
    return {"frame_phone_precision": None if prec is None else round(prec, 3), "frame_phone_recall": round(tp_frames / len(gt), 3),
            "incidents_total": len(incs), "incidents_correct": len(ok), "incidents_false": len(bad),
            "detection_delay_s": round(min(i.started_ts for i in ok) - t0, 2) if ok else None,
            "gt_window_alert_coverage": round(cover, 3), "duplicate_incidents": max(0, len(ok) - 1)}


def run():
    meta = json.loads(Path(str(FIX) + ".fixture.json").read_text())
    gt = {int(k): tuple(v) for k, v in meta["ground_truth_phone_boxes_by_frame"].items()}
    t0, t1 = meta["phone_visible_from_s"], meta["phone_visible_to_s"]
    info = probe_video(FIX)
    out = []
    for imgsz in (640, 960):
        det = YoloDetector(ROOT / "models/yolo11n.pt", person_conf=0.10, phone_conf=0.05, imgsz=imgsz)
        rows, ms_full, ms_crop = build_cache(det, info, 416)
        for assist, conf, sup, resolve in itertools.product((False, True), (0.10, 0.25, 0.35), (3, 5), (2.0, 4.0)):
            m = evaluate(rows, gt, t0, t1, info.fps, conf, sup, resolve, assist)
            r = {"imgsz": imgsz, "crop_assist": assist, "crop_imgsz": 416 if assist else None, "phone_conf": conf,
                 "support_frames": sup, "resolve_after_s": resolve, **m,
                 "infer_ms_per_frame": round(ms_full + (ms_crop if assist else 0), 1)}
            out.append(r)
            print(r)
    path = ROOT / "artifacts/metrics/phone_fixture_eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"WARNING": "SYNTHETIC composite fixture - not real-world accuracy; thresholds explored on this fixture only",
                                "fixture": meta["source_video"], "iou_threshold": IOU_T, "gt_frames": len(gt),
                                "gt_window_s": [t0, t1], "rows": out}, indent=2))


if __name__ == "__main__":
    run()
