"""Tracking FRAGMENTATION PROXY on unannotated clips (NOT IDF1/HOTA/MOTA).

Without ground-truth identities we cannot measure ID switches or IDF1. This reports only:
  max_concurrent : most people tracked in one frame (a LOWER BOUND on distinct people)
  n_ids          : distinct track IDs created
  ids_ge_1s      : IDs lasting >= 1 s
  fragmentation  : n_ids / max_concurrent  (>1 means IDs were created beyond the simultaneous count;
                   it mixes true new people with identity fragmentation, so it is only a rough indicator)
Detections are computed once per (clip, imgsz) and cached so tracker settings can be compared cheaply.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

from examguard.detection.detector import PERSON, YoloDetector
from examguard.tracking.tracker import PersonTracker
from examguard.video.reader import iter_frames, probe_video

ROOT = Path(__file__).resolve().parents[1]


def detect_all(path: Path, imgsz: int, conf: float):
    det = YoloDetector(ROOT / "models/yolo11n.pt", person_conf=conf, imgsz=imgsz)
    info = probe_video(path)
    out = []
    for idx, ts, fr in iter_frames(info):
        out.append([d for d in det.detect(fr, idx, ts) if d.label == PERSON])
    return info, out


def run_tracker(dets, **kw):
    tr = PersonTracker(**kw)
    spans, conc = {}, 0
    for frame_dets in dets:
        res = tr.update(frame_dets)
        conc = max(conc, len(res))
        for t in res:
            ts = t.timestamp_s
            a = spans.setdefault(t.track_id, [ts, ts]); a[1] = ts
    durs = [b - a for a, b in spans.values()]
    return {"n_ids": len(spans), "max_concurrent": conc, "ids_ge_1s": sum(d >= 1.0 for d in durs),
            "median_id_duration_s": round(statistics.median(durs), 2) if durs else 0,
            "fragmentation": round(len(spans) / max(conc, 1), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/metrics/tracking_proxy.json"))
    a = ap.parse_args()
    clips = sorted((ROOT / "data/samples").glob("*.webm"))
    grid = {"track_buffer": [30, 60, 90], "new_track_thresh": [0.4, 0.3], "high": [0.35, 0.5]}
    results = []
    for clip in clips:
        info, dets = detect_all(clip, 640, 0.25)  # low floor; the tracker's own thresholds do the filtering
        for buf, new, high in itertools.product(grid["track_buffer"], grid["new_track_thresh"], grid["high"]):
            m = run_tracker(dets, high=high, low=0.1, new=new, buffer=buf, match=0.8)
            results.append({"clip": clip.name, "frames": len(dets), "fps": info.fps, "imgsz": 640, "person_floor": 0.25,
                            "track_buffer": buf, "new_track_thresh": new, "track_high_thresh": high, **m})
            print(f"{clip.name[:24]:24} buf={buf:3} new={new} high={high} -> {m}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"note": "fragmentation PROXY only; IDF1/HOTA/MOTA NOT EVALUATED (no ground truth)",
                                       "results": results}, indent=2))


if __name__ == "__main__":
    main()
