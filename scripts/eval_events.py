"""Event-level validation.

PART A  PHONE_VISIBLE_NEAR_TRACK on HELD-OUT *synthetic* fixtures (software-level, NOT real phone use) +
        their unmodified 'clean' twins for false alerts. Runs the real VideoProcessor.
PART B  MONITORED_REGION_EXIT on REAL MOT15 footage. Ground-truth events are derived by feeding the GROUND-TRUTH
        person boxes through the same EventEngine rule; predicted events come from detector + tracker. This
        measures how much detection/tracking degrades the event; it is not human-labelled exit annotation.
Event matching: predicted event matches a GT event if start times differ <= tol and the person boxes overlap (IoU>=0.3).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_tracking_mot import MOT, detect_sequence, load_gt, seq_info  # noqa: E402

from examguard.config import Paths, Settings  # noqa: E402
from examguard.database.db import Database  # noqa: E402
from examguard.database.repository import Repository  # noqa: E402
from examguard.detection.detector import Detection  # noqa: E402
from examguard.events.engine import REGION_EVENT, EventEngine  # noqa: E402
from examguard.events.geometry import iou  # noqa: E402
from examguard.tracking.tracker import PersonTracker, TrackedPerson  # noqa: E402
from examguard.video.processor import VideoProcessor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HELD = ROOT / "data/fixtures/heldout"
REGIONS = {"left-half": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]], "right-half": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]],
           "center-band": [[0.25, 0], [0.75, 0], [0.75, 1], [0.25, 1]]}


def prf(tp, fp, fn):
    return {"precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3) if tp + fn else None}


# ------------------------------------------------------------------------------------------- PART A
def part_a(configs, held=None):
    HELD = held or globals()["HELD"]
    metas = {p.stem.replace("_phone", ""): json.loads(p.read_text()) for p in sorted(HELD.glob("*_phone.json"))}
    out = []
    for cname, overrides in configs.items():
        tot = {"gt_events": 0, "gt_found": 0, "incidents_correct": 0, "incidents_false": 0, "dups": 0, "delays": [], "cover": [], "false_clean": 0,
               "hours_clean": 0.0, "hours_all": 0.0, "wall": 0.0, "frames": 0, "peak_rss_mb": 0}
        per = []
        for seq, meta in metas.items():
            for kind in ("phone", "clean"):
                video = HELD / f"{seq}_{kind}.mp4"
                tmp = Path(tempfile.mkdtemp())
                paths = Paths(tmp).ensure(); repo = Repository(Database(paths.db))
                s = Settings(notifications_enabled=False, save_annotated_video=False, **overrides)
                import psutil
                proc = psutil.Process(); t0 = time.perf_counter()
                summ = VideoProcessor(s, paths, repo).process(video)
                tot["wall"] += time.perf_counter() - t0; tot["frames"] += summ.frames_read
                tot["peak_rss_mb"] = max(tot["peak_rss_mb"], proc.memory_info().rss / 2**20)
                incs = repo.list_incidents(); repo.db.close()
                hrs = meta["duration_s"] / 3600; tot["hours_all"] += hrs
                if kind == "clean":
                    tot["false_clean"] += len(incs); tot["hours_clean"] += hrs
                    per.append({"seq": seq, "kind": kind, "incidents": len(incs)}); continue
                t0s, t1s = meta["phone_visible_from_s"], meta["phone_visible_to_s"]
                gt = {int(k): v for k, v in meta["ground_truth_phone_boxes_by_frame"].items()}
                fps = meta["fps"]
                ok, bad = [], []
                for i in sorted(incs, key=lambda x: x.first_detected_at):
                    ev = json.loads(i.evidence_json) if i.evidence_json else {}
                    pb = ev.get("phone_bbox")
                    last_f = int(round(i.last_detected_at * fps))
                    near = min(gt, key=lambda f: abs(f - last_f)) if gt else None
                    right = bool(pb and near is not None and abs(near - last_f) <= 3 and iou(tuple(pb), tuple(gt[near])) >= 0.3)
                    in_win = i.first_detected_at <= t1s + 1.0 and i.last_detected_at >= t0s
                    (ok if (in_win and right) else bad).append(i)
                tot["gt_events"] += 1; tot["gt_found"] += int(bool(ok)); tot["incidents_correct"] += len(ok); tot["incidents_false"] += len(bad)
                tot["dups"] += max(0, len(ok) - 1)
                if ok:
                    tot["delays"].append(min(i.first_detected_at for i in ok) - t0s)
                    tot["cover"].append(sum(max(0.0, min(i.last_detected_at, t1s) - max(i.first_detected_at, t0s)) for i in ok) / max(t1s - t0s, 1e-6))
                per.append({"seq": seq, "kind": kind, "correct": len(ok), "false": len(bad), "delay_s": round(min(i.first_detected_at for i in ok) - t0s, 2) if ok else None})
        n_false = tot["incidents_false"] + tot["false_clean"]
        out.append({"config": cname, "overrides": overrides, "event_recall": round(tot["gt_found"] / tot["gt_events"], 3), "gt_events": tot["gt_events"],
                    "event_precision": round(tot["incidents_correct"] / max(tot["incidents_correct"] + n_false, 1), 3),
                    "duplicate_incidents": tot["dups"], "duplicate_rate": round(tot["dups"] / max(tot["incidents_correct"], 1), 3),
                    "mean_detection_delay_s": round(float(np.mean(tot["delays"])), 2) if tot["delays"] else None,
                    "mean_window_coverage": round(float(np.mean(tot["cover"])), 3) if tot["cover"] else None,
                    "false_incidents_in_positive_videos": tot["incidents_false"], "false_incidents_in_clean_twins": tot["false_clean"],
                    "false_alerts_per_hour_clean": round(tot["false_clean"] / tot["hours_clean"], 1) if tot["hours_clean"] else None, "clean_hours": round(tot["hours_clean"], 4),
                    "end_to_end_fps": round(tot["frames"] / tot["wall"], 1), "peak_rss_mb": round(tot["peak_rss_mb"]), "per_video": per})
        print(f"A {cname:34} recall={out[-1]['event_recall']} precision={out[-1]['event_precision']} dups={tot['dups']} delay={out[-1]['mean_detection_delay_s']} "
              f"cover={out[-1]['mean_window_coverage']} FA/h(clean)={out[-1]['false_alerts_per_hour_clean']} fps={out[-1]['end_to_end_fps']}", flush=True)
    return out


# ------------------------------------------------------------------------------------------- PART B
def region_events(seq, tracks_by_frame, poly, fps, w, h):
    s = Settings(region_polygon=poly, region_exit_min_seconds=1.5, resolve_after_seconds=2.0, event_cooldown_seconds=10.0).validate()
    eng = EventEngine(s, fps); eng.frame_size = (w, h)
    started = []
    for k in sorted(tracks_by_frame):
        trs = [TrackedPerson(i, (x1, y1, x2, y2), 1.0, k, k / fps, None) for i, x1, y1, x2, y2 in tracks_by_frame[k]]
        for inc in eng.process(k, k / fps, trs, []).started:
            started.append({"ts": inc.started_ts, "frame": k, "bbox": inc.person_bbox, "track": inc.track_id})
    return started


def part_b(configs, seqs, tol_s=2.5):
    rows = []
    for cname, (tparams, floor) in configs.items():
        agg = {"tp": 0, "fp": 0, "fn": 0, "dups": 0, "delays": [], "hours": 0.0, "n_gt": 0}
        per = []
        for seq in seqs:
            info = seq_info(seq); gt = load_gt(seq)
            gt_tracks = {k: [(i, x, y, x + w, y + h) for i, x, y, w, h in v] for k, v in gt.items()}
            dets, _ = detect_sequence(seq)
            tr = PersonTracker(**tparams); hyp = {}
            for k in range(1, info["n"] + 1):
                persons = [Detection("person", 0, c, b, k, k / info["fps"], "m") for c, b in dets[k - 1] if c >= floor]
                img = cv2.imread(str(MOT / seq / "img1" / f"{k:06d}{info['ext']}")) if tparams.get("kind") == "botsort" else None
                hyp[k] = [(t.track_id, *t.bbox) for t in tr.update(persons, img)]
            for rname, poly in REGIONS.items():
                ge = region_events(seq, gt_tracks, poly, info["fps"], info["w"], info["h"])
                pe = region_events(seq, hyp, poly, info["fps"], info["w"], info["h"])
                used = set(); matched_pred = set(); delays = []
                for gi, g in enumerate(ge):
                    cand = [pi for pi, p in enumerate(pe) if abs(p["ts"] - g["ts"]) <= tol_s and
                            iou(p["bbox"], next((b[1:] for b in gt_tracks.get(p["frame"], []) if b[0] == g["track"]), (0, 0, 1, 1))) >= 0.3]
                    if cand:
                        used.add(gi); matched_pred.update(cand)
                        agg["dups"] += len(cand) - 1
                        delays.append(min(pe[c]["ts"] for c in cand) - g["ts"])
                tp = len(used); fn = len(ge) - tp; fp = len(pe) - len(matched_pred)
                agg["tp"] += tp; agg["fn"] += fn; agg["fp"] += fp; agg["n_gt"] += len(ge); agg["delays"] += delays
                agg["hours"] += info["n"] / info["fps"] / 3600
                per.append({"seq": seq, "region": rname, "gt_events": len(ge), "pred_events": len(pe), "tp": tp, "fp": fp, "fn": fn})
        rows.append({"config": cname, "tracker": tparams, "n_gt_events": agg["n_gt"], **prf(agg["tp"], agg["fp"], agg["fn"]), "tp": agg["tp"], "fp": agg["fp"], "fn": agg["fn"],
                     "duplicate_alerts": agg["dups"], "mean_delay_s": round(float(np.mean(agg["delays"])), 2) if agg["delays"] else None,
                     "false_alerts_per_hour": round(agg["fp"] / agg["hours"], 1) if agg["hours"] else None, "hours_evaluated": round(agg["hours"], 4), "per_run": per})
        r = rows[-1]
        print(f"B {cname:40} GT={r['n_gt_events']:3} P={r['precision']} R={r['recall']} dup={r['duplicate_alerts']} delay={r['mean_delay_s']} FA/h={r['false_alerts_per_hour']} ({r['hours_evaluated']} h)", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="ab"); ap.add_argument("--configs", required=True, help="JSON file: {'a': {name: settings_overrides}, 'b': {name: [tracker_kwargs, floor]}}")
    ap.add_argument("--b-seqs", default="test"); ap.add_argument("--fixtures", default=str(HELD)); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.configs).read_text())
    spl = json.loads((ROOT / "artifacts/metrics/mot15_split.json").read_text())
    res = {"note": "A: held-out SYNTHETIC fixtures (software-level). B: REAL MOT15 footage, GT-trajectory-derived region-exit events."}
    if "a" in a.part:
        res["part_a_phone_synthetic_heldout"] = part_a(cfg["a"], Path(a.fixtures))
    if "b" in a.part:
        seqs = spl["heldout_test"] if a.b_seqs == "test" else spl["dev_tuning"] if a.b_seqs == "dev" else spl["dev_tuning"] + spl["heldout_test"]
        res["part_b_region_exit_real_footage"] = {"sequences": seqs, "rows": part_b({k: (v[0], v[1]) for k, v in cfg["b"].items()}, seqs)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("written", a.out)
