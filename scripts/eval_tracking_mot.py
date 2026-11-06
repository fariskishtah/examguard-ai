"""Tracking evaluation on MOTChallenge MOT15 train sequences (real footage + ground-truth identities).

Data: MOT15 (CC BY-NC-SA 3.0), used locally for research evaluation only, never redistributed.
Split frozen in artifacts/metrics/mot15_split.json: tuning on DEV sequences only, TEST sequences held out.
Detector = our production YoloDetector (generic COCO yolo11n); tracker = production PersonTracker.
Metrics via py-motmetrics (IDF1, MOTA, ID switches, fragmentations, IDs created vs GT identities).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np

if not hasattr(np, "asfarray"):  # py-motmetrics 1.4 still calls np.asfarray, removed in NumPy 2.0 (evaluation-only shim)
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]
import motmetrics as mm  # noqa: E402

from examguard.detection.detector import PERSON, Detection, YoloDetector
from examguard.tracking.tracker import PersonTracker

ROOT = Path(__file__).resolve().parents[1]
MOT = ROOT / "data/eval/mot15/MOT15/train"
CACHE = ROOT / "data/eval/cache"


def seq_info(seq):
    kv = dict(l.strip().split("=", 1) for l in (MOT / seq / "seqinfo.ini").read_text().splitlines() if "=" in l)
    return {"fps": float(kv["frameRate"]), "n": int(kv["seqLength"]), "w": int(kv["imWidth"]), "h": int(kv["imHeight"]), "ext": kv.get("imExt", ".jpg")}


def load_gt(seq):
    gt = {}
    for row in np.loadtxt(MOT / seq / "gt" / "gt.txt", delimiter=","):
        f, i, x, y, w, h, conf = row[:7]
        if conf != 0:
            gt.setdefault(int(f), []).append((int(i), x, y, w, h))
    return gt


def detect_sequence(seq, imgsz=640, model="yolo11n.pt"):
    """Per-frame person detections, cached as plain JSON (no pickle)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{seq}_{Path(model).stem}_{imgsz}.json"
    info = seq_info(seq)
    if f.exists():
        z = json.loads(f.read_text())
        return [[(c, tuple(b)) for c, b in fr] for fr in z["dets"]], float(z["ms"])
    det = YoloDetector(ROOT / "models" / model, person_conf=0.05, phone_conf=0.99, imgsz=imgsz)
    dets, t0 = [], time.perf_counter()
    for k in range(1, info["n"] + 1):
        img = cv2.imread(str(MOT / seq / "img1" / f"{k:06d}{info['ext']}"))
        dets.append([(d.confidence, d.bbox) for d in det.detect(img, k, k / info["fps"]) if d.label == PERSON])
    ms = (time.perf_counter() - t0) / info["n"] * 1000
    f.write_text(json.dumps({"dets": [[[c, list(b)] for c, b in fr] for fr in dets], "ms": ms}))
    return dets, ms


def run_tracker(seq, dets, params, floor):
    info = seq_info(seq)
    kw = dict(params)
    tr = PersonTracker(**kw)
    need_img = kw.get("kind") == "botsort"
    out, t_track = {}, 0.0
    for k in range(1, info["n"] + 1):
        persons = [Detection("person", 0, c, b, k, k / info["fps"], "mot") for c, b in dets[k - 1] if c >= floor]
        img = cv2.imread(str(MOT / seq / "img1" / f"{k:06d}{info['ext']}")) if need_img else None
        t = time.perf_counter(); res = tr.update(persons, img); t_track += time.perf_counter() - t
        out[k] = [(t_.track_id, *t_.bbox) for t_ in res]
    return out, t_track / info["n"] * 1000


def evaluate(seq, hyp, gt):
    acc = mm.MOTAccumulator(auto_id=True)
    for k in range(1, seq_info(seq)["n"] + 1):
        g = gt.get(k, []); h = hyp.get(k, [])
        gb = [[x, y, w, hh] for _, x, y, w, hh in g]
        hb = [[x1, y1, x2 - x1, y2 - y1] for _, x1, y1, x2, y2 in h]
        d = mm.distances.iou_matrix(gb, hb, max_iou=0.5) if g and h else np.empty((len(g), len(h)))
        acc.update([i for i, *_ in g], [i for i, *_ in h], d)
    s = mm.metrics.create().compute(acc, metrics=["idf1", "mota", "num_switches", "num_fragmentations", "mostly_tracked",
                                                  "mostly_lost", "num_unique_objects", "precision", "recall"], name=seq)
    ids_created = len({t[0] for v in hyp.values() for t in v})
    r = s.iloc[0]
    return {"IDF1": round(float(r.idf1), 4), "MOTA": round(float(r.mota), 4), "IDSW": int(r.num_switches), "frag": int(r.num_fragmentations),
            "GT_ids": int(r.num_unique_objects), "ids_created": ids_created, "id_ratio": round(ids_created / max(int(r.num_unique_objects), 1), 2),
            "recall": round(float(r.recall), 4), "precision": round(float(r.precision), 4), "MT": int(r.mostly_tracked), "ML": int(r.mostly_lost)}


def eval_config(name, params, floor, seqs, det_cache, gt_cache):
    rows, ms = {}, []
    for s in seqs:
        hyp, t = run_tracker(s, det_cache[s][0], params, floor)
        rows[s] = evaluate(s, hyp, gt_cache[s]); ms.append(t)
    mean = lambda k: round(float(np.mean([r[k] for r in rows.values()])), 4)  # noqa: E731
    return {"config": name, "params": params, "det_floor": floor, "per_seq": rows, "mean_IDF1": mean("IDF1"), "mean_MOTA": mean("MOTA"),
            "total_IDSW": sum(r["IDSW"] for r in rows.values()), "mean_id_ratio": mean("id_ratio"), "tracker_ms_per_frame": round(float(np.mean(ms)), 2)}


CONFIGS = {  # explicit named configurations
    "old_production (byte .35/.4 buf30 floor .35)": (dict(high=0.35, low=0.1, new=0.4, buffer=30, match=0.8), 0.35),
    "current_production (byte .5/.5 buf60 floor .1)": (dict(high=0.5, low=0.1, new=0.5, buffer=60, match=0.8), 0.10),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/metrics/tracking_mot15.json"))
    ap.add_argument("--reid", default="")  # e.g. yolo11n-cls.pt
    a = ap.parse_args()
    spl = json.loads((ROOT / "artifacts/metrics/mot15_split.json").read_text())
    dev, test = spl["dev_tuning"], spl["heldout_test"]
    det_cache = {s: detect_sequence(s) for s in dev + test}
    gt_cache = {s: load_gt(s) for s in dev + test}
    print("detections cached; det ms/frame:", {s: round(det_cache[s][1], 1) for s in det_cache}, flush=True)

    # 1) ByteTrack grid, selection on DEV only
    grid = []
    for high, buf, match, floor in itertools.product((0.4, 0.5, 0.6), (30, 60, 120), (0.7, 0.8, 0.9), (0.10, 0.25)):
        grid.append((f"byte high={high} buf={buf} match={match} floor={floor}", dict(high=high, low=0.1, new=high, buffer=buf, match=match), floor))
    # 2) BoT-SORT variants (GMC, optional appearance)
    bot = [("botsort gmc=none", dict(kind="botsort", high=0.5, low=0.1, new=0.5, buffer=60, match=0.8, gmc="none"), 0.10),
           ("botsort gmc=sparseOptFlow", dict(kind="botsort", high=0.5, low=0.1, new=0.5, buffer=60, match=0.8, gmc="sparseOptFlow"), 0.10)]
    if a.reid:
        for at in (0.5, 0.8):
            bot.append((f"botsort gmc=sparseOptFlow reid={a.reid} app={at}", dict(kind="botsort", high=0.5, low=0.1, new=0.5, buffer=60, match=0.8,
                                                                               gmc="sparseOptFlow", reid_model=a.reid, appearance_thresh=at), 0.10))
    named = [(n, p, f) for n, (p, f) in CONFIGS.items()]
    all_cfg = named + grid + bot
    dev_res = []
    for n, p, f in all_cfg:
        r = eval_config(n, p, f, dev, det_cache, gt_cache); dev_res.append(r)
        print(f"DEV  {n:62} IDF1={r['mean_IDF1']:.3f} MOTA={r['mean_MOTA']:.3f} IDSW={r['total_IDSW']:4} id_ratio={r['mean_id_ratio']} {r['tracker_ms_per_frame']}ms", flush=True)
    best_byte = max((r for r in dev_res if r["config"].startswith("byte ")), key=lambda r: r["mean_IDF1"])
    best_bot = max((r for r in dev_res if r["config"].startswith("botsort")), key=lambda r: r["mean_IDF1"])
    chosen = [r["config"] for r in dev_res if r["config"] in CONFIGS] + [best_byte["config"], best_bot["config"]]
    cfg_by_name = {n: (p, f) for n, p, f in all_cfg}
    test_res = []
    for n in chosen + [n for n, *_ in bot if n not in chosen]:
        p, f = cfg_by_name[n]
        r = eval_config(n, p, f, test, det_cache, gt_cache); test_res.append(r)
        print(f"TEST {n:62} IDF1={r['mean_IDF1']:.3f} MOTA={r['mean_MOTA']:.3f} IDSW={r['total_IDSW']:4} id_ratio={r['mean_id_ratio']} {r['tracker_ms_per_frame']}ms", flush=True)
    Path(a.out).write_text(json.dumps({"note": "MOT15 train sequences; selection on dev only; real ground-truth identities; IoU>=0.5",
                                       "dev_sequences": dev, "test_sequences": test, "selected_on_dev": {"byte": best_byte["config"], "botsort": best_bot["config"]},
                                       "dev": dev_res, "test": test_res}, indent=1))
    print("written", a.out)


if __name__ == "__main__":
    main()
