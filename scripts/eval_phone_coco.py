"""Phone-detection strategy benchmark on REAL labelled images (COCO val2017, held out from YOLO training).

NOT exam footage - general scenes. Purpose: (1) find WHY small phones are missed, (2) compare strategies
fairly, using a frozen dev/test split (artifacts/metrics/phone_coco_split.json). All selection decisions are
made on DEV only; TEST numbers are reported alongside but never used to choose.

Conditions:
  normal : image as is.
  far    : same real image shrunk (long side 426 px) into a 1280x720 canvas -> phones ~3x smaller relative to
           the frame, a stress test that mimics a camera far from the desk (a labelled transformation of real data).
Strategies call the production YoloDetector code path (crop assist) so results transfer to the app.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import pyarrow.parquet as pq

from examguard.detection.detector import PERSON, PHONE, Detection, YoloDetector, merge_detections
from examguard.events.geometry import containment, expand_box, iou

ROOT = Path(__file__).resolve().parents[1]
BUCKETS = [("<0.05%", 0, 0.0005), ("0.05-0.25%", 0.0005, 0.0025), ("0.25-1%", 0.0025, 0.01), (">=1%", 0.01, 2)]
FLOOR = 0.05
MODELS = {"yolo11n": "yolo11n.pt", "yolo11s": "yolo11s.pt", "yolo11m": "yolo11m.pt", "yolov8n": "yolov8n.pt", "yolo26n": "yolo26n.pt"}


# ----------------------------------------------------------------------------- data
def load_split():
    return json.loads((ROOT / "artifacts/metrics/phone_coco_split.json").read_text())


def load_images(ids: set[int]):
    out = {}
    for f in sorted((ROOT / "data/eval/coco_hf").glob("*.parquet")):
        p = pq.ParquetFile(f)
        for rg in range(p.num_row_groups):
            ids_rg = p.read_row_group(rg, columns=["image_id"]).column("image_id").to_pylist()
            if not ids.intersection(ids_rg):
                continue
            for r in p.read_row_group(rg).to_pylist():
                if r["image_id"] in ids:
                    img = cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
                    ob = r["objects"]
                    phones = [tuple(b) for b, c in zip(ob["bbox"], ob["category"]) if c == 67]
                    out[r["image_id"]] = (img, phones)
    return out


def to_far(img, boxes, long_side=426, canvas=(1280, 720)):
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    c = np.full((canvas[1], canvas[0], 3), 114, np.uint8)
    ox, oy = (canvas[0] - small.shape[1]) // 2, (canvas[1] - small.shape[0]) // 2
    c[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    return c, [(b[0] * s + ox, b[1] * s + oy, b[2] * s + ox, b[3] * s + oy) for b in boxes]


# ----------------------------------------------------------------------------- strategies
def tiles(img, size=640, overlap=0.25):
    H, W = img.shape[:2]
    step = int(size * (1 - overlap))
    xs = list(range(0, max(W - size, 0) + 1, step)) or [0]
    ys = list(range(0, max(H - size, 0) + 1, step)) or [0]
    if xs[-1] + size < W:
        xs.append(W - size)
    if ys[-1] + size < H:
        ys.append(H - size)
    return [(x, y, min(x + size, W), min(y + size, H)) for y in ys for x in xs]


def detect_tiled(det: YoloDetector, img, size=640, overlap=0.25):
    out = []
    for x1, y1, x2, y2 in tiles(img, size, overlap):
        r = det.model.predict(img[y1:y2, x1:x2], imgsz=size, conf=FLOOR, iou=det.nms_iou, classes=[67], device=det.device, verbose=False)[0]
        for bx, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append(Detection(PHONE, 67, float(cf), (bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1), 0, 0.0, det.version))
    return merge_detections([], out, 0.5)


class Runner:
    def __init__(self, name):
        self.name = name
        self.det = YoloDetector(ROOT / "models" / MODELS[name], person_conf=0.1, phone_conf=FLOOR, imgsz=640)

    def persons(self, img):
        return [d for d in self.det.detect(img, 0, 0.0) if d.label == PERSON and d.confidence >= 0.5]

    def full(self, img, size):
        self.det.imgsz = size
        return [d for d in self.det.detect(img, 0, 0.0) if d.label == PHONE]

    def crops(self, img, persons, size, margin):
        return self.det.detect_phones_in_crops(img, [p.bbox for p in persons], 0, 0.0, imgsz=size, margin=margin, max_crops=12)


def strategies_r2(name):
    """Round 2: focus on input resolution. Persons for crops come from the same full pass (no extra person pass)."""
    def full_p(r, im, size):  # one pass at `size`: phones + persons (conf>=0.5)
        r.det.imgsz = size
        d = r.det.detect(im, 0, 0.0)
        return [x for x in d if x.label == PHONE], [x for x in d if x.label == PERSON and x.confidence >= 0.5]

    def uni(size, csize):
        def f(r, im, _p):
            ph, pe = full_p(r, im, size)
            return merge_detections(ph, r.crops(im, pe, csize, 0.15))
        return f
    s = [("full@640", lambda r, im, p: r.full(im, 640), False), ("full@960", lambda r, im, p: r.full(im, 960), False),
         ("full@1280", lambda r, im, p: r.full(im, 1280), False), ("full@1920", lambda r, im, p: r.full(im, 1920), False),
         ("tiles@640", lambda r, im, p: detect_tiled(r.det, im, 640, 0.25), False),
         ("union full640+crop416 [shipped]", uni(640, 416), False), ("union full640+crop640", uni(640, 640), False),
         ("union full1280+crop640", uni(1280, 640), False), ("union full1920+crop640", uni(1920, 640), False)]
    return s


def strategies(name):
    """(id, fn(runner, img, persons)->detections, uses_cached_persons)."""
    s = [("full@640", lambda r, im, p: r.full(im, 640)), ("crop@416", lambda r, im, p: r.crops(im, p, 416, 0.15)),
         ("crop@640", lambda r, im, p: r.crops(im, p, 640, 0.15)),
         ("union full640+crop640", lambda r, im, p: merge_detections(r.full(im, 640), r.crops(im, p, 640, 0.15)))]
    if name == "yolo11n":
        s += [("full@960", lambda r, im, p: r.full(im, 960)), ("full@1280", lambda r, im, p: r.full(im, 1280)),
              ("crop@320", lambda r, im, p: r.crops(im, p, 320, 0.15)), ("crop@960", lambda r, im, p: r.crops(im, p, 960, 0.15)),
              ("crop@640 margin0", lambda r, im, p: r.crops(im, p, 640, 0.0)), ("crop@640 margin.3", lambda r, im, p: r.crops(im, p, 640, 0.3)),
              ("union full640+crop416", lambda r, im, p: merge_detections(r.full(im, 640), r.crops(im, p, 416, 0.15))),
              ("tiles@640", lambda r, im, p: detect_tiled(r.det, im, 640, 0.25)),
              ("union tiles640+crop640", lambda r, im, p: merge_detections(detect_tiled(r.det, im, 640, 0.25), r.crops(im, p, 640, 0.15)))]
    return [(sid, fn, "crop" in sid) for sid, fn in s]


# ----------------------------------------------------------------------------- metrics
def match(dets, gts, thr=0.5):
    """Greedy match by confidence. Returns list of (conf, is_tp) and per-GT matched flags."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i].confidence)
    used, res = [False] * len(gts), []
    for i in order:
        best, bj = 0.0, -1
        for j, g in enumerate(gts):
            if not used[j]:
                v = iou(dets[i].bbox, g)
                if v > best:
                    best, bj = v, j
        if best >= thr:
            used[bj] = True
            res.append((dets[i].confidence, True))
        else:
            res.append((dets[i].confidence, False))
    return res, used


def ap50(pairs, n_gt):
    if n_gt == 0:
        return float("nan")
    pairs = sorted(pairs, key=lambda x: -x[0])
    tp = np.cumsum([p[1] for p in pairs]); fp = np.cumsum([not p[1] for p in pairs])
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def near_person(d, persons):
    return any(containment(d.bbox, expand_box(p.bbox, 0.15)) >= 0.15 for p in persons)


def score(records, area_of_image, conf_list=(0.25, 0.35, 0.5)):
    """records: list of dict(dets, gts, persons, shape, is_pos). Returns metrics incl. size buckets + near-person variant."""
    out = {}
    for variant in ("all", "near_person"):
        pairs, n_gt = [], 0
        per_thr = {c: {"tp": 0, "fp": 0, "fp_neg_img": 0} for c in conf_list}
        bucket_tot = {b[0]: 0 for b in BUCKETS}; bucket_hit = {c: {b[0]: 0 for b in BUCKETS} for c in conf_list}
        for rec in records:
            dets = [d for d in rec["dets"] if variant == "all" or near_person(d, rec["persons"])]
            gts = rec["gts"]; n_gt += len(gts)
            m, _ = match(dets, gts)
            pairs += m
            for c in conf_list:
                dc = [d for d in dets if d.confidence >= c]
                mm, used = match(dc, gts)
                tp = sum(1 for _, t in mm if t); fp = len(mm) - tp
                per_thr[c]["tp"] += tp; per_thr[c]["fp"] += fp
                if not rec["is_pos"]:
                    per_thr[c]["fp_neg_img"] += int(fp > 0)
                if variant == "all":
                    for g, u in zip(gts, used):
                        frac = ((g[2] - g[0]) * (g[3] - g[1])) / (rec["shape"][0] * rec["shape"][1])
                        for bn, lo, hi in BUCKETS:
                            if lo <= frac < hi:
                                bucket_hit[c][bn] += int(u)
            if variant == "all":
                for g in gts:
                    frac = ((g[2] - g[0]) * (g[3] - g[1])) / (rec["shape"][0] * rec["shape"][1])
                    for bn, lo, hi in BUCKETS:
                        if lo <= frac < hi:
                            bucket_tot[bn] += 1
        n_neg = sum(1 for r in records if not r["is_pos"])
        v = {"AP50": round(ap50(pairs, n_gt), 4), "n_gt": n_gt, "n_neg_images": n_neg}
        for c in conf_list:
            t = per_thr[c]
            v[f"@{c}"] = {"recall": round(t["tp"] / n_gt, 3) if n_gt else None, "precision": round(t["tp"] / (t["tp"] + t["fp"]), 3) if (t["tp"] + t["fp"]) else None,
                          "FP": t["fp"], "neg_images_with_FP": t["fp_neg_img"]}
            if variant == "all":
                v[f"@{c}"]["recall_by_size"] = {b: (round(bucket_hit[c][b] / bucket_tot[b], 3) if bucket_tot[b] else None) for b in bucket_tot}
        if variant == "all":
            v["gt_by_size"] = bucket_tot
        out[variant] = v
    return out


# ----------------------------------------------------------------------------- main
def run(models, conditions, splits, sset="round1", only_strats=None):
    spl = load_split()
    all_ids = {i for s in splits for k in ("positives", "negatives") for i in spl[s][k]}
    imgs = load_images(all_ids)
    print(f"loaded {len(imgs)} images", flush=True)
    results = []
    proc = psutil.Process()
    for mname in models:
        rss0 = proc.memory_info().rss
        runner = Runner(mname)
        rss_after_load = proc.memory_info().rss
        runner.persons(next(iter(imgs.values()))[0])  # warm-up
        for cond in conditions:
            cache = {}
            for s in splits:
                for i in spl[s]["positives"] + spl[s]["negatives"]:
                    img, ph = imgs[i]
                    if cond == "far":
                        img, ph = to_far(img, ph)
                    elif cond == "far1080":  # phones keep COCO pixel size; frame is 1920x1080 -> a 640 input shrinks them 3x
                        img, ph = to_far(img, ph, 640, (1920, 1080))
                    t = time.perf_counter(); pers = runner.persons(img); tp_ = time.perf_counter() - t
                    cache[(s, i)] = (img, ph, pers, tp_)
            strat_list = strategies_r2(mname) if sset == "round2" else strategies(mname)
            if only_strats:
                strat_list = [t for t in strat_list if t[0] in only_strats]
            for sid, fn, cached in strat_list:
                for s in splits:
                    recs, times = [], []
                    for i in spl[s]["positives"] + spl[s]["negatives"]:
                        img, ph, pers, tp_ = cache[(s, i)]
                        t = time.perf_counter(); d = fn(runner, img, pers); dt = time.perf_counter() - t
                        times.append(dt + (tp_ if cached else 0))
                        recs.append({"dets": d, "gts": ph, "persons": pers, "shape": img.shape[:2], "is_pos": i in set(spl[s]["positives"])})
                    sc = score(recs, None)
                    row = {"model": mname, "condition": cond, "split": s, "strategy": sid, "ms_per_image": round(1000 * float(np.mean(times)), 1),
                           "process_rss_mb": round(proc.memory_info().rss / 2**20), "model_load_rss_delta_mb": round((rss_after_load - rss0) / 2**20), **sc}
                    results.append(row)
                    a = row["all"]
                    print(f"{mname:8} {cond:6} {s:4} {sid:26} AP50={a['AP50']:.3f} R@.25={a['@0.25']['recall']} P@.25={a['@0.25']['precision']} FP@.25={a['@0.25']['FP']} "
                          f"nearP AP50={row['near_person']['AP50']:.3f} {row['ms_per_image']}ms", flush=True)
        del runner
    return results, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS)); ap.add_argument("--conditions", default="normal,far")
    ap.add_argument("--splits", default="dev,test"); ap.add_argument("--set", default="round1"); ap.add_argument("--only", default=""); ap.add_argument("--out", default=str(ROOT / "artifacts/metrics/phone_coco_benchmark.json"))
    a = ap.parse_args()
    res, peak = run(a.models.split(","), a.conditions.split(","), a.splits.split(","), a.set, set(x for x in a.only.split("|") if x) or None)
    Path(a.out).write_text(json.dumps({"note": "REAL labelled COCO val2017 images (general scenes, not exam footage); selection on dev only",
                                       "peak_process_maxrss_mb": round(peak), "rows": res}, indent=1))
    print("written", a.out)
