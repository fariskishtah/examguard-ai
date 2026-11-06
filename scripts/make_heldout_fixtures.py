"""Build HELD-OUT synthetic phone fixtures (software-level validation, NOT real phone-use footage).

Real MOT15 held-out sequences (see artifacts/metrics/mot15_split.json) + real CC0 phone photographs.
A phone crop is pasted at a fixed place inside a ground-truth person box for a known time window, so the exact
phone box per frame is known. Each sequence also gets an UNMODIFIED "negative twin" video (no phone added)
used to count false alerts. Output goes to data/fixtures/heldout/ (git-ignored; rebuildable).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_tracking_mot import MOT, load_gt, seq_info  # noqa: E402

from examguard.detection.detector import PHONE, YoloDetector  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Two fixture sets. F1 = MOT15 held-out sequences (it influenced the design after the first evaluation, so it is now
# DESIGN data). F2 = MOT15 dev sequences: first confirmation set, its results led to the track-merge rule, so also DESIGN
# data. F3 = three further MOT15 train sequences (one genuine 1080p) never used for anything before the FINAL run.
SETS = {
    "F1": ("heldout", {"PETS09-S2L1": "Smartphone_in_Hand_Unsplash_.jpg", "TUD-Stadtmitte": "Hand-smartphone-technology-calling.jpg",
                       "ETH-Sunnyday": "Smartphone_Use.jpg"}),
    "F3": ("final", {"ETH-Bahnhof": "Smartphone_in_Hand_Unsplash_.jpg", "KITTI-13": "Hand-smartphone-technology-calling.jpg",
                     "ADL-Rundle-8": "Smartphone_Use.jpg"}),
    "F2": ("confirm", {"ETH-Pedcross2": "Hand-smartphone-technology-calling.jpg", "TUD-Campus": "Smartphone_Use.jpg",
                       "KITTI-17": "Smartphone_in_Hand_Unsplash_.jpg"}),
}
WINDOW_S = {"PETS09-S2L1": 20.0, "TUD-Stadtmitte": 4.0, "ETH-Sunnyday": 10.0, "ETH-Pedcross2": 10.0, "TUD-Campus": 2.0, "KITTI-17": 8.0, "ETH-Bahnhof": 10.0, "KITTI-13": 8.0, "ADL-Rundle-8": 8.0}
PHONE_FRAC = 0.15  # phone height as a fraction of the person's box height (small, realistic)


def phone_crop(det: YoloDetector, photo: Path):
    """Crop the phone out of a real photograph (box found by a larger detector on the photo itself)."""
    img = cv2.imread(str(photo))
    det.imgsz = 1280
    ph = [d for d in det.detect(img, 0, 0.0) if d.label == PHONE]
    if not ph:
        raise SystemExit(f"no phone found in {photo.name}")
    b = max(ph, key=lambda d: d.confidence).bbox
    w, h = b[2] - b[0], b[3] - b[1]
    x1, y1, x2, y2 = max(0, int(b[0] - 0.04 * w)), max(0, int(b[1] - 0.04 * h)), min(img.shape[1], int(b[2] + 0.04 * w)), min(img.shape[0], int(b[3] + 0.04 * h))
    crop = img[y1:y2, x1:x2].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.rectangle(mask, (3, 3), (crop.shape[1] - 4, crop.shape[0] - 4), 255, -1)
    return crop, cv2.GaussianBlur(mask, (0, 0), 2)


def paste(frame, crop, mask, cx, cy, height):
    s = height / crop.shape[0]
    p = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (p.shape[1], p.shape[0])).astype(np.float32)[..., None] / 255
    h, w = p.shape[:2]
    x, y = int(cx - w / 2), int(cy - h / 2)
    H, W = frame.shape[:2]
    xa, ya, xb, yb = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if xb - xa < 4 or yb - ya < 4:
        return None
    sub = frame[ya:yb, xa:xb].astype(np.float32)
    mm_ = m[ya - y:yb - y, xa - x:xb - x]
    frame[ya:yb, xa:xb] = (sub * (1 - mm_) + p[ya - y:yb - y, xa - x:xb - x] * mm_).astype(np.uint8)
    return [xa, ya, xb, yb]


def best_segment(gt, n, min_h=90):
    """Longest contiguous run of frames in which one GT person is visible with a box at least min_h tall."""
    ids = {i for fr in gt.values() for i, *_ in fr}
    best = (0, None, 0, 0)
    for pid in ids:
        run_start, prev = None, None
        for f in range(1, n + 2):
            ok = any(i == pid and h >= min_h for i, _x, _y, _w, h in gt.get(f, []))
            if ok and run_start is None:
                run_start = f
            if (not ok) and run_start is not None:
                if f - run_start > best[0]:
                    best = (f - run_start, pid, run_start, f - 1)
                run_start = None
    return best


def writer(path, fps, size):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    assert w.isOpened()
    return w


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "F1"
    sub, PHOTOS = SETS[which]
    OUT = ROOT / "data/fixtures" / sub
    OUT.mkdir(parents=True, exist_ok=True)
    det = YoloDetector(ROOT / "models/yolo11m.pt", person_conf=0.5, phone_conf=0.3, imgsz=1280)
    for seq, photo in PHOTOS.items():
        info, gt = seq_info(seq), load_gt(seq)
        crop, mask = phone_crop(det, ROOT / "data/samples" / photo)
        _len, pid, f0, f1 = best_segment(gt, info["n"])
        wf = int(WINDOW_S[seq] * info["fps"])
        f1 = min(f1, f0 + wf - 1)
        size = (info["w"], info["h"])
        pos, neg = writer(OUT / f"{seq}_phone.mp4", info["fps"], size), writer(OUT / f"{seq}_clean.mp4", info["fps"], size)
        gtb = {}
        for k in range(1, info["n"] + 1):
            img = cv2.imread(str(MOT / seq / "img1" / f"{k:06d}{info['ext']}"))
            neg.write(img)
            if f0 <= k <= f1:
                for i, x, y, w, h in gt[k]:
                    if i == pid:
                        r = paste(img, crop, mask, x + w / 2, y + 0.40 * h, PHONE_FRAC * h)
                        if r:
                            gtb[k - 1] = r  # 0-based frame index as used by the video reader
            pos.write(img)
        pos.release(); neg.release()
        meta = {"SYNTHETIC_COMPOSITE": True, "purpose": "HELD-OUT software-level event test; NOT real phone use", "sequence": seq,
                "source": "MOT15 (CC BY-NC-SA 3.0, local research use only)", "phone_photo": photo, "fps": info["fps"], "size": list(size),
                "phone_visible_from_s": (f0 - 1) / info["fps"], "phone_visible_to_s": f1 / info["fps"], "target_gt_id": pid,
                "phone_frac_of_person_height": PHONE_FRAC, "n_frames": info["n"], "duration_s": info["n"] / info["fps"],
                "ground_truth_phone_boxes_by_frame": gtb}
        (OUT / f"{seq}_phone.json").write_text(json.dumps(meta))
        print(f"{seq}: window {meta['phone_visible_from_s']:.1f}-{meta['phone_visible_to_s']:.1f}s, {len(gtb)} GT phone frames, target GT id {pid}, photo {photo}")


if __name__ == "__main__":
    main()
