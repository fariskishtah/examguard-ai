"""Build a clearly-labelled SYNTHETIC composite test video.

A real CC0 phone photograph is pasted (feathered) at a fixed position inside one tracked person's
box for a known time window. This is a *software integration fixture*: it exercises
detect -> track -> associate -> confirm -> evidence -> DB with the real model, but it is NOT real
footage of phone use and NOT evidence of real-world accuracy. A sidecar JSON records that.

Usage: python scripts/make_synthetic_fixture.py SRC.webm OUT.mp4 --start 4 --end 10 [--phone-frac 0.22]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from examguard.detection.detector import PERSON, YoloDetector
from examguard.tracking.tracker import PersonTracker
from examguard.video.reader import iter_frames, probe_video

ROOT = Path(__file__).resolve().parents[1]
PHONE_PHOTO = ROOT / "data/samples/Black_smartphone_in_hand_Unsplash_.jpg"
# phone outline in a 1200x800 rendering of the photo (hand-annotated polygon + bounding rect)
POLY = np.array([(470, 150), (540, 135), (690, 140), (720, 240), (790, 590), (560, 660), (500, 330)], np.float32)


def load_patch() -> tuple[np.ndarray, np.ndarray]:
    img = cv2.resize(cv2.imread(str(PHONE_PHOTO)), (1200, 800))
    x0, y0 = POLY.min(0).astype(int) - 6
    x1, y1 = POLY.max(0).astype(int) + 6
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [POLY.astype(np.int32)], 255)
    mask = cv2.GaussianBlur(mask, (0, 0), 3)
    return img[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def paste(frame: np.ndarray, patch: np.ndarray, mask: np.ndarray, cx: int, cy: int, height: int) -> tuple[int, int, int, int]:
    s = height / patch.shape[0]
    p = cv2.resize(patch, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
    h, w = p.shape[:2]
    x, y = cx - w // 2, cy - h // 2
    H, W = frame.shape[:2]
    xa, ya, xb, yb = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if xb <= xa or yb <= ya:
        return (0, 0, 0, 0)
    sub = frame[ya:yb, xa:xb].astype(np.float32)
    frame[ya:yb, xa:xb] = (sub * (1 - m[ya - y:yb - y, xa - x:xb - x]) + p[ya - y:yb - y, xa - x:xb - x] * m[ya - y:yb - y, xa - x:xb - x]).astype(np.uint8)
    return (xa, ya, xb, yb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--start", type=float, required=True); ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--phone-frac", type=float, default=0.22, help="phone height as fraction of person box height")
    ap.add_argument("--max-seconds", type=float, default=15.0)
    a = ap.parse_args()
    info = probe_video(a.src)
    det = YoloDetector(ROOT / "models/yolo11n.pt")
    tracker = PersonTracker()
    patch, mask = load_patch()
    # pass 1: pick the track that is visible throughout the window the longest
    frames, boxes = [], []
    for idx, ts, fr in iter_frames(info):
        if ts > a.max_seconds:
            break
        persons = [d for d in det.detect(fr, idx, ts) if d.label == PERSON]
        boxes.append((ts, tracker.update(persons, fr)))
        frames.append(fr)
    cover: dict[int, int] = {}
    for ts, trs in boxes:
        if a.start <= ts <= a.end:
            for t in trs:
                cover[t.track_id] = cover.get(t.track_id, 0) + 1
    target = max(cover, key=cover.get)
    writer = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"avc1"), info.fps, (info.width, info.height))
    placed = 0
    gt: dict[int, list[int]] = {}
    for fi, (fr, (ts, trs)) in enumerate(zip(frames, boxes)):
        if a.start <= ts <= a.end:
            for t in trs:
                if t.track_id == target:
                    x1, y1, x2, y2 = t.bbox
                    bh = y2 - y1
                    cx, cy = int((x1 + x2) / 2), int(y1 + 0.38 * bh)
                    rect = paste(fr, patch, mask, cx, cy, max(12, int(a.phone_frac * bh)))
                    if rect[2]:
                        placed += 1
                        gt[fi] = list(rect)  # exact pasted-phone box (ground truth for this SYNTHETIC fixture)
        writer.write(fr)
    writer.release()
    meta = {"SYNTHETIC_COMPOSITE": True, "purpose": "software integration test only; not real phone-use footage",
            "source_video": Path(a.src).name, "phone_photo": PHONE_PHOTO.name, "licenses": "see data/samples/MANIFEST.json",
            "phone_visible_from_s": a.start, "phone_visible_to_s": a.end, "phone_frac_of_person_height": a.phone_frac,
            "frames_with_phone_pasted": placed, "ground_truth_phone_boxes_by_frame": gt, "fps": info.fps, "size": [info.width, info.height],
            "ground_truth_event": "PHONE_VISIBLE_NEAR_TRACK", "target_track_note": "target chosen by ByteTrack on source; ids not guaranteed equal in later runs"}
    Path(str(a.out) + ".fixture.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
