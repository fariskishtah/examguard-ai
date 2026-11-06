"""Frame-level phone detection on synthetic fixtures (design/diagnosis aid; synthetic, not real phone use).

Per fixture video: full-frame at several input sizes, and person-crop union at several crop sizes, scored against the
exact pasted-phone boxes (IoU>=0.3). Writes a JSON table. Fixtures used here become DESIGN data, not held-out data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from examguard.detection.detector import PERSON, PHONE, YoloDetector, merge_detections
from examguard.events.geometry import iou
from examguard.video.reader import iter_frames, probe_video

ROOT = Path(__file__).resolve().parents[1]


def run(fixture_dir: Path, weights: str, full_sizes, crop_cfgs, stride=2):
    det = YoloDetector(ROOT / weights, person_conf=0.1, phone_conf=0.05, imgsz=640)
    rows = []
    for meta_p in sorted(fixture_dir.glob("*_phone.json")):
        meta = json.loads(meta_p.read_text()); gt = {int(k): tuple(v) for k, v in meta["ground_truth_phone_boxes_by_frame"].items()}
        video = meta_p.with_name(meta_p.name.replace(".json", ".mp4")); info = probe_video(video)
        frames = [(i, f) for i, t, f in iter_frames(info, stride) if f is not None and i % stride == 0 and i in gt]
        for fs in full_sizes:
            det.imgsz = fs
            for label, crop in [("full", None)] + [(f"full+crop{c}", c) for c in crop_cfgs]:
                tp_f = tp_d = fp_d = 0
                for i, f in frames:
                    d = det.detect(f, i, 0.0)
                    ph = [x for x in d if x.label == PHONE]
                    if crop:
                        pe = [x for x in d if x.label == PERSON and x.confidence >= 0.5]
                        ph = merge_detections(ph, det.detect_phones_in_crops(f, [p.bbox for p in pe], i, 0.0, imgsz=crop, margin=0.15))
                    ph = [x for x in ph if x.confidence >= 0.25]
                    hit = [x for x in ph if iou(x.bbox, gt[i]) >= 0.3]
                    tp_d += len(hit); fp_d += len(ph) - len(hit); tp_f += bool(hit)
                rows.append({"seq": meta["sequence"], "full_imgsz": fs, "mode": label, "frames": len(frames), "frame_recall": round(tp_f / len(frames), 3),
                             "precision": round(tp_d / (tp_d + fp_d), 3) if tp_d + fp_d else None, "fp_dets": fp_d})
                print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=str(ROOT / "data/fixtures/heldout")); ap.add_argument("--weights", default="models/yolo11n.pt")
    ap.add_argument("--out", required=True); a = ap.parse_args()
    rows = run(Path(a.dir), a.weights, (640, 1280, 1920), (416, 640, 960))
    Path(a.out).write_text(json.dumps(rows, indent=1))
