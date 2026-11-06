"""Safety test for the cross-track incident MERGE rule: two SEPARATE phone events on two NEIGHBOURING people.

SYNTHETIC composite (real MOT15 footage + real CC0 phone photos), software-level only. Person A gets a phone during
window A; person B (the closest neighbour of A when A's window ends) gets a phone shortly AFTER. A correct system
raises TWO incidents. If the merge rule were too aggressive it would fold B's event into A's (one incident).
Usage:  python scripts/make_two_phone_fixture.py build SEQ     |   python scripts/make_two_phone_fixture.py eval
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_tracking_mot import MOT, load_gt, seq_info  # noqa: E402
from make_heldout_fixtures import paste, phone_crop, writer  # noqa: E402

PHONE_FRAC = 0.30  # larger than the event fixtures on purpose: this test targets the MERGE logic, not detector sensitivity

from examguard.config import Paths, Settings  # noqa: E402
from examguard.database.db import Database  # noqa: E402
from examguard.database.repository import Repository  # noqa: E402
from examguard.detection.detector import YoloDetector  # noqa: E402
from examguard.events.geometry import iou  # noqa: E402
from examguard.video.processor import VideoProcessor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/fixtures/twophone"
PHOTO_A, PHOTO_B = "Smartphone_in_Hand_Unsplash_.jpg", "Smartphone_Use.jpg"


def centre(b):
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


def find_pair(gt, n, fps, win_s=5.0, gap_s=1.5, min_h=90):
    """Pick (A, B, a0, a1, b0, b1): A visible during window A, B visible during window B, and B is A's closest
    neighbour at the hand-over, with non-overlapping boxes."""
    wa, gp = int(win_s * fps), int(gap_s * fps)
    vis = {f: {i: (x, y, w, h) for i, x, y, w, h in v if h >= min_h} for f, v in gt.items()}
    best = None
    for a0 in range(1, n - 2 * wa - gp):
        a1, b0, b1 = a0 + wa - 1, a0 + wa + gp, a0 + 2 * wa + gp - 1
        cand_a = set.intersection(*[set(vis.get(f, {})) for f in range(a0, a1 + 1, 2)]) if all(vis.get(f) for f in range(a0, a1 + 1, 2)) else set()
        for A in cand_a:
            for B in set.intersection(*[set(vis.get(f, {})) for f in range(b0, b1 + 1, 2)]) - {A} if all(vis.get(f) for f in range(b0, b1 + 1, 2)) else set():
                if A in vis.get(b0, {}) and B in vis.get(a1, {}):
                    ca, cb = centre(vis[a1][A]), centre(vis[a1][B])
                    d = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
                    wdt = vis[a1][A][2]
                    if d > 1.0 * wdt and (best is None or d < best[0]):
                        best = (d, A, B, a0, a1, b0, b1, d / wdt)
    return best


def build(seq):
    OUT.mkdir(parents=True, exist_ok=True)
    info, gt = seq_info(seq), load_gt(seq)
    best = find_pair(gt, info["n"], info["fps"])
    if best is None:
        raise SystemExit("no suitable neighbouring pair found")
    d, A, B, a0, a1, b0, b1, dw = best
    det = YoloDetector(ROOT / "models/yolo11m.pt", person_conf=0.5, phone_conf=0.3, imgsz=1280)
    ca, ma = phone_crop(det, ROOT / "data/samples" / PHOTO_A)
    cb, mb = phone_crop(det, ROOT / "data/samples" / PHOTO_B)
    size = (info["w"], info["h"]); w = writer(OUT / f"{seq}_twophone.mp4", info["fps"], size)
    gtA, gtB = {}, {}
    for k in range(1, info["n"] + 1):
        img = cv2.imread(str(MOT / seq / "img1" / f"{k:06d}{info['ext']}"))
        for pid, lo, hi, crop, mask, store in ((A, a0, a1, ca, ma, gtA), (B, b0, b1, cb, mb, gtB)):
            if lo <= k <= hi:
                for i, x, y, ww, hh in gt[k]:
                    if i == pid:
                        r = paste(img, crop, mask, x + ww / 2, y + 0.40 * hh, PHONE_FRAC * hh)
                        if r:
                            store[k - 1] = r
        w.write(img)
    w.release()
    meta = {"SYNTHETIC_COMPOSITE": True, "purpose": "merge-rule safety test (two separate events)", "sequence": seq, "fps": info["fps"],
            "A": {"gt_id": A, "from_s": (a0 - 1) / info["fps"], "to_s": a1 / info["fps"], "boxes": gtA},
            "B": {"gt_id": B, "from_s": (b0 - 1) / info["fps"], "to_s": b1 / info["fps"], "boxes": gtB},
            "neighbour_distance_in_person_widths_at_handover": round(dw, 2), "neighbour_distance_px": round(d, 1)}
    (OUT / f"{seq}_twophone.json").write_text(json.dumps(meta))
    print(f"{seq}: A=GT{A} {meta['A']['from_s']:.1f}-{meta['A']['to_s']:.1f}s  B=GT{B} {meta['B']['from_s']:.1f}-{meta['B']['to_s']:.1f}s  "
          f"neighbours {d:.0f}px apart ({dw:.1f} person widths)")


def evaluate():
    rows = []
    for mp in sorted(OUT.glob("*_twophone.json")):
        meta = json.loads(mp.read_text()); video = mp.with_suffix(".mp4"); fps = meta["fps"]
        for name, ov in (("merge ON (default)", {}), ("merge OFF", {"phone_merge_gap_seconds": 0})):
            tmp = Path(tempfile.mkdtemp()); paths = Paths(tmp).ensure(); repo = Repository(Database(paths.db))
            VideoProcessor(Settings(notifications_enabled=False, save_annotated_video=False, **ov), paths, repo).process(video)
            incs = sorted(repo.list_incidents(), key=lambda i: i.first_detected_at)
            res = {"A": 0, "B": 0, "other": 0}
            for i in incs:
                pb = (json.loads(i.evidence_json) or {}).get("phone_bbox") if i.evidence_json else None
                last_f = int(round(i.last_detected_at * fps)); hit = None
                for tag in ("A", "B"):
                    bx = {int(k): v for k, v in meta[tag]["boxes"].items()}
                    if bx:
                        near = min(bx, key=lambda f: abs(f - last_f))
                        if pb and abs(near - last_f) <= 3 and iou(tuple(pb), tuple(bx[near])) >= 0.3:
                            hit = tag
                res[hit or "other"] += 1
            rows.append({"sequence": meta["sequence"], "config": name, "incidents_total": len(incs), "matched_to_A": res["A"], "matched_to_B": res["B"],
                         "unmatched": res["other"], "both_events_reported_separately": res["A"] >= 1 and res["B"] >= 1,
                         "neighbour_distance_person_widths": meta["neighbour_distance_in_person_widths_at_handover"]})
            print(rows[-1], flush=True); repo.db.close()
    (ROOT / "artifacts/metrics/merge_safety_twophone.json").write_text(json.dumps({"note": "SYNTHETIC two-event composite; software-level", "rows": rows}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
    else:
        evaluate()
