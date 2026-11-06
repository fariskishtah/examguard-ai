"""Process a video without the GUI and print a JSON summary (uses an isolated runtime dir)."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from examguard.config import Paths, Settings
from examguard.database.db import Database
from examguard.database.repository import Repository
from examguard.video.processor import VideoProcessor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--home", required=True, help="runtime dir (db, evidence, outputs)")
    ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--phone-conf", type=float, default=0.25); ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--region", help="normalized polygon as JSON, e.g. [[0,0],[0.5,0],[0.5,1],[0,1]]")
    ap.add_argument("--support", type=int, default=5)
    a = ap.parse_args()
    root = Path(a.home)
    s = Settings(imgsz=a.imgsz, frame_stride=a.stride, phone_conf=a.phone_conf, save_annotated_video=not a.no_video,
                 notifications_enabled=False, weights_path=str(Path(__file__).resolve().parents[1] / "models/yolo11n.pt"),
                 phone_min_support_frames=a.support, phone_window_frames=max(15, a.support),
                 region_polygon=json.loads(a.region) if a.region else [])
    paths = Paths(root).ensure()
    repo = Repository(Database(paths.db))
    summ = VideoProcessor(s, paths, repo).process(a.video)
    print(json.dumps(asdict(summ), indent=2, default=str))
    for i in repo.list_incidents():
        print(f"incident #{i.id} {i.event_type} track={i.anonymous_track_id} t={i.first_detected_at:.2f}-{i.last_detected_at:.2f}s "
              f"conf={i.confidence:.2f} state={i.lifecycle_state} review={i.review_status}\n   {i.screenshot_path}")
    sys.exit(0 if summ.status == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
