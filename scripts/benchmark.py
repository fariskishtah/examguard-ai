"""Measured end-to-end benchmark (real model, real clips). Writes artifacts/metrics/benchmark.json."""
from __future__ import annotations

import json
import platform
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import psutil

from examguard.config import Paths, Settings
from examguard.database.db import Database
from examguard.database.repository import Repository
from examguard.detection.detector import YoloDetector
from examguard.video.processor import VideoProcessor

ROOT = Path(__file__).resolve().parents[1]
CLIPS = {
    "street_1080p": ROOT / "data/samples/Video_Codec_Test_pedestrian_area_1080p25.y4m.webm",
    "classroom_1080p": ROOT / "data/samples/Student_studies_in_classroom_during_face_to_face_learning_in_Padang_Indonesia.webm",
}


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.p, self.peak_rss, self.cpu, self.stop = psutil.Process(), 0, [], False

    def run(self):
        self.p.cpu_percent(None)
        while not self.stop:
            self.peak_rss = max(self.peak_rss, self.p.memory_info().rss)
            self.cpu.append(self.p.cpu_percent(None))
            time.sleep(0.25)


def bench(video: Path, device: str, assist: bool, save_video: bool, imgsz: int = 640) -> dict:
    tmp = Path(tempfile.mkdtemp())
    paths = Paths(tmp).ensure()
    repo = Repository(Database(paths.db))
    s = Settings(device=device, phone_crop_assist=assist, imgsz=imgsz, save_annotated_video=save_video, notifications_enabled=False)
    det = YoloDetector(ROOT / "models/yolo11n.pt", 0.10, s.phone_conf, s.nms_iou, imgsz, device)
    det.detect(__import__("numpy").zeros((720, 1280, 3), "uint8"), 0, 0.0)  # warm-up (kernel compile / allocation)
    sm = Sampler(); sm.start()
    summ = VideoProcessor(s, paths, repo, detector=det).process(video)
    sm.stop = True; sm.join()
    repo.db.close()
    return {"clip": video.name[:30], "device": det.device, "imgsz": imgsz, "phone_crop_assist": assist, "annotated_video_written": save_video,
            "frames": summ.frames_read, "wall_s": round(summ.wall_seconds, 2), "end_to_end_fps": round(summ.avg_fps, 1),
            "model_ms_per_frame": round(summ.avg_infer_ms, 1), "peak_rss_mb": round(sm.peak_rss / 2**20),
            "mean_cpu_pct": round(sum(sm.cpu) / len(sm.cpu), 1) if sm.cpu else None, "status": summ.status}


def main():
    low = Path(tempfile.mkdtemp()) / "classroom_360p.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(CLIPS["classroom_1080p"]), "-vf", "scale=640:360", "-an", str(low)], check=True)
    CLIPS["classroom_360p"] = low
    rows = []
    for name, clip in CLIPS.items():
        for device, assist, save in [("mps", False, False), ("mps", True, False), ("mps", True, True), ("cpu", False, False)]:
            r = bench(clip, device, assist, save)
            r["clip"] = name
            rows.append(r)
            print(r, flush=True)
    out = {"hardware": {"machine": platform.machine(), "platform": platform.platform(), "cpu_count": psutil.cpu_count(),
                        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)},
           "note": "peak_rss/mean_cpu are whole-process (Python, torch, OpenCV, DB); cpu% can exceed 100 on multiple cores",
           "rows": rows}
    p = ROOT / "artifacts/metrics/benchmark.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
