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


CONFIGS = {  # name -> (weights, Settings overrides)
    "BEFORE: 640 + crop416 + bytetrack (shipped earlier)": ("yolo11n.pt", dict(imgsz=640, phone_crop_assist=True, phone_crop_imgsz=416, tracker_kind="bytetrack", phone_merge_gap_seconds=0)),
    "BEFORE (fast): 640, no assist, bytetrack": ("yolo11n.pt", dict(imgsz=640, phone_crop_assist=False, tracker_kind="bytetrack", phone_merge_gap_seconds=0)),
    "AFTER default: auto-native + crop640 + botsort + merge": ("yolo11n.pt", {}),
    "AFTER fast: auto-native, no assist": ("yolo11n.pt", dict(phone_crop_assist=False)),
    "AFTER accurate: yolo11s + crop640": ("yolo11s.pt", {}),
}


class TimedDetector(YoloDetector):
    """Records model time per frame (full-frame pass + crop-assist pass) so median/p95 latency can be reported."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.frame_ms: dict[int, float] = {}

    def detect(self, frame_bgr, frame_index, timestamp_s):
        t = time.perf_counter()
        out = super().detect(frame_bgr, frame_index, timestamp_s)
        self.frame_ms[frame_index] = self.frame_ms.get(frame_index, 0.0) + (time.perf_counter() - t) * 1000
        return out

    def detect_phones_in_crops(self, frame_bgr, boxes, frame_index, timestamp_s, **kw):
        t = time.perf_counter()
        out = super().detect_phones_in_crops(frame_bgr, boxes, frame_index, timestamp_s, **kw)
        self.frame_ms[frame_index] = self.frame_ms.get(frame_index, 0.0) + (time.perf_counter() - t) * 1000
        return out


def bench(video: Path, weights: str, overrides: dict, device: str = "mps") -> dict:
    tmp = Path(tempfile.mkdtemp())
    paths = Paths(tmp).ensure()
    repo = Repository(Database(paths.db))
    s = Settings(device=device, save_annotated_video=False, notifications_enabled=False, weights_path=str(ROOT / "models" / weights), **overrides)
    det = TimedDetector(ROOT / "models" / weights, 0.10, s.phone_conf, s.nms_iou, 640, device)
    det.detect(__import__("numpy").zeros((720, 1280, 3), "uint8"), 0, 0.0)  # warm-up (kernel compile / allocation)
    det.frame_ms.clear()
    sm = Sampler(); sm.start()
    summ = VideoProcessor(s, paths, repo, detector=det).process(video)
    sm.stop = True; sm.join()
    repo.db.close()
    return {"clip": video.name[:30], "device": det.device, "weights": weights, "frames": summ.frames_read, "wall_s": round(summ.wall_seconds, 2),
            "end_to_end_fps": round(summ.avg_fps, 1), "model_ms_per_frame": round(summ.avg_infer_ms, 1), "model_ms_median": round(float(__import__("numpy").median(list(det.frame_ms.values()))), 1),
            "model_ms_p95": round(float(__import__("numpy").percentile(list(det.frame_ms.values()), 95)), 1), "imgsz_used": det.imgsz, "peak_rss_mb": round(sm.peak_rss / 2**20),
            "mean_cpu_pct": round(sum(sm.cpu) / len(sm.cpu), 1) if sm.cpu else None, "status": summ.status, "tracks": summ.n_tracks}


def main():
    adl = ROOT / "data/fixtures/final/ADL-Rundle-8_clean.mp4"
    clips = {"street_1080p": CLIPS["street_1080p"], "ADL-Rundle-8_1080p": adl, "classroom_1080p": CLIPS["classroom_1080p"]}
    rows = []
    for cname, clip in clips.items():
        for name, (weights, ov) in CONFIGS.items():
            r = bench(clip, weights, ov); r["clip"], r["config"] = cname, name
            rows.append(r); print(r, flush=True)
    out = {"hardware": {"machine": platform.machine(), "platform": platform.platform(), "cpu_count": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)},
           "note": "peak_rss/mean_cpu are whole-process (Python, torch, OpenCV, DB); cpu% can exceed 100 on multiple cores; annotated video NOT written",
           "rows": rows}
    p = ROOT / "artifacts/metrics/benchmark_final.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
