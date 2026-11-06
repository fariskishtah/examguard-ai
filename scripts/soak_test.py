"""Long-video stability test (real model, real VideoProcessor).

The long video is the CC0 street footage + the synthetic phone fixture LOOPED (so it is long but not diverse -
it tests resource stability, not accuracy). Measures RSS/CPU over time, FPS per minute, pause/resume, cancellation,
alert cooldown, evidence files, DB persistence and the exported video. Nothing personal is stored.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import psutil

from examguard.config import Paths, Settings
from examguard.database.db import Database
from examguard.database.repository import Repository
from examguard.video.processor import ProcessingControl, VideoProcessor

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "data/fixtures/synthetic_phone_pedestrian.mp4"


def make_long(minutes: float, dest: Path) -> int:
    loops = max(1, int(round(minutes * 60 / 15.0)))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-stream_loop", str(loops - 1), "-i", str(FIX), "-c", "copy", str(dest)], check=True)
    return loops


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.p, self.rows, self.stop = psutil.Process(), [], False

    def run(self):
        self.p.cpu_percent(None); t0 = time.time()
        while not self.stop:
            self.rows.append((round(time.time() - t0, 1), self.p.memory_info().rss / 2**20, self.p.cpu_percent(None)))
            time.sleep(1.0)


def run_once(video: Path, minutes: float, pause_at: float | None, stop_at: float | None, tag: str) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"eg_soak_{tag}_"))
    paths = Paths(tmp).ensure(); repo = Repository(Database(paths.db))
    s = Settings(notifications_enabled=False, save_annotated_video=True)
    ctl = ProcessingControl()
    frames_t: list[tuple[float, int]] = []
    alerts: list[float] = []
    pause_info = {}
    t_start = time.perf_counter()

    def emit(kind, payload):
        if kind == "frame":
            frames_t.append((time.perf_counter() - t_start, payload.frame_index))
        elif kind == "alert":
            alerts.append(payload.timestamp_s)

    def pauser():
        time.sleep(pause_at)
        n0 = frames_t[-1][1] if frames_t else 0
        ctl.pause(); time.sleep(6.0)
        n1 = frames_t[-1][1] if frames_t else 0
        pause_info.update({"paused_at_s": pause_at, "paused_for_s": 6.0, "frames_advanced_during_pause": n1 - n0})
        ctl.resume()

    if pause_at:
        threading.Thread(target=pauser, daemon=True).start()
    if stop_at:
        threading.Timer(stop_at, ctl.stop).start()
    sm = Sampler(); sm.start()
    summ = VideoProcessor(s, paths, repo, emit=emit).process(video, ctl)
    sm.stop = True; sm.join()
    # ---- checks
    incs = repo.list_incidents(limit=100000)
    shots = [Path(i.screenshot_path).is_file() and Path(i.raw_screenshot_path).is_file() for i in incs]
    sess = repo.get_session(summ.session_id)
    vid_ok, vid_frames = False, 0
    if summ.annotated_video and Path(summ.annotated_video).is_file():
        cap = cv2.VideoCapture(summ.annotated_video); vid_ok = cap.isOpened()
        vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    rss = [r[1] for r in sm.rows]; cpu = [r[2] for r in sm.rows]
    per_min = []
    if frames_t:
        for m in range(int(frames_t[-1][0] // 60) + 1):
            a = [f for t, f in frames_t if m * 60 <= t < (m + 1) * 60]
            if len(a) > 1:
                per_min.append(round((a[-1] - a[0]) / max(frames_t[[f for _, f in frames_t].index(a[-1])][0] - frames_t[[f for _, f in frames_t].index(a[0])][0], 1e-6), 1))
    third = max(len(rss) // 3, 1)
    gaps = sorted(b - a for a, b in zip(sorted(i.first_detected_at for i in incs), sorted(i.first_detected_at for i in incs)[1:]))
    out = {"tag": tag, "video_minutes_nominal": minutes, "status": summ.status, "frames_read": summ.frames_read, "wall_s": round(summ.wall_seconds, 1),
           "avg_fps": round(summ.avg_fps, 1), "fps_per_minute_window": per_min, "rss_mb_start": round(rss[2] if len(rss) > 2 else rss[0]), "rss_mb_first_third_mean": round(sum(rss[:third]) / third),
           "rss_mb_last_third_mean": round(sum(rss[-third:]) / third), "rss_mb_peak": round(max(rss)), "rss_mb_end": round(rss[-1]), "mean_cpu_pct": round(sum(cpu) / len(cpu), 1),
           "incidents": len(incs), "alerts_emitted": len(alerts), "evidence_files_all_present": all(shots) if shots else None, "min_gap_between_incident_starts_s": round(gaps[0], 2) if gaps else None,
           "db_session_status": sess.processing_status, "db_frame_count": sess.frame_count, "db_tracks": repo.count_tracks(summ.session_id), "tracks_created": summ.n_tracks,
           "annotated_video_playable": vid_ok, "annotated_video_frames": vid_frames, "pause": pause_info or None, "error": summ.error,
           "all_incidents_resolved": all(i.lifecycle_state == "RESOLVED" for i in incs)}
    repo.db.close()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)  # do not keep derived footage/evidence
    return out


def tracker_cleanup_probe(n_frames: int = 20000) -> dict:
    """Run the tracker alone for many frames on synthetic moving boxes; verify internal state stays bounded."""
    import numpy as np
    from examguard.detection.detector import Detection
    from examguard.tracking.tracker import PersonTracker
    tr = PersonTracker(0.5, 0.1, 0.5, 60, 0.8, kind="botsort")  # the production default
    rng = np.random.default_rng(0)
    sizes, times = [], []
    for k in range(n_frames):
        dets = []
        for j in range(8):  # people constantly entering/leaving: ids churn
            x = (k * 3 + j * 150) % 1800
            if (k // 200 + j) % 3:  # each person appears/disappears periodically
                dets.append(Detection("person", 0, 0.9, (x, 200 + 20 * j, x + 60, 360 + 20 * j), k, k / 25, "t"))
        t = time.perf_counter(); tr.update(dets); times.append(time.perf_counter() - t)
        if k % 2000 == 0:
            t_ = tr._t
            sizes.append({"frame": k, "tracked": len(t_.tracked_stracks), "lost": len(t_.lost_stracks), "removed": len(getattr(t_, "removed_stracks", []))})
    return {"frames": n_frames, "state_samples": sizes, "ms_per_frame_first_1000": round(1000 * float(np.mean(times[:1000])), 3), "ms_per_frame_last_1000": round(1000 * float(np.mean(times[-1000:])), 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0); ap.add_argument("--out", default=str(ROOT / "artifacts/metrics/soak_test.json"))
    a = ap.parse_args()
    tmpv = Path(tempfile.mkdtemp(prefix="eg_longvideo_")) / "long.mp4"
    loops = make_long(a.minutes, tmpv)
    cap = cv2.VideoCapture(str(tmpv)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    print(f"long video: {loops} loops, {n} frames, {n / fps / 60:.1f} min, {tmpv.stat().st_size / 1e6:.0f} MB", flush=True)
    res = {"note": "LOOPED CC0 street footage + synthetic phone fixture: tests resource stability, not accuracy", "long_video": {"loops": loops, "frames": n, "minutes": round(n / fps / 60, 2)}}
    res["tracker_cleanup_probe"] = tracker_cleanup_probe(); print("tracker probe:", json.dumps(res["tracker_cleanup_probe"]), flush=True)
    res["cancel_run"] = run_once(tmpv, a.minutes, None, 40.0, "cancel"); print("cancel:", json.dumps({k: v for k, v in res["cancel_run"].items() if k != "fps_per_minute_window"}), flush=True)
    res["full_run"] = run_once(tmpv, a.minutes, 60.0, None, "full"); print("full:", json.dumps(res["full_run"]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    import shutil
    shutil.rmtree(tmpv.parent, ignore_errors=True)
    print("written", a.out)
