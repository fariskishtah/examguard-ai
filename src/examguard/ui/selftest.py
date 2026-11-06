"""`examguard --selftest VIDEO`: drive the real GUI + pipeline once (used to verify the packaged app)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def run_selftest(home: Path, video: Path) -> int:
    """Drive the real GUI + pipeline once, then review an incident and run every export. Used to verify the packaged app."""
    import json as _json
    from examguard.reports import export as ex
    from examguard.ui.app import ExamGuardApp
    from examguard.utils.resources import find_weights

    app = ExamGuardApp(home)
    out = {"frozen": bool(getattr(sys, "frozen", False)), "weights": str(find_weights(app.ctl.settings.weights_path)),
           "db": str(app.ctl.paths.db), "tracker": app.ctl.settings.tracker_kind}
    try:
        mon = app.views["monitor"]
        app.navigate("monitor")
        if not mon.load_video(video):
            out["error"] = "video rejected"
            return _finish(out, 2)
        mon.auth_var.set(True)
        mon._update_buttons()
        mon.start()
        t0 = time.time()
        while time.time() - t0 < 1800 and not (app.ctl.worker and app.ctl.worker.finished.is_set() and not mon._polling):
            app.update()
            time.sleep(0.01)
        app.update()
        repo = app.ctl.repo
        st = repo.stats()
        s = repo.list_sessions()[0] if st["videos"] else None
        out.update({"stats": {k: v for k, v in st.items() if k != "last_session"},
                    "session_status": s.processing_status if s else None, "detector": s.detector_version if s else None,
                    "tracker_version": s.tracker_version if s else None, "frames": s.frame_count if s else None})
        inc = repo.list_incidents()
        out["evidence_files_exist"] = all(Path(i.screenshot_path).is_file() and Path(i.raw_screenshot_path).is_file() for i in inc)
        # human review through the repository the GUI uses, then every export
        if inc:
            repo.review(inc[0].id, "NEEDS_MORE_EVIDENCE", "selftest reviewer note")
            exp = app.ctl.paths.exports
            incs = repo.list_incidents()
            ex.export_csv(repo, incs, exp / "incidents.csv"); ex.export_json(repo, incs, exp / "incidents.json")
            n_shots = ex.export_screenshots(incs, exp / "shots")
            ex.export_report_html(repo, s.id, exp / "report.html")
            vid = ex.export_annotated_video(repo.get_session(s.id), exp)
            data = _json.loads((exp / "incidents.json").read_text())
            out["exports"] = {"csv_bytes": (exp / "incidents.csv").stat().st_size, "json_incidents": len(data["incidents"]),
                              "json_review_history": [len(i["review_history"]) for i in data["incidents"]], "screenshot_files": n_shots,
                              "report_bytes": (exp / "report.html").stat().st_size, "video": str(vid) if vid else None}
            if vid:
                import cv2
                cap = cv2.VideoCapture(str(vid))
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("latin1")
                out["exports"].update({"video_fourcc": fourcc, "video_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "video_playable": cap.isOpened()})
                cap.release()
        ok = bool(s and s.processing_status == "COMPLETED")
        return _finish(out, 0 if ok else 1)
    finally:
        app.on_close()


def run_verify_persistence(home: Path) -> int:
    """Second launch on the same data folder: records, review history and evidence must have survived a restart."""
    from examguard.ui.app import ExamGuardApp

    app = ExamGuardApp(home)
    out = {}
    try:
        repo = app.ctl.repo
        st = repo.stats()
        inc = repo.list_incidents()
        app.show_incident(inc[0].id) if inc else None
        app.update()
        iv = app.views["incidents"]
        out.update({"stats": {k: v for k, v in st.items() if k != "last_session"}, "needs_more": st["needs_more"],
                    "review_notes": [r.reviewer_note for r in repo.reviews_for(inc[0].id)] if inc else [],
                    "evidence_viewer_loaded_image": bool(inc and iv.viewer._img is not None),
                    "dashboard_cards": {k: c.var.get() for k, c in app.views["dashboard"].cards.items()}})
        ok = bool(inc and out["needs_more"] >= 1 and "selftest reviewer note" in out["review_notes"] and out["evidence_viewer_loaded_image"])
        return _finish(out, 0 if ok else 1)
    finally:
        app.on_close()


def _finish(out: dict, code: int) -> int:
    print("SELFTEST " + json.dumps(out, default=str), flush=True)
    return code


def run_demo(home: Path, video: Path) -> int:
    """Drive the real GUI to showcase states and WAIT at each so the window can be captured.

    Handshake: the stage name is written to <home>/stage.txt; the driver continues when <home>/next appears.
    """
    from examguard.ui.app import ExamGuardApp

    app = ExamGuardApp(home)
    app.geometry("1360x860+60+60")
    stage_file, next_file = home / "stage.txt", home / "next"

    def pump(cond, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout and not cond():
            app.update()
            time.sleep(0.01)
        app.update()

    def stage(name):
        next_file.unlink(missing_ok=True)
        stage_file.write_text(name)
        pump(next_file.exists, timeout=900)

    try:
        mon = app.views["monitor"]
        app.navigate("monitor")
        mon.load_video(video)
        mon.auth_var.set(True)
        mon._update_buttons()
        stage("loaded")
        mon.start()
        w = app.ctl.worker
        seen = {"idx": 0}

        def frame_ready():
            return mon.progress["value"] > 12
        pump(frame_ready)
        w.pause(); mon._paused = True; mon.btn_pause.configure(text="▶  Resume")
        time.sleep(0.3); app.update()
        stage("tracking")
        w.resume(); mon._paused = False; mon.btn_pause.configure(text="⏸  Pause")
        pump(lambda: app.ctl.repo.stats()["incidents"] >= 1 or w.finished.is_set())
        time.sleep(0.6); app.update()
        w.pause(); mon._paused = True
        time.sleep(0.3); app.update()
        stage("alert")
        w.resume(); mon._paused = False
        pump(lambda: w.finished.is_set() and not mon._polling)
        app.navigate("dashboard"); app.update()
        stage("dashboard")
        app.navigate("incidents"); app.update()
        inv = app.views["incidents"]
        kids = inv.tree.get_children()
        if kids:
            inv.tree.selection_set(kids[0]); app.update()
        stage("incidents")
        if kids:
            inv.note.insert("1.0", "Reviewed evidence frame; context unclear.")
            inv.review("NEEDS_MORE_EVIDENCE"); app.update()
        stage("reviewed")
        app.navigate("settings"); app.update()
        stage("settings")
        return 0
    finally:
        app.on_close()
