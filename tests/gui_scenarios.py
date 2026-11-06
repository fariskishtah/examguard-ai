"""GUI scenarios, each run in its OWN process (one Tk root per process).

Why: Tk 9.0.4 on macOS segfaults (Tk_MacOSXGetTkWindow in showRootWindow) when a second Tk() root is
shown after an earlier one was destroyed in the same process. The shipped app creates exactly one root,
so it is unaffected; the tests therefore isolate scenarios in subprocesses (which also makes the
"restart" scenario a real process restart).

Usage: python tests/gui_scenarios.py <scenario> <home_dir> [incident_id]
The detector is a scripted FakeDetector: this exercises the GUI + pipeline plumbing, not model accuracy.
"""
import json
import sys
import time
from pathlib import Path
from tkinter import messagebox

sys.path.insert(0, str(Path(__file__).parent))
from conftest import write_video  # noqa: E402
from test_processor import FakeDetector  # noqa: E402

from examguard.config import Settings  # noqa: E402

for _n in ("showerror", "showinfo", "showwarning"):
    setattr(messagebox, _n, lambda *a, **k: None)
messagebox.askyesno = lambda *a, **k: True


def pump(app, cond, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.update()
        if cond():
            return True
        time.sleep(0.01)
    return False


def make_app(home):
    from examguard.ui.app import ExamGuardApp
    app = ExamGuardApp(Path(home))
    app.ctl._get_detector = lambda: FakeDetector()
    app.ctl.settings = Settings(**{**app.ctl.settings.to_dict(), "phone_min_support_frames": 4, "phone_window_frames": 8, "phone_min_duration_seconds": 0.0,
                                   "resolve_after_seconds": 0.5})
    return app


def e2e(home, _=None):
    app = make_app(home)
    mon = app.views["monitor"]
    app.navigate("monitor")
    assert str(mon.btn_start["state"]) == "disabled"
    assert mon.load_video(write_video(Path(home) / "exam.mp4", n=100))
    assert str(mon.btn_start["state"]) == "disabled", "start must stay disabled until authorisation is confirmed"
    mon.auth_var.set(True)
    mon._update_buttons()
    assert str(mon.btn_start["state"]) == "normal"
    mon.start()
    assert pump(app, lambda: app.ctl.worker.finished.is_set() and not mon._polling), "processing did not finish"
    app.update()
    assert app.ctl.repo.stats()["incidents"] == 1
    assert len(mon.alerts.get_children()) == 1, "alert centre not populated"
    assert app.toast.count >= 1, "no non-blocking toast"
    app.navigate("incidents")
    inv = app.views["incidents"]
    inc_id = int(inv.tree.get_children()[0])
    inv.tree.selection_set(str(inc_id))
    app.update()
    assert inv.current_id == inc_id and inv.viewer._img is not None, "evidence screenshot not shown"
    inv.note.insert("1.0", "looked like a calculator")
    inv.review("DISMISSED")
    assert app.ctl.repo.get_incident(inc_id).review_status == "DISMISSED"
    app.navigate("dashboard")
    d = app.views["dashboard"].cards
    assert d["dismissed"].var.get() == "1" and d["pending"].var.get() == "0"
    app.on_close()
    return {"incident_id": inc_id}


def restart(home, inc_id):
    app = make_app(home)
    try:
        d = app.views["dashboard"].cards
        assert d["videos"].var.get() == "1" and d["dismissed"].var.get() == "1", "dashboard lost data after restart"
        app.show_incident(int(inc_id))
        app.update()
        assert app.views["incidents"].viewer._img is not None, "evidence not reopenable after restart"
        hist = app.views["incidents"].hist.get("1.0", "end")
        assert "calculator" in hist and "Dismissed" in hist, "review history lost"
    finally:
        app.on_close()
    return {}


def misc(home, _=None):
    app = make_app(home)
    try:
        mon = app.views["monitor"]
        bad = Path(home) / "bad.mp4"
        bad.write_bytes(b"zzz" * 20)
        assert mon.load_video(bad) is False and mon.video is None, "corrupted video accepted"
        sid = app.ctl.repo.create_session("x.mp4", "a" * 64)
        iid = app.ctl.repo.create_incident(sid, 1, "PHONE_VISIBLE_NEAR_TRACK", 1.0, 3, 0.5, None, "/no/such.jpg", None)
        app.show_incident(iid)
        app.update()
        assert app.views["incidents"].viewer._img is None, "missing evidence should show placeholder"
        sv = app.views["settings"]
        sv.vars["phone_conf"].set("7")
        assert sv.save() is False and "Not saved" in sv.msg.get()
        sv.vars["phone_conf"].set("abc")
        assert sv.save() is False
        sv.vars["phone_conf"].set("0.3")
        sv.vars["phone_min_support_frames"].set("6")
        assert sv.save() is True and app.ctl.settings.phone_conf == 0.3
        # region drawing is stored normalised -> resizing cannot make it drift
        app.navigate("monitor")
        mon.load_video(write_video(Path(home) / "v.mp4", n=5, size=(320, 180)))
        app.update()
        mon.toggle_draw()
        ox, oy, w, h = mon.canvas.image_rect()
        for fx, fy in ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9)):
            mon.canvas.event_generate("<Button-1>", x=int(ox + fx * w), y=int(oy + fy * h))
            app.update()
        assert len(mon.region) == 3 and abs(mon.region[0][0] - 0.1) < 0.02
        before = list(mon.region)
        app.geometry("900x700")
        app.update()
        assert mon.region == before
        mon.toggle_draw()
        assert len(app.ctl.settings.region_polygon) == 3
    finally:
        app.on_close()
    return {}


if __name__ == "__main__":
    name, home = sys.argv[1], sys.argv[2]
    result = {"e2e": e2e, "restart": restart, "misc": misc}[name](home, sys.argv[3] if len(sys.argv) > 3 else None)
    print("RESULT " + json.dumps(result))
