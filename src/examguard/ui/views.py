"""Application views (Dashboard, Monitor, Incidents, Settings)."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from examguard.config import DISCLAIMER, SUPPORTED_VIDEO_EXTENSIONS, ConfigError, Settings
from examguard.database.models import REVIEW_CONFIRMED, REVIEW_DISMISSED, REVIEW_MORE, REVIEW_STATUSES
from examguard.database.repository import IncidentFilter
from examguard.reports import export as ex
from examguard.ui import theme as T
from examguard.ui.widgets import ImageCanvas, StatCard
from examguard.video.annotate import fmt_ts
from examguard.video.reader import VideoError, probe_video


def mmss(ts: float) -> str:
    return fmt_ts(ts)[:-4]


def open_in_system(path: str) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
    elif sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606
    else:
        subprocess.Popen(["xdg-open", path])


def make_tree(parent, columns: list[tuple[str, str, int]], height: int = 10) -> ttk.Treeview:
    tree = ttk.Treeview(parent, columns=[c[0] for c in columns], show="headings", height=height, selectmode="browse")
    for cid, title, width in columns:
        tree.heading(cid, text=title)
        tree.column(cid, width=width, anchor="w", stretch=True)
    for status, color in T.STATUS_COLORS.items():
        tree.tag_configure(status, foreground=color)
    return tree


def disclaimer_banner(parent) -> ttk.Label:
    return ttk.Label(parent, text="⚠  " + DISCLAIMER, style="Warn.TLabel", wraplength=1000)


# =============================================================================== Dashboard
class DashboardView(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=24)
        self.app = app
        ttk.Label(self, text="Dashboard", style="H1.TLabel").pack(anchor="w")
        ttk.Label(self, text="Live figures from the local incident database", style="Muted.TLabel").pack(anchor="w", pady=(0, 14))
        disclaimer_banner(self).pack(fill="x", pady=(0, 16))
        row = ttk.Frame(self)
        row.pack(fill="x")
        self.cards = {k: StatCard(row, t, c) for k, t, c in [
            ("videos", "Videos processed", T.BLUE), ("incidents", "Detected events", T.AMBER),
            ("pending", "Awaiting review", T.AMBER), ("dismissed", "Dismissed", T.MUTED)]}
        for i, c in enumerate(self.cards.values()):
            c.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 12, 0))
            row.columnconfigure(i, weight=1, uniform="c")
        row2 = ttk.Frame(self)
        row2.pack(fill="x", pady=12)
        self.cards2 = {k: StatCard(row2, t, c) for k, t, c in [
            ("confirmed", "Confirmed observations", T.RED), ("needs_more", "Needs more evidence", T.BLUE_HI)]}
        for i, c in enumerate(self.cards2.values()):
            c.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 12, 0))
        self.status_card = ttk.Frame(row2, style="Card.TFrame", padding=(18, 14))
        self.status_card.grid(row=0, column=2, columnspan=2, sticky="nsew", padx=(12, 0))
        row2.columnconfigure((0, 1, 2, 3), weight=1, uniform="d")
        ttk.Label(self.status_card, text="PROCESSING STATUS", style="PanelMuted.TLabel").pack(anchor="w")
        self.proc_var, self.last_var = tk.StringVar(), tk.StringVar()
        ttk.Label(self.status_card, textvariable=self.proc_var, style="Panel.TLabel", font=T.FONT_H2).pack(anchor="w", pady=(6, 2))
        ttk.Label(self.status_card, textvariable=self.last_var, style="PanelMuted.TLabel", wraplength=420).pack(anchor="w")
        ttk.Label(self, text="Recent events", style="H2.TLabel").pack(anchor="w", pady=(10, 6))
        self.tree = make_tree(self, [("id", "ID", 50), ("when", "Recorded (UTC)", 150), ("video", "Video", 220), ("track", "Track", 70),
                                     ("event", "Event", 210), ("conf", "Conf.", 60), ("status", "Review status", 160)], 8)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self._open())

    def _open(self):
        sel = self.tree.selection()
        if sel:
            self.app.show_incident(int(sel[0]))

    def refresh(self) -> None:
        st = self.app.ctl.repo.stats()
        for k, c in {**self.cards, **self.cards2}.items():
            c.set(st[k])
        self.proc_var.set("Processing…" if self.app.ctl.running else "Idle")
        ls = st["last_session"]
        self.last_var.set(f"Last session: {ls[0]} - {ls[1].title()} ({ls[2]:%Y-%m-%d %H:%M} UTC)" if ls else "No videos processed yet.")
        self.tree.delete(*self.tree.get_children())
        for i in self.app.ctl.repo.list_incidents(limit=8):
            self.tree.insert("", "end", iid=str(i.id), tags=(i.review_status,), values=(
                i.id, f"{i.created_at:%Y-%m-%d %H:%M:%S}", i.session.filename, f"#{i.anonymous_track_id}",
                T.EVENT_NAMES.get(i.event_type, i.event_type), f"{i.confidence:.2f}", T.STATUS_LABELS[i.review_status]))


# =============================================================================== Monitor
class MonitorView(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=20)
        self.app = app
        self.video: Path | None = None
        self.info = None
        self.draw_mode = False
        self.region: list[tuple[float, float]] = [tuple(p) for p in app.ctl.settings.region_polygon]
        self._polling = False
        self._paused = False
        self._last_status = ""

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Video Monitor", style="H1.TLabel").pack(side="left")
        ttk.Label(self, text="Upload an exam video, process it locally, and review flagged observable events.",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 10))

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(0, 8))
        self.btn_upload = ttk.Button(bar, text="Upload Video…", command=self.upload)
        self.btn_upload.pack(side="left")
        self.btn_start = ttk.Button(bar, text="▶  Start Processing", style="Accent.TButton", command=self.start, state="disabled")
        self.btn_start.pack(side="left", padx=8)
        self.btn_pause = ttk.Button(bar, text="⏸  Pause", command=self.toggle_pause, state="disabled")
        self.btn_pause.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="■  Stop", style="Danger.TButton", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=8)
        self.auth_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="I am authorised to process this video", variable=self.auth_var,
                        command=self._update_buttons).pack(side="left", padx=18)
        self.file_var = tk.StringVar(value="No video selected")
        ttk.Label(self, textvariable=self.file_var, style="Muted.TLabel").pack(anchor="w")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=(8, 0))
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = ImageCanvas(left, placeholder="Upload a video to begin")
        self.canvas.pack(fill="both", expand=True)
        prog = ttk.Frame(left)
        prog.pack(fill="x", pady=(8, 0))
        self.progress = ttk.Progressbar(prog, maximum=100)
        self.progress.pack(fill="x")
        stats = ttk.Frame(prog)
        stats.pack(fill="x", pady=(4, 0))
        self.v_frame, self.v_time, self.v_speed, self.v_status = (tk.StringVar(value=x) for x in
                                                                  ("Frame: -", "Time: -", "Speed: -", "Idle"))
        for v in (self.v_frame, self.v_time, self.v_speed):
            ttk.Label(stats, textvariable=v, style="Muted.TLabel").pack(side="left", padx=(0, 18))
        ttk.Label(stats, textvariable=self.v_status, style="Muted.TLabel").pack(side="right")
        reg = ttk.Frame(left)
        reg.pack(fill="x", pady=(6, 0))
        self.btn_region = ttk.Button(reg, text="Draw monitored region", command=self.toggle_draw)
        self.btn_region.pack(side="left")
        ttk.Button(reg, text="Clear region", command=self.clear_region).pack(side="left", padx=8)
        self.region_var = tk.StringVar()
        ttk.Label(reg, textvariable=self.region_var, style="Muted.TLabel").pack(side="left", padx=8)
        self._region_text()

        right = ttk.Frame(body, width=450)
        right.pack(side="right", fill="y", padx=(14, 0))
        right.pack_propagate(False)
        ttk.Label(right, text="Alert Center", style="H2.TLabel").pack(anchor="w")
        ttk.Label(right, text="Double-click an alert to review it", style="Muted.TLabel").pack(anchor="w", pady=(0, 6))
        self.alerts = make_tree(right, [("t", "Time", 54), ("track", "Track", 48), ("event", "Event", 132), ("conf", "Conf", 44),
                                        ("status", "Status", 120)], 14)
        self.alerts.pack(fill="both", expand=True)
        self.alerts.bind("<Double-1>", lambda e: self._open_alert())
        self.refresh_alerts()

    # ---- region editor ------------------------------------------------------------
    def _region_text(self):
        self.region_var.set(f"Region: {len(self.region)} points" + (" (active)" if len(self.region) >= 3 else
                                                                   " (needs ≥3 points - disabled)" if self.region else " (none)"))

    def toggle_draw(self):
        if self.video is None:
            messagebox.showinfo("Draw region", "Upload a video first so the region can be drawn on a real frame.")
            return
        self.draw_mode = not self.draw_mode
        if self.draw_mode:
            self.region = []
            self.canvas.set_region(self.region)
            self.canvas.on_click_normalised(self._add_point)
            self.btn_region.configure(text="Finish region")
            self.v_status.set("Click on the preview to add region corners, then press Finish")
        else:
            self.canvas.on_click_normalised(None)
            self.btn_region.configure(text="Draw monitored region")
            if 0 < len(self.region) < 3:
                messagebox.showwarning("Region", "A region needs at least 3 points; it was discarded.")
                self.region = []
                self.canvas.set_region([])
            self._save_region()
            self.v_status.set("Idle")
        self._region_text()

    def _add_point(self, x: float, y: float):
        self.region.append((round(x, 4), round(y, 4)))
        self.canvas.set_region(self.region)
        self._region_text()

    def clear_region(self):
        self.region = []
        self.canvas.set_region([])
        self._save_region()
        self._region_text()

    def _save_region(self):
        s = Settings(**self.app.ctl.settings.to_dict())
        s.region_polygon = [list(p) for p in self.region] if len(self.region) >= 3 else []
        self.app.ctl.save_settings(s)

    # ---- upload / controls ------------------------------------------------------------
    def upload(self):
        exts = " ".join(f"*{e}" for e in SUPPORTED_VIDEO_EXTENSIONS)
        path = filedialog.askopenfilename(title="Select exam video", filetypes=[("Video files", exts), ("All files", "*.*")])
        if path:
            self.load_video(Path(path))

    def load_video(self, path: Path) -> bool:
        try:
            info = probe_video(path)
        except VideoError as e:
            messagebox.showerror("Cannot use this video", str(e))
            return False
        self.video, self.info = path, info
        self.file_var.set(f"{path.name}   |   {info.width}x{info.height}   |   {info.fps:.2f} fps   |   "
                          f"{info.frame_count} frames   |   {info.duration_s:.1f} s   |   {info.size_bytes / 1e6:.1f} MB")
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if ok:
            self.canvas.set_image(frame)
            self.canvas.set_region(self.region if len(self.region) >= 3 else [])
        self.progress["value"] = 0
        self._update_buttons()
        return True

    def _update_buttons(self):
        running = self.app.ctl.running
        self.btn_upload.configure(state="disabled" if running else "normal")
        self.btn_start.configure(state="normal" if (self.video and self.auth_var.get() and not running) else "disabled")
        self.btn_pause.configure(state="normal" if running else "disabled")
        self.btn_stop.configure(state="normal" if running else "disabled")

    def start(self):
        if not self.video or not self.auth_var.get():
            return
        if self.draw_mode:
            self.toggle_draw()
        try:
            self.app.ctl.settings.validate()
            self.app.ctl.start_processing(self.video)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Cannot start", str(e))
            return
        self._paused = False
        self.btn_pause.configure(text="⏸  Pause")
        self.v_status.set("Starting…")
        self._update_buttons()
        self.app.refresh_dashboard()
        if not self._polling:
            self._polling = True
            self.after(30, self._poll)

    def toggle_pause(self):
        w = self.app.ctl.worker
        if not w:
            return
        if self._paused:
            w.resume()
            self.btn_pause.configure(text="⏸  Pause")
            self.v_status.set("Processing…")
        else:
            w.pause()
            self.btn_pause.configure(text="▶  Resume")
            self.v_status.set("Paused")
        self._paused = not self._paused

    def stop(self):
        if self.app.ctl.worker:
            self.app.ctl.worker.stop()
            self.v_status.set("Stopping…")

    # ---- polling the worker queue (UI thread only) ------------------------------------------
    def _poll(self):
        w = self.app.ctl.worker
        if w is None:
            self._polling = False
            return
        for kind, payload in w.drain():
            self._handle(kind, payload)
        fr = w.latest_frame()
        if fr is not None:
            self.canvas.set_image(fr.image)
            total = max(fr.total_frames, 1)
            self.progress["value"] = min(100.0, 100.0 * (fr.frame_index + 1) / total)
            self.v_frame.set(f"Frame: {fr.frame_index + 1}/{fr.total_frames}")
            self.v_time.set(f"Time: {mmss(fr.timestamp_s)}")
            self.v_speed.set(f"Speed: {fr.fps_proc:.1f} fps")
            if not self._paused:
                self.v_status.set(f"Processing… {fr.n_tracks} tracks")
        if w.alive or not w.messages.empty():
            self.after(30, self._poll)
        else:
            self._polling = False
            self._update_buttons()
            self.app.refresh_dashboard()

    def _handle(self, kind: str, payload):
        app = self.app
        if kind == "status":
            self.v_status.set(str(payload))
        elif kind == "session":
            app.ctl.current_session_id = payload["session_id"]
        elif kind == "alert":
            a = payload
            if app.ctl.settings.notifications_enabled:
                app.toast.show("Review Required", a.message, T.AMBER, on_click=lambda i=a.incident_db_id: app.show_incident(i))
            self.refresh_alerts()
            app.refresh_dashboard()
        elif kind == "incident_resolved":
            self.refresh_alerts()
        elif kind == "error":
            app.toast.show("Processing error", str(payload), T.RED, ms=12000)
            self.v_status.set(f"Error: {payload}")
        elif kind == "done":
            s = payload
            if s is not None:
                self.progress["value"] = 100 if s.status == "COMPLETED" else self.progress["value"]
                msg = (f"{s.status.title()}: {s.frames_read} frames in {s.wall_seconds:.1f}s ({s.avg_fps:.1f} fps), "
                       f"{s.n_tracks} tracks, {s.n_incidents} events flagged.")
                if s.error:
                    msg += f" Error: {s.error}"
                self.v_status.set(msg)
                app.toast.show("Processing finished", msg, T.GREEN if s.status == "COMPLETED" else T.AMBER)
            self.refresh_alerts()
            app.refresh_dashboard()
            app.refresh_incidents()

    def refresh_alerts(self):
        sid = self.app.ctl.current_session_id
        flt = IncidentFilter(session_id=sid) if sid else None
        self.alerts.delete(*self.alerts.get_children())
        for i in self.app.ctl.repo.list_incidents(flt, limit=100):
            self.alerts.insert("", "end", iid=str(i.id), tags=(i.review_status,), values=(
                mmss(i.first_detected_at), f"#{i.anonymous_track_id}", T.EVENT_NAMES.get(i.event_type, i.event_type),
                f"{i.confidence:.2f}", T.STATUS_LABELS[i.review_status]))

    def _open_alert(self):
        sel = self.alerts.selection()
        if sel:
            self.app.show_incident(int(sel[0]))


# =============================================================================== Incidents
class IncidentsView(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=20)
        self.app = app
        self.current_id: int | None = None
        self._mode = tk.StringVar(value="annotated")
        ttk.Label(self, text="Incident History", style="H1.TLabel").pack(anchor="w")
        ttk.Label(self, text="Search, review and export recorded observations. Reviews are stored with history.",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 8))

        f = ttk.Frame(self)
        f.pack(fill="x")
        self.v_from, self.v_to, self.v_track, self.v_text = (tk.StringVar() for _ in range(4))
        self.v_session, self.v_event, self.v_status = (tk.StringVar(value="All") for _ in range(3))

        def field(label, widget, col, row):
            ttk.Label(f, text=label, style="Muted.TLabel").grid(row=row * 2, column=col, sticky="w", padx=(0, 8))
            widget.grid(row=row * 2 + 1, column=col, sticky="ew", padx=(0, 8), pady=(0, 4))
        field("From (YYYY-MM-DD)", ttk.Entry(f, textvariable=self.v_from, width=13), 0, 0)
        field("To (YYYY-MM-DD)", ttk.Entry(f, textvariable=self.v_to, width=13), 1, 0)
        self.cb_session = ttk.Combobox(f, textvariable=self.v_session, state="readonly", width=22)
        field("Video session", self.cb_session, 2, 0)
        field("Event type", ttk.Combobox(f, textvariable=self.v_event, state="readonly", width=18,
                                        values=["All", *T.EVENT_NAMES.values()]), 3, 0)
        field("Review status", ttk.Combobox(f, textvariable=self.v_status, state="readonly", width=20,
                                           values=["All", *T.STATUS_LABELS.values()]), 0, 1)
        field("Track ID", ttk.Entry(f, textvariable=self.v_track, width=13), 1, 1)
        field("Search", ttk.Entry(f, textvariable=self.v_text, width=22), 2, 1)
        bb = ttk.Frame(f)
        bb.grid(row=3, column=3, sticky="w", pady=(0, 4))
        ttk.Button(bb, text="Apply", style="Accent.TButton", command=self.refresh).pack(side="left", padx=(0, 6))
        ttk.Button(bb, text="Reset", command=self.reset_filters).pack(side="left")
        self.err = tk.StringVar()
        ttk.Label(self, textvariable=self.err, foreground=T.RED, background=T.BG).pack(anchor="w")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=(6, 0))
        lf = ttk.Frame(body, width=572)  # fixed-width list column; the evidence/review pane takes the rest
        lf.pack(side="left", fill="y")
        lf.pack_propagate(False)
        self.tree = make_tree(lf, [("id", "ID", 32), ("when", "Recorded", 100), ("track", "Track", 56),
                                   ("event", "Event", 140), ("vt", "At", 66), ("conf", "Conf", 50), ("status", "Status", 128)], 14)
        for col in self.tree["columns"]:
            self.tree.column(col, stretch=False, minwidth=24)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._select())
        self.count_var = tk.StringVar()
        ttk.Label(lf, textvariable=self.count_var, style="Muted.TLabel").pack(anchor="w", pady=(4, 0))
        ex_bar = ttk.Frame(lf)
        ex_bar.pack(fill="x", pady=(6, 0))
        for text, cmd in [("CSV", self.exp_csv), ("JSON", self.exp_json), ("Screenshots", self.exp_shots),
                          ("Report", self.exp_report), ("Video", self.exp_video)]:
            ttk.Button(ex_bar, text=text, command=cmd, padding=(5, 6)).pack(side="left", padx=(0, 4))

        rf = ttk.Frame(body, padding=(14, 0, 0, 0))
        rf.pack(side="left", fill="both", expand=True)
        rf.bind("<Configure>", lambda e: self.meta_label.configure(wraplength=max(200, e.width - 40)))
        self.viewer = ImageCanvas(rf, placeholder="Select an incident to view its evidence")
        self.viewer.pack(fill="both", expand=True)
        vb = ttk.Frame(rf)
        vb.pack(fill="x", pady=(6, 0))
        ttk.Radiobutton(vb, text="Annotated", variable=self._mode, value="annotated", command=self._show_image).pack(side="left")
        ttk.Radiobutton(vb, text="Original frame", variable=self._mode, value="raw", command=self._show_image).pack(side="left", padx=10)
        ttk.Button(vb, text="Open in viewer", command=self._open_file).pack(side="right")
        self.meta = tk.StringVar(value="")
        self.meta_label = ttk.Label(rf, textvariable=self.meta, justify="left", wraplength=420, style="Muted.TLabel")
        self.meta_label.pack(anchor="w", pady=(8, 4))
        rv = ttk.Frame(rf)
        rv.pack(fill="x")
        ttk.Label(rv, text="Decision: Confirm observation · Dismiss (false alert) · More evidence",
                  style="Muted.TLabel").pack(anchor="w")
        ttk.Label(rv, text="Reviewer note", style="Muted.TLabel").pack(anchor="w", pady=(4, 0))
        self.note = tk.Text(rv, height=3, bg=T.PANEL2, fg=T.TEXT, insertbackground=T.TEXT, relief="flat", wrap="word", font=T.FONT)
        self.note.pack(fill="x", pady=(2, 6))
        btns = ttk.Frame(rv)
        btns.pack(fill="x")
        self.review_btns = [
            ttk.Button(btns, text="Confirm", style="Accent.TButton", padding=(10, 7), command=lambda: self.review(REVIEW_CONFIRMED)),
            ttk.Button(btns, text="Dismiss", padding=(10, 7), command=lambda: self.review(REVIEW_DISMISSED)),
            ttk.Button(btns, text="More evidence", padding=(10, 7), command=lambda: self.review(REVIEW_MORE))]
        for b in self.review_btns:
            b.pack(side="left", padx=(0, 6))
            b.state(["disabled"])
        self.hist = tk.Text(rv, height=4, bg=T.PANEL, fg=T.MUTED, relief="flat", font=T.FONT_S, state="disabled", wrap="word")
        self.hist.pack(fill="x", pady=(8, 0))
        self._img_cache: dict[str, object] = {}
        self.refresh()

    # ---- filtering ---------------------------------------------------------------------------
    def _sessions(self):
        sess = self.app.ctl.repo.list_sessions()
        self._session_map = {f"{s.id}: {s.filename}"[:40]: s.id for s in sess}
        self.cb_session.configure(values=["All", *self._session_map])

    def reset_filters(self):
        for v in (self.v_from, self.v_to, self.v_track, self.v_text):
            v.set("")
        for v in (self.v_session, self.v_event, self.v_status):
            v.set("All")
        self.refresh()

    def _build_filter(self) -> IncidentFilter | None:
        self.err.set("")
        f = IncidentFilter()
        try:
            if self.v_from.get().strip():
                f.date_from = datetime.strptime(self.v_from.get().strip(), "%Y-%m-%d")
            if self.v_to.get().strip():
                f.date_to = datetime.strptime(self.v_to.get().strip(), "%Y-%m-%d")
            if self.v_track.get().strip():
                f.track_id = int(self.v_track.get().strip())
        except ValueError:
            self.err.set("Invalid filter: dates must be YYYY-MM-DD and Track ID a whole number.")
            return None
        if self.v_session.get() != "All":
            f.session_id = self._session_map.get(self.v_session.get())
        rev_e = {v: k for k, v in T.EVENT_NAMES.items()}
        rev_s = {v: k for k, v in T.STATUS_LABELS.items()}
        f.event_type = rev_e.get(self.v_event.get())
        f.review_status = rev_s.get(self.v_status.get())
        f.text = self.v_text.get().strip() or None
        return f

    def refresh(self):
        self._sessions()
        flt = self._build_filter()
        if flt is None:
            return
        keep = self.current_id
        self.tree.delete(*self.tree.get_children())
        rows = self.app.ctl.repo.list_incidents(flt)
        for i in rows:
            self.tree.insert("", "end", iid=str(i.id), tags=(i.review_status,), values=(
                i.id, f"{i.created_at:%m-%d %H:%M}", f"#{i.anonymous_track_id}",
                T.EVENT_NAMES.get(i.event_type, i.event_type), mmss(i.first_detected_at), f"{i.confidence:.2f}",
                T.STATUS_LABELS[i.review_status]))
        self.count_var.set(f"{len(rows)} incident(s) - machine-generated observations pending human review")
        self._rows = rows
        if keep and self.tree.exists(str(keep)):
            self.tree.selection_set(str(keep))

    def select(self, incident_id: int):
        self.reset_filters()
        if self.tree.exists(str(incident_id)):
            self.tree.selection_set(str(incident_id))
            self.tree.see(str(incident_id))

    # ---- detail / viewer / review ---------------------------------------------------------------
    def _select(self):
        sel = self.tree.selection()
        if not sel:
            return
        self.current_id = int(sel[0])
        self._load_detail()

    def _load_detail(self):
        inc = self.app.ctl.repo.get_incident(self.current_id)
        if inc is None:
            return
        self.meta.set(
            f"Incident #{inc.id} - {T.EVENT_NAMES.get(inc.event_type, inc.event_type)}\n"
            f"Anonymous Track #{inc.anonymous_track_id}  |  Video time {fmt_ts(inc.first_detected_at)} - {fmt_ts(inc.last_detected_at)}  |  "
            f"Confidence {inc.confidence:.2f}  |  {inc.lifecycle_state.title()}\nVideo: {inc.session.filename}  |  Recorded "
            f"{inc.created_at:%Y-%m-%d %H:%M:%S} UTC  |  Status: {T.STATUS_LABELS[inc.review_status]}"
            + (f"\nRegion: {inc.region_id}" if inc.region_id else ""))
        for b in self.review_btns:
            b.state(["!disabled"])
        self.hist.configure(state="normal")
        self.hist.delete("1.0", "end")
        for r in inc.reviews:
            self.hist.insert("end", f"{r.reviewed_at:%Y-%m-%d %H:%M} UTC - {T.STATUS_LABELS[r.review_decision]}"
                                    + (f": {r.reviewer_note}" if r.reviewer_note else "") + "\n")
        if not inc.reviews:
            self.hist.insert("end", "No review recorded yet.")
        self.hist.configure(state="disabled")
        self._show_image()

    def _evidence_path(self) -> str | None:
        inc = self.app.ctl.repo.get_incident(self.current_id) if self.current_id else None
        if not inc:
            return None
        return inc.raw_screenshot_path if self._mode.get() == "raw" else inc.screenshot_path

    def _show_image(self):
        p = self._evidence_path()
        img = cv2.imread(p) if p and Path(p).is_file() else None
        if img is None:
            self.viewer._placeholder = "Evidence file is missing or unreadable"
        self.viewer.set_image(img)

    def _open_file(self):
        p = self._evidence_path()
        if p and Path(p).is_file():
            open_in_system(p)
        else:
            messagebox.showwarning("Evidence", "The evidence file is missing (it may have been deleted by the retention policy).")

    def review(self, decision: str):
        if self.current_id is None:
            return
        self.app.ctl.repo.review(self.current_id, decision, self.note.get("1.0", "end").strip())
        self.note.delete("1.0", "end")
        self.refresh()
        self._load_detail()
        self.app.refresh_dashboard()
        self.app.refresh_alerts()

    # ---- exports ------------------------------------------------------------------------------------
    def _current_list(self):
        flt = self._build_filter()
        return self.app.ctl.repo.list_incidents(flt) if flt else []

    def _export_dir(self) -> Path:
        return self.app.ctl.paths.exports

    def exp_csv(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialdir=self._export_dir(), initialfile="incidents.csv")
        if p:
            ex.export_csv(self.app.ctl.repo, self._current_list(), Path(p))
            self.app.toast.show("Export complete", f"CSV saved: {p}", T.GREEN)

    def exp_json(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", initialdir=self._export_dir(), initialfile="incidents.json")
        if p:
            ex.export_json(self.app.ctl.repo, self._current_list(), Path(p))
            self.app.toast.show("Export complete", f"JSON saved: {p}", T.GREEN)

    def exp_shots(self):
        d = filedialog.askdirectory(initialdir=self._export_dir(), title="Folder for screenshots")
        if d:
            n = ex.export_screenshots(self._current_list(), Path(d))
            self.app.toast.show("Export complete", f"{n} screenshot file(s) copied.", T.GREEN)

    def _session_for_export(self) -> int | None:
        if self.current_id:
            return self.app.ctl.repo.get_incident(self.current_id).video_session_id
        sid = self._session_map.get(self.v_session.get())
        if sid is None:
            messagebox.showinfo("Select a session", "Select an incident or choose a video session in the filter first.")
        return sid

    def exp_report(self):
        sid = self._session_for_export()
        if sid is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".html", initialdir=self._export_dir(), initialfile=f"session_{sid}_report.html")
        if p:
            ex.export_report_html(self.app.ctl.repo, sid, Path(p))
            self.app.toast.show("Report created", p, T.GREEN, on_click=lambda: open_in_system(p))

    def exp_video(self):
        sid = self._session_for_export()
        if sid is None:
            return
        d = filedialog.askdirectory(initialdir=self._export_dir(), title="Folder for annotated video")
        if d:
            out = ex.export_annotated_video(self.app.ctl.repo.get_session(sid), Path(d))
            if out:
                self.app.toast.show("Export complete", f"Video saved: {out}", T.GREEN)
            else:
                messagebox.showwarning("Annotated video", "No annotated video exists for this session.")


# =============================================================================== Settings
class SettingsView(ttk.Frame):
    FIELDS = [  # (key, label, kind, extra)
        ("person_conf", "Person confidence threshold (0.05-0.99)", float, None),
        ("phone_conf", "Phone confidence threshold (0.01-0.99)", float, None),
        ("imgsz", "YOLO input size px (0 = auto: native, max 1920)", int, [0, 640, 960, 1280, 1920]),
        ("weights_path", "Detection model (yolo11s = more accurate, slower)", str, ["models/yolo11n.pt", "models/yolo11s.pt"]),
        ("tracker_kind", "Tracker", str, ["botsort", "bytetrack"]),
        ("phone_crop_imgsz", "Phone crop-assist resolution (px)", int, [416, 640, 960]),
        ("frame_stride", "Frame sampling: run detection every N frames", int, None),
        ("device", "Compute device", str, ["auto", "mps", "cpu"]),
        ("phone_window_frames", "Temporal window (processed frames)", int, None),
        ("phone_min_support_frames", "Frames required inside window", int, None),
        ("phone_min_duration_seconds", "Minimum event duration to alert (s)", float, None),
        ("resolve_after_seconds", "Resolve event after absence (s)", float, None),
        ("event_cooldown_seconds", "Event cooldown after resolve (s)", float, None),
        ("region_exit_min_seconds", "Region exit: minimum time outside (s)", float, None),
        ("retention_days", "Data retention (days, 0 = keep forever)", int, None),
    ]

    def __init__(self, master, app):
        super().__init__(master, padding=24)
        self.app = app
        ttk.Label(self, text="Settings", style="H1.TLabel").pack(anchor="w")
        ttk.Label(self, text="All values are validated before saving. Thresholds are engineering defaults, not calibrated values.",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 12))
        form = ttk.Frame(self)
        form.pack(anchor="w")
        self.vars: dict[str, tk.StringVar] = {}
        for r, (key, label, _kind, choices) in enumerate(self.FIELDS):
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w", pady=4, padx=(0, 18))
            v = tk.StringVar()
            self.vars[key] = v
            (ttk.Combobox(form, textvariable=v, values=choices, width=12) if choices else ttk.Entry(form, textvariable=v, width=14)
             ).grid(row=r, column=1, sticky="w")
        r = len(self.FIELDS)
        ttk.Label(form, text="Output folder (applies after restart)").grid(row=r, column=0, sticky="w", pady=4)
        self.v_out = tk.StringVar()
        ttk.Entry(form, textvariable=self.v_out, width=44).grid(row=r, column=1, columnspan=2, sticky="w")
        ttk.Button(form, text="Choose…", command=self._choose).grid(row=r, column=3, padx=6)
        self.v_notif, self.v_sound, self.v_video, self.v_crop = (tk.BooleanVar() for _ in range(4))
        for i, (txt, var) in enumerate([("Show alert notifications", self.v_notif), ("Notification sound (macOS)", self.v_sound),
                                        ("Save annotated video", self.v_video),
                                        ("Phone crop-assist (optional: +recall, -precision, slower)", self.v_crop)]):
            ttk.Checkbutton(form, text=txt, variable=var).grid(row=r + 1 + i, column=0, sticky="w", pady=2)
        bar = ttk.Frame(self)
        bar.pack(anchor="w", pady=16)
        ttk.Button(bar, text="Save settings", style="Accent.TButton", command=self.save).pack(side="left")
        ttk.Button(bar, text="Reset to defaults", command=self.defaults).pack(side="left", padx=8)
        ttk.Button(bar, text="Apply retention cleanup now", command=self.cleanup).pack(side="left")
        self.msg = tk.StringVar()
        ttk.Label(self, textvariable=self.msg, style="Muted.TLabel", wraplength=800).pack(anchor="w")
        ttk.Label(self, text="Privacy: processing is fully local. Nothing is uploaded. No face recognition or identity matching is performed.",
                  style="Muted.TLabel", wraplength=800).pack(anchor="w", pady=(16, 0))
        self.load()

    def load(self, s: Settings | None = None):
        s = s or self.app.ctl.settings
        for key, _l, _k, _c in self.FIELDS:
            self.vars[key].set(str(getattr(s, key)))
        self.v_out.set(s.output_dir)
        self.v_notif.set(s.notifications_enabled)
        self.v_sound.set(s.notification_sound)
        self.v_video.set(s.save_annotated_video)
        self.v_crop.set(s.phone_crop_assist)

    def _choose(self):
        d = filedialog.askdirectory(title="Output folder")
        if d:
            self.v_out.set(d)

    def collect(self) -> Settings:
        data = self.app.ctl.settings.to_dict()
        for key, label, kind, _c in self.FIELDS:
            raw = self.vars[key].get().strip()
            try:
                data[key] = kind(raw)
            except ValueError:
                raise ConfigError(f"'{label}' must be a valid {kind.__name__}") from None
        data["output_dir"] = self.v_out.get().strip()
        data["notifications_enabled"], data["notification_sound"] = self.v_notif.get(), self.v_sound.get()
        data["save_annotated_video"] = self.v_video.get()
        data["phone_crop_assist"] = self.v_crop.get()
        if data["output_dir"] and not Path(data["output_dir"]).expanduser().is_dir():
            raise ConfigError("Output folder does not exist")
        return Settings(**data).validate()

    def save(self):
        try:
            s = self.collect()
            self.app.ctl.save_settings(s)
        except (ConfigError, ValueError) as e:
            self.msg.set(f"Not saved: {e}")
            return False
        self.msg.set("Settings saved." + (" Output folder change takes effect after restart." if s.output_dir else ""))
        return True

    def defaults(self):
        self.load(Settings())
        self.msg.set("Defaults loaded - press Save to apply.")

    def cleanup(self):
        n = self.app.ctl.apply_retention()
        self.msg.set(f"Retention cleanup removed {n} evidence file(s) older than {self.app.ctl.settings.retention_days} days.")
        self.app.refresh_dashboard()
        self.app.refresh_incidents()
