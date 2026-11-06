"""Main window: sidebar navigation + views + toast layer."""
from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from examguard.config import APP_NAME, DISCLAIMER
from examguard.ui import theme as T
from examguard.ui.controller import AppController
from examguard.ui.views import DashboardView, IncidentsView, MonitorView, SettingsView
from examguard.ui.widgets import Toast

log = logging.getLogger(__name__)
NAV = [("dashboard", "◧  Dashboard"), ("monitor", "▶  Video Monitor"), ("incidents", "☰  Incidents"), ("settings", "⚙  Settings")]


class ExamGuardApp(tk.Tk):
    def __init__(self, home: Path | None = None):
        super().__init__()
        self.title(f"{APP_NAME} - Computer Vision Monitoring & Event Review Platform")
        self.geometry("1360x860")
        self.minsize(1100, 720)
        T.apply_style(self)
        self.ctl = AppController(home)
        self.toast = Toast(self)
        try:
            n = self.ctl.apply_retention()
            if n:
                log.info("retention removed %d files", n)
        except Exception:  # noqa: BLE001
            log.exception("retention failed")

        side = ttk.Frame(self, style="Side.TFrame", width=232)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk.Label(side, text="ExamGuard AI", bg=T.PANEL, fg=T.TEXT, font=("Helvetica Neue", 20, "bold")).pack(anchor="w", padx=18, pady=(22, 0))
        tk.Label(side, text="Computer Vision Monitoring &\nEvent Review Platform", bg=T.PANEL, fg=T.MUTED, font=T.FONT_S,
                 justify="left").pack(anchor="w", padx=18, pady=(2, 22))
        self.nav_btns: dict[str, ttk.Button] = {}
        for key, label in NAV:
            b = ttk.Button(side, text=label, style="Nav.TButton", command=lambda k=key: self.navigate(k))
            b.pack(fill="x")
            self.nav_btns[key] = b
        tk.Label(side, text=DISCLAIMER, bg=T.PANEL, fg=T.AMBER, font=T.FONT_S, wraplength=200, justify="left").pack(
            side="bottom", anchor="w", padx=16, pady=16)

        self.container = ttk.Frame(self)
        self.container.pack(side="left", fill="both", expand=True)
        self.views = {
            "dashboard": DashboardView(self.container, self), "monitor": MonitorView(self.container, self),
            "incidents": IncidentsView(self.container, self), "settings": SettingsView(self.container, self)}
        self.current = ""
        self.navigate("dashboard")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._closing = False

    # -- navigation / refresh ---------------------------------------------------
    def navigate(self, key: str) -> None:
        for k, v in self.views.items():
            v.pack_forget()
            self.nav_btns[k].configure(style="Nav.TButton")
        self.views[key].pack(fill="both", expand=True)
        self.nav_btns[key].configure(style="NavSel.TButton")
        self.current = key
        if key == "dashboard":
            self.refresh_dashboard()
        elif key == "incidents":
            self.refresh_incidents()

    def show_incident(self, incident_id: int) -> None:
        self.navigate("incidents")
        self.views["incidents"].select(incident_id)

    def refresh_dashboard(self) -> None:
        self.views["dashboard"].refresh()

    def refresh_incidents(self) -> None:
        self.views["incidents"].refresh()

    def refresh_alerts(self) -> None:
        self.views["monitor"].refresh_alerts()

    # -- shutdown ----------------------------------------------------------------
    def on_close(self) -> None:
        if self._closing:
            return
        if self.ctl.running and not messagebox.askyesno("Quit", "A video is still being processed. Stop and quit?"):
            return
        self._closing = True
        stopped = self.ctl.shutdown()
        if not stopped:
            log.warning("worker did not stop within timeout; exiting anyway (daemon thread)")
        self.destroy()
