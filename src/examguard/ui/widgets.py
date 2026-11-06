"""Reusable Tk widgets: scaled image canvas with ROI overlay, toasts, stat cards."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

import cv2
import numpy as np

from examguard.ui import theme as T


def bgr_to_photo(bgr: np.ndarray) -> tk.PhotoImage:
    """BGR ndarray -> tk.PhotoImage using binary PPM (fast, no Pillow/Tk bridge needed)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return tk.PhotoImage(data=b"P6 %d %d 255\n" % (rgb.shape[1], rgb.shape[0]) + rgb.tobytes(), format="PPM")


class ImageCanvas(tk.Canvas):
    """Letterboxed image view. Overlays are stored in NORMALISED image coordinates, so they are
    re-projected on every resize and can never drift from the image content."""

    def __init__(self, master, placeholder: str = "No video", **kw):
        super().__init__(master, bg="#050B16", highlightthickness=1, highlightbackground=T.LINE, **kw)
        self._img: np.ndarray | None = None
        self._photo: tk.PhotoImage | None = None
        self._rect = (0, 0, 1, 1)  # ox, oy, w, h of displayed image inside the canvas
        self._region: list[tuple[float, float]] = []
        self._placeholder = placeholder
        self._click_cb: Callable[[float, float], None] | None = None
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Button-1>", self._on_click)

    # -- public ---------------------------------------------------------------
    def set_image(self, bgr: np.ndarray | None) -> None:
        self._img = bgr
        self.redraw()

    def set_region(self, pts: list[tuple[float, float]]) -> None:
        self._region = [tuple(p) for p in pts]
        self._draw_overlay()

    def on_click_normalised(self, cb: Callable[[float, float], None] | None) -> None:
        self._click_cb = cb
        self.configure(cursor="crosshair" if cb else "")

    def image_rect(self) -> tuple[int, int, int, int]:
        return self._rect

    def to_normalised(self, x: float, y: float) -> tuple[float, float] | None:
        ox, oy, w, h = self._rect
        if w <= 0 or h <= 0 or not (ox <= x <= ox + w and oy <= y <= oy + h):
            return None
        return (x - ox) / w, (y - oy) / h

    # -- internals --------------------------------------------------------------
    def redraw(self) -> None:
        self.delete("all")
        cw, ch = max(self.winfo_width(), 2), max(self.winfo_height(), 2)
        if self._img is None:
            self._rect = (0, 0, 1, 1)
            self.create_text(cw // 2, ch // 2, text=self._placeholder, fill=T.MUTED, font=T.FONT_H2)
            return
        h, w = self._img.shape[:2]
        s = min(cw / w, ch / h)
        sw, sh = max(1, int(w * s)), max(1, int(h * s))
        small = cv2.resize(self._img, (sw, sh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        self._photo = bgr_to_photo(small)  # keep a reference or Tk garbage-collects it
        ox, oy = (cw - sw) // 2, (ch - sh) // 2
        self._rect = (ox, oy, sw, sh)
        self.create_image(ox, oy, image=self._photo, anchor="nw")
        self._draw_overlay()

    def _draw_overlay(self) -> None:
        self.delete("roi")
        if not self._region or self._img is None:
            return
        ox, oy, w, h = self._rect
        pts = [(ox + x * w, oy + y * h) for x, y in self._region]
        for x, y in pts:
            self.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#FFC800", outline="", tags="roi")
        if len(pts) >= 2:
            flat = [c for p in pts for c in p]
            if len(pts) >= 3:
                self.create_polygon(*flat, outline="#FFC800", fill="", width=2, tags="roi")
            else:
                self.create_line(*flat, fill="#FFC800", width=2, tags="roi")

    def _on_click(self, e) -> None:
        if self._click_cb is None:
            return
        n = self.to_normalised(e.x, e.y)
        if n:
            self._click_cb(*n)


class Toast:
    """Non-blocking in-app notification that auto-dismisses; never grabs focus or blocks the UI."""

    def __init__(self, root: tk.Misc):
        self.root = root
        self._active: list[tk.Frame] = []

    def show(self, title: str, text: str, color: str = T.AMBER, on_click: Callable[[], None] | None = None,
             ms: int = 8000) -> tk.Frame:
        f = tk.Frame(self.root, bg=T.PANEL2, highlightthickness=2, highlightbackground=color, cursor="hand2" if on_click else "")
        tk.Label(f, text=title, bg=T.PANEL2, fg=color, font=T.FONT_B, anchor="w").pack(fill="x", padx=12, pady=(10, 0))
        tk.Label(f, text=text, bg=T.PANEL2, fg=T.TEXT, font=T.FONT, wraplength=330, justify="left", anchor="w").pack(fill="x", padx=12, pady=(2, 10))
        if on_click:
            for w in (f, *f.winfo_children()):
                w.bind("<Button-1>", lambda e: (on_click(), self._close(f)))
        self._active.append(f)
        self._layout()
        self.root.after(ms, lambda: self._close(f))
        return f

    def _close(self, f: tk.Frame) -> None:
        if f in self._active:
            self._active.remove(f)
            f.destroy()
            self._layout()

    def _layout(self) -> None:
        y = 16
        for f in self._active:
            f.place(relx=1.0, x=-16, y=y, anchor="ne")
            f.lift()
            f.update_idletasks()
            y += f.winfo_reqheight() + 10

    @property
    def count(self) -> int:
        return len(self._active)


class StatCard(ttk.Frame):
    def __init__(self, master, title: str, accent: str = T.BLUE):
        super().__init__(master, style="Card.TFrame", padding=(18, 14))
        tk.Frame(self, bg=accent, height=3).pack(fill="x", pady=(0, 10))
        ttk.Label(self, text=title.upper(), style="PanelMuted.TLabel").pack(anchor="w")
        self.var = tk.StringVar(value="-")
        ttk.Label(self, textvariable=self.var, style="Big.TLabel").pack(anchor="w", pady=(4, 0))

    def set(self, v) -> None:
        self.var.set(str(v))
