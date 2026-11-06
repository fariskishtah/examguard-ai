"""Navy / graphite / electric-blue theme for ttk (macOS-friendly 'clam' base)."""
from __future__ import annotations

from tkinter import ttk

BG = "#0A1628"; PANEL = "#102340"; PANEL2 = "#16294A"; GRAPHITE = "#243248"; LINE = "#2A3F63"
BLUE = "#2F7BFF"; BLUE_HI = "#5A98FF"; TEXT = "#E8EEF9"; MUTED = "#8FA3C2"
AMBER = "#F5A623"; RED = "#EB3C3C"; GREEN = "#38C172"
FONT = ("Helvetica Neue", 12); FONT_B = ("Helvetica Neue", 12, "bold"); FONT_H1 = ("Helvetica Neue", 22, "bold")
FONT_H2 = ("Helvetica Neue", 15, "bold"); FONT_S = ("Helvetica Neue", 10); FONT_BIG = ("Helvetica Neue", 30, "bold")

STATUS_COLORS = {"PENDING_REVIEW": AMBER, "CONFIRMED_OBSERVATION": RED, "DISMISSED": MUTED, "NEEDS_MORE_EVIDENCE": BLUE_HI}
STATUS_LABELS = {"PENDING_REVIEW": "Pending Review", "CONFIRMED_OBSERVATION": "Confirmed Observation",
                 "DISMISSED": "Dismissed", "NEEDS_MORE_EVIDENCE": "Needs More Evidence"}
EVENT_NAMES = {"PHONE_VISIBLE_NEAR_TRACK": "Phone near track", "MONITORED_REGION_EXIT": "Region exit"}


def apply_style(root) -> ttk.Style:
    st = ttk.Style(root)
    st.theme_use("clam")
    root.configure(bg=BG)
    st.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL2, bordercolor=LINE, font=FONT,
                 lightcolor=PANEL, darkcolor=PANEL, troughcolor=PANEL)
    st.configure("TFrame", background=BG)
    st.configure("Panel.TFrame", background=PANEL)
    st.configure("Card.TFrame", background=PANEL, relief="flat")
    st.configure("Side.TFrame", background=PANEL)
    st.configure("TLabel", background=BG, foreground=TEXT)
    st.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
    st.configure("Muted.TLabel", background=BG, foreground=MUTED, font=FONT_S)
    st.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED, font=FONT_S)
    st.configure("H1.TLabel", font=FONT_H1)
    st.configure("H2.TLabel", font=FONT_H2)
    st.configure("Big.TLabel", background=PANEL, font=FONT_BIG, foreground=TEXT)
    st.configure("Warn.TLabel", background="#3A2C0A", foreground=AMBER, font=FONT_S, padding=8)
    st.configure("TButton", background=GRAPHITE, foreground=TEXT, padding=(12, 7), borderwidth=0, focusthickness=0)
    st.map("TButton", background=[("active", LINE), ("disabled", PANEL)], foreground=[("disabled", MUTED)])
    st.configure("Accent.TButton", background=BLUE, foreground="white", font=FONT_B)
    st.map("Accent.TButton", background=[("active", BLUE_HI), ("disabled", PANEL)])
    st.configure("Danger.TButton", background="#7A2330", foreground="white")
    st.map("Danger.TButton", background=[("active", RED), ("disabled", PANEL)])
    st.configure("Nav.TButton", background=PANEL, foreground=MUTED, anchor="w", padding=(16, 10), font=FONT_B)
    st.map("Nav.TButton", background=[("active", PANEL2)], foreground=[("active", TEXT)])
    st.configure("NavSel.TButton", background=PANEL2, foreground=TEXT, anchor="w", padding=(16, 10), font=FONT_B)
    st.configure("TEntry", fieldbackground=PANEL2, foreground=TEXT, insertcolor=TEXT, padding=5)
    st.configure("TCombobox", fieldbackground=PANEL2, foreground=TEXT, background=GRAPHITE, arrowcolor=TEXT, padding=4)
    st.map("TCombobox", fieldbackground=[("readonly", PANEL2)], foreground=[("readonly", TEXT)])
    st.configure("TCheckbutton", background=BG, foreground=TEXT)
    st.map("TCheckbutton", background=[("active", BG)])
    st.configure("Panel.TCheckbutton", background=PANEL)
    st.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=28, borderwidth=0)
    st.configure("Treeview.Heading", background=GRAPHITE, foreground=TEXT, font=FONT_B, padding=6, relief="flat")
    st.map("Treeview", background=[("selected", BLUE)], foreground=[("selected", "white")])
    st.configure("Horizontal.TProgressbar", background=BLUE, troughcolor=PANEL2, thickness=10, borderwidth=0)
    st.configure("TSeparator", background=LINE)
    st.configure("TNotebook", background=BG)
    return st
