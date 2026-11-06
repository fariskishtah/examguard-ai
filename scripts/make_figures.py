"""Render the architecture diagram and the evaluation overview from the measured JSON artifacts."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/screenshots"
BG, PANEL, LINE, TEXT, MUTED = (10, 22, 40), (16, 35, 64), (42, 63, 99), (232, 238, 249), (143, 163, 194)
BLUE, AMBER, GREEN, RED = (47, 123, 255), (245, 166, 35), (56, 193, 114), (235, 60, 60)


def font(size, bold=False):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/HelveticaNeue.ttc", "/Library/Fonts/Arial.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size, index=1 if bold else 0)
            except Exception:  # noqa: BLE001
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def arch():
    W, H = 1600, 760
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((40, 28), "ExamGuard AI - pipeline architecture", font=font(34, True), fill=TEXT)
    d.text((40, 76), "Local-first. Detection, tracking, event analysis, storage and UI are separate modules.", font=font(18), fill=MUTED)
    steps = [("Video input", "OpenCV decode\nstreamed, true\ntimestamps", "video/reader"), ("YOLO11n", "person + cell phone\nnative resolution (<=1920)\n+ 640-px crop assist", "detection/"),
             ("BoT-SORT", "anonymous\ntrack IDs\n(no identity)", "tracking/"), ("Event engine", "association, >=0.5 s\nconfirmation, ID-switch\nmerge, lifecycle", "events/"),
             ("Evidence", "raw + annotated\nframe, atomic,\nunique names", "evidence/"), ("SQLite", "Alembic migrations\nsessions, tracks,\nincidents, reviews", "database/")]
    bw, gap, x0, y0 = 220, 50, 40, 150
    for i, (t, sub, mod) in enumerate(steps):
        x = x0 + i * (bw + gap) - (0 if i < 6 else 0)
        d.rounded_rectangle((x, y0, x + bw - 20, y0 + 210), 14, fill=PANEL, outline=BLUE if i in (1, 3) else LINE, width=3)
        d.text((x + 16, y0 + 14), t, font=font(24, True), fill=TEXT)
        d.multiline_text((x + 16, y0 + 62), sub, font=font(17), fill=MUTED, spacing=5)
        d.text((x + 16, y0 + 176), mod, font=font(15), fill=BLUE)
        if i < len(steps) - 1:
            d.line((x + bw - 20, y0 + 105, x + bw + gap - 4, y0 + 105), fill=BLUE, width=4)
            d.polygon([(x + bw + gap - 4, y0 + 105), (x + bw + gap - 16, y0 + 97), (x + bw + gap - 16, y0 + 113)], fill=BLUE)
    d.rounded_rectangle((40, 420, 1560, 560), 14, fill=PANEL, outline=LINE, width=3)
    d.text((60, 434), "Tkinter desktop app (single process)", font=font(24, True), fill=TEXT)
    d.text((60, 474), "Worker thread  ->  thread-safe queue (+ latest-frame slot)  ->  UI thread polls via after()", font=font(19), fill=MUTED)
    d.text((60, 506), "Dashboard · Video Monitor · Alert Center · Incident History · Evidence viewer · Human review · Settings · Exports", font=font(19), fill=MUTED)
    for lbl, col, x in (("GREEN  normal track", GREEN, 60), ("AMBER  evidence accumulating", AMBER, 420), ("RED  active event - REVIEW REQUIRED", RED, 860)):
        d.rectangle((x, 600, x + 34, 626), outline=col, width=4); d.text((x + 46, 600), lbl, font=font(20), fill=TEXT)
    d.text((40, 690), "Automated alerts identify observable events requiring human review. They are not proof of academic misconduct.", font=font(19), fill=AMBER)
    im.save(OUT / "08_system_architecture.png")


def bars(d, x, y, w, h, items, maxv, title, fmt):
    d.text((x, y - 44), title, font=font(22, True), fill=TEXT)
    bh = h / len(items)
    for i, (lbl, v, col) in enumerate(items):
        yy = y + i * bh
        d.text((x, yy + 6), lbl, font=font(16), fill=MUTED)
        L = int((w - 250) * v / maxv)
        d.rectangle((x + 250, yy + 4, x + 250 + max(L, 2), yy + bh - 12), fill=col)
        d.text((x + 258 + L, yy + 6), fmt(v), font=font(16), fill=TEXT)


def evaluation():
    r2 = json.loads((ROOT / "artifacts/metrics/phone_coco_round2_test.json").read_text())["rows"]
    bm = json.loads((ROOT / "artifacts/metrics/benchmark_final.json").read_text())["rows"]
    pick = lambda strat, cond: next(r for r in r2 if r["model"] == "yolo11n" and r["condition"] == cond and r["strategy"] == strat)["all"]["AP50"]  # noqa: E731
    W, H = 1600, 920
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((40, 28), "Evaluation overview - measured values only", font=font(34, True), fill=TEXT)
    d.text((40, 76), "Left: REAL labelled COCO images (held-out test split), small-phone stress.  Right: controlled speed benchmark, 1080p, Apple M2 8 GB.", font=font(17), fill=AMBER)
    items = [("full-frame @640 (original)", pick("full@640", "far"), BLUE), ("640 + crop416 (shipped earlier)", pick("union full640+crop416 [shipped]", "far"), AMBER),
             ("native 1920 (new)", pick("full@1920", "far"), GREEN), ("1280 + crop640 union", pick("union full1280+crop640", "far"), GREEN)]
    bars(d, 40, 170, 720, 180, items, 0.6, "Phone AP50, yolo11n, 'far' condition (test)", lambda v: f"{v:.2f}")
    sub = lambda c: [r for r in bm if r["clip"] == "street_1080p" and r["config"].startswith(c)][0]["end_to_end_fps"]  # noqa: E731
    sp = [("BEFORE 640+crop416", sub("BEFORE: 640 + crop416"), AMBER), ("BEFORE-fast 640", sub("BEFORE (fast)"), BLUE),
          ("AFTER default (native+crop640)", sub("AFTER default"), GREEN), ("AFTER-fast (native)", sub("AFTER fast"), GREEN), ("AFTER-accurate (yolo11s)", sub("AFTER accurate"), RED)]
    bars(d, 820, 170, 740, 230, sp, 70, "End-to-end speed on 1080p street clip (fps)", lambda v: f"{v:.1f}")
    notes = ["NOT EVALUATED (no labelled real exam footage): phone events in real exams, real false-alert rate, tracking HOTA.",
             "Synthetic composites (pasted CC0 phones, real MOT15 footage): untouched set F3 found 1/3 staged events (shipped-earlier: 0/3), 0 clean-twin false alerts.",
             "Tracking on real MOT15 ground truth: IDF1 0.77-0.80 for every sensible config on 3 test sequences - differences are within noise.",
             "Default costs ~2.7x processing time vs the earlier shipped config (5-8 fps on 1080p); native resolution alone is free. Soak: 10 min video, no memory growth.",
             "Two real false alerts found (face mask - fixed by a 0.5 s minimum duration; 6x10 px head blob - not fixed). Every alert needs human review."]
    d.rounded_rectangle((40, 520, 1560, 880), 14, fill=PANEL, outline=LINE, width=3)
    for i, t in enumerate(notes):
        d.text((64, 548 + i * 64), "- " + t, font=font(18), fill=AMBER if i == 0 else TEXT)
    im.save(OUT / "09_evaluation_overview.png")


if __name__ == "__main__":
    arch(); evaluation(); print("figures written")
