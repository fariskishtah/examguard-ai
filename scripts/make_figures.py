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
    steps = [("Video input", "OpenCV decode\nstreamed, true\ntimestamps", "video/reader"), ("YOLO11n", "person + cell phone\nMPS / CPU\n(+ person-crop assist)", "detection/"),
             ("ByteTrack", "anonymous\ntrack IDs\n(no identity)", "tracking/"), ("Event engine", "association,\ntemporal confirm,\nlifecycle, cooldown", "events/"),
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
    ph = json.loads((ROOT / "artifacts/metrics/phone_fixture_eval.json").read_text())["rows"]
    bm = json.loads((ROOT / "artifacts/metrics/benchmark.json").read_text())["rows"]
    pick = lambda **kw: next(r for r in ph if all(r[k] == v for k, v in kw.items()))  # noqa: E731
    base = dict(imgsz=640, phone_conf=0.25, support_frames=5, resolve_after_s=2.0)
    off, on = pick(crop_assist=False, **base), pick(crop_assist=True, **base)
    W, H = 1600, 900
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((40, 28), "Evaluation overview - measured values only", font=font(34, True), fill=TEXT)
    d.text((40, 76), "Phone panel = SYNTHETIC composite fixture (not real-world accuracy).  Mac M2 8 GB, yolo11n, 640 px.", font=font(18), fill=AMBER)
    bars(d, 40, 170, 720, 180, [("frame recall - full frame", off["frame_phone_recall"], BLUE), ("frame recall - crop assist", on["frame_phone_recall"], GREEN),
                                ("window covered - full frame", off["gt_window_alert_coverage"], BLUE), ("window covered - crop assist", on["gt_window_alert_coverage"], GREEN)],
         1.0, "Phone events (synthetic fixture, conf 0.25)", lambda v: f"{v:.2f}")
    d.text((40, 372), f"precision: full-frame {off['frame_phone_precision']:.2f} | crop-assist {on['frame_phone_precision']:.2f};  "
                      f"alerts correct/false: {off['incidents_correct']}/{off['incidents_false']} vs {on['incidents_correct']}/{on['incidents_false']}", font=font(16), fill=MUTED)
    rows = [r for r in bm if r["clip"] in ("street_1080p", "classroom_1080p", "classroom_360p")]
    items = []
    for r in rows:
        if r["annotated_video_written"]:
            continue
        col = GREEN if r["phone_crop_assist"] else BLUE
        items.append((f"{r['clip']} {r['device'].upper()} {'assist' if r['phone_crop_assist'] else 'plain'}", r["end_to_end_fps"], col if r["device"] == "mps" else AMBER))
    bars(d, 820, 170, 740, 330, items, 70, "End-to-end speed (frames/s)", lambda v: f"{v:.1f}")
    notes = ["NOT EVALUATED (no labelled real data): person/phone mAP, tracking IDF1/HOTA/MOTA, real event precision/recall, false alerts per hour.",
             "Normal footage: 0 alerts on 25 s of real clips (classroom + street) - an observation, not a false-alert rate.",
             "Tracking proxy: seated classroom >= 11 people, 19 IDs created (ByteTrack defaults gave 32) - IDs still fragment.",
             "Phone recall is weak (small objects); crop assist helps (0.10 -> 0.30 frame recall) at ~3x inference cost."]
    d.rounded_rectangle((40, 560, 1560, 850), 14, fill=PANEL, outline=LINE, width=3)
    for i, t in enumerate(notes):
        d.text((64, 590 + i * 58), "• " + t, font=font(19), fill=TEXT if i else AMBER)
    im.save(OUT / "09_evaluation_overview.png")


if __name__ == "__main__":
    arch(); evaluation(); print("figures written")
