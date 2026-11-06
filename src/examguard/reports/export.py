"""Exports. Everything is labelled as machine-generated observations pending human review."""
from __future__ import annotations

import base64
import csv
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2

from examguard.config import APP_NAME, APP_SUBTITLE, DISCLAIMER
from examguard.database.models import Incident, VideoSession
from examguard.database.repository import Repository
from examguard.events.engine import EVENT_LABELS
from examguard.video.annotate import fmt_ts

CSV_COLUMNS = ["incident_id", "session_id", "video_filename", "anonymous_track_id", "event_type", "first_detected_video_time",
               "last_detected_video_time", "confidence", "lifecycle_state", "review_status", "region_id",
               "screenshot_path", "raw_screenshot_path", "created_at_utc", "latest_reviewer_note", "notice"]
NOTICE = "Machine-generated observation pending human review; not proof of misconduct."


def incident_dict(inc: Incident, with_reviews: bool = False) -> dict:
    d = {
        "incident_id": inc.id, "session_id": inc.video_session_id, "video_filename": inc.session.filename,
        "anonymous_track_id": inc.anonymous_track_id, "event_type": inc.event_type,
        "first_detected_video_time": round(inc.first_detected_at, 3), "last_detected_video_time": round(inc.last_detected_at, 3),
        "confidence": round(inc.confidence, 4), "lifecycle_state": inc.lifecycle_state, "review_status": inc.review_status,
        "region_id": inc.region_id, "screenshot_path": inc.screenshot_path, "raw_screenshot_path": inc.raw_screenshot_path,
        "created_at_utc": inc.created_at.isoformat(timespec="seconds"), "notice": NOTICE,
    }
    revs = inc.reviews
    d["latest_reviewer_note"] = revs[-1].reviewer_note if revs else None
    if with_reviews:
        d["review_history"] = [{"decision": r.review_decision, "note": r.reviewer_note,
                                "reviewed_at_utc": r.reviewed_at.isoformat(timespec="seconds")} for r in revs]
        d["evidence"] = json.loads(inc.evidence_json) if inc.evidence_json else None
    return d


def _csv_safe(v):
    """Neutralise spreadsheet formula injection from free text."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def _load(repo: Repository, incidents: list[Incident]) -> list[Incident]:
    return [repo.get_incident(i.id) for i in incidents]  # eager-load reviews/session


def export_csv(repo: Repository, incidents: list[Incident], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for inc in _load(repo, incidents):
            w.writerow({k: _csv_safe(v) for k, v in incident_dict(inc).items()})
    return path


def export_json(repo: Repository, incidents: list[Incident], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_by": APP_NAME, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "disclaimer": DISCLAIMER,
               "incidents": [incident_dict(i, with_reviews=True) for i in _load(repo, incidents)]}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def export_screenshots(incidents: list[Incident], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for inc in incidents:
        for p in (inc.screenshot_path, inc.raw_screenshot_path):
            if p and Path(p).is_file():
                shutil.copy2(p, dest / Path(p).name)
                n += 1
    return n


def export_annotated_video(session: VideoSession, dest_dir: Path) -> Path | None:
    if not session.annotated_video_path or not Path(session.annotated_video_path).is_file():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / Path(session.annotated_video_path).name
    shutil.copy2(session.annotated_video_path, out)
    return out


def _thumb_b64(path: str | None, width: int = 520) -> str | None:
    if not path or not Path(path).is_file():
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    s = width / img.shape[1]
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def export_report_html(repo: Repository, session_id: int, path: Path, limitations: list[str] | None = None) -> Path:
    s = repo.get_session(session_id)
    if s is None:
        raise KeyError(session_id)
    from examguard.database.repository import IncidentFilter
    incs = _load(repo, repo.list_incidents(IncidentFilter(session_id=session_id)))
    e = html.escape
    by_type: dict[str, int] = {}
    for i in incs:
        by_type[i.event_type] = by_type.get(i.event_type, 0) + 1
    limitations = limitations or [
        "Detections come from a generic COCO-pretrained YOLO model that was not fine-tuned for exam scenes.",
        "Phone detection can miss small, occluded or unusual-angle phones and can confuse similar objects.",
        "Track IDs are temporary and anonymous; they can switch after occlusion or when people cross.",
        "An alert records an observable event only. Context and human judgement are required.",
    ]
    rows = []
    for i in sorted(incs, key=lambda x: x.first_detected_at):
        b64 = _thumb_b64(i.screenshot_path)
        img = f'<img src="data:image/jpeg;base64,{b64}" alt="evidence">' if b64 else "<em>evidence file unavailable</em>"
        note = e(i.reviews[-1].reviewer_note or "") if i.reviews else ""
        rows.append(f"<tr><td>#{i.id}</td><td>Track #{i.anonymous_track_id}</td><td>{e(EVENT_LABELS.get(i.event_type, i.event_type))}</td>"
                    f"<td>{fmt_ts(i.first_detected_at)}</td><td>{i.confidence:.2f}</td><td>{e(i.review_status.replace('_', ' ').title())}</td>"
                    f"<td>{note}</td><td>{img}</td></tr>")
    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>{APP_NAME} - Session {s.id} report</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:32px;color:#0b1b33}}h1{{margin:0}}
.sub{{color:#4a5d7a}}.warn{{background:#fff4d6;border-left:5px solid #f0a000;padding:10px 14px;margin:18px 0}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cfd8e6;padding:6px 8px;vertical-align:top;font-size:13px}}
th{{background:#0b1b33;color:#fff;text-align:left}}img{{max-width:260px}}.kv td:first-child{{font-weight:600;width:220px}}</style></head><body>
<h1>{APP_NAME}</h1><div class="sub">{e(APP_SUBTITLE)}</div><h2>Incident report - session {s.id}</h2>
<div class="warn"><b>{e(DISCLAIMER)}</b><br>All entries below are machine-generated observations pending human review.</div>
<table class="kv"><tr><td>Video file</td><td>{e(s.filename)}</td></tr><tr><td>File SHA-256</td><td>{e(s.file_hash)}</td></tr>
<tr><td>Resolution / FPS / duration</td><td>{s.width}x{s.height} / {s.fps and round(s.fps, 3)} / {s.duration_seconds and round(s.duration_seconds, 1)} s</td></tr>
<tr><td>Frames processed</td><td>{s.frame_count}</td></tr><tr><td>Processing status</td><td>{e(s.processing_status)}</td></tr>
<tr><td>Processed (UTC)</td><td>{s.processing_started_at} - {s.processing_completed_at}</td></tr>
<tr><td>Detector version</td><td>{e(s.detector_version or "")}</td></tr><tr><td>Tracker version</td><td>{e(s.tracker_version or "")}</td></tr>
<tr><td>Anonymous tracks monitored</td><td>{repo.count_tracks(s.id)}</td></tr><tr><td>Flagged events</td><td>{len(incs)} {e(json.dumps(by_type))}</td></tr>
<tr><td>Report generated (UTC)</td><td>{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}</td></tr></table>
<h3>Flagged events</h3><table><tr><th>ID</th><th>Track</th><th>Event</th><th>Video time</th><th>Conf.</th><th>Review status</th><th>Reviewer note</th><th>Evidence</th></tr>
{''.join(rows) or '<tr><td colspan=8>No events were flagged.</td></tr>'}</table>
<h3>Technical limitations</h3><ul>{''.join(f'<li>{e(x)}</li>' for x in limitations)}</ul>
<p class="sub">Track IDs are anonymous, temporary identifiers. No identity verification or face recognition is performed.</p></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path
