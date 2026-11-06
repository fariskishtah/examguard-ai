import csv
import json

import numpy as np
import cv2
import pytest

from examguard.database.db import Database
from examguard.database.repository import IncidentFilter, Repository
from examguard.reports.export import (export_annotated_video, export_csv, export_json, export_report_html,
                                      export_screenshots)


@pytest.fixture()
def repo(tmp_path):
    r = Repository(Database(tmp_path / "e.db"))
    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.full((60, 80, 3), 90, np.uint8))
    sid = r.create_session("exam.mp4", "f" * 64, fps=25.0, width=80, height=60, duration_seconds=10.0, frame_count=250)
    a = r.create_incident(sid, 4, "PHONE_VISIBLE_NEAR_TRACK", 3.2, 80, 0.61, 0.9, str(img), None)
    r.create_incident(sid, 5, "MONITORED_REGION_EXIT", 5.0, 125, 0.8, region_id="region-1", screenshot_path="/missing/x.jpg")
    r.review(a, "DISMISSED", "=HYPERLINK(\"http://evil\")")
    r.update_session(sid, annotated_video_path=str(img))
    return r


def test_csv_content_and_formula_injection_guard(repo, tmp_path):
    p = export_csv(repo, repo.list_incidents(), tmp_path / "o.csv")
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 2 and all("not proof of misconduct" in r["notice"] for r in rows)
    notes = [r["latest_reviewer_note"] for r in rows]
    assert "'=HYPERLINK(\"http://evil\")" in notes
    assert {r["event_type"] for r in rows} == {"PHONE_VISIBLE_NEAR_TRACK", "MONITORED_REGION_EXIT"}


def test_json_has_history_and_disclaimer(repo, tmp_path):
    d = json.loads(export_json(repo, repo.list_incidents(), tmp_path / "o.json").read_text())
    assert "not proof of academic misconduct" in d["disclaimer"] and len(d["incidents"]) == 2
    rev = [i for i in d["incidents"] if i["review_history"]][0]["review_history"]
    assert rev[0]["decision"] == "DISMISSED"


def test_filtered_export_only_contains_filtered(repo, tmp_path):
    only = repo.list_incidents(IncidentFilter(event_type="MONITORED_REGION_EXIT"))
    rows = list(csv.DictReader(open(export_csv(repo, only, tmp_path / "f.csv"))))
    assert len(rows) == 1 and rows[0]["event_type"] == "MONITORED_REGION_EXIT"


def test_screenshot_and_video_export_handle_missing_files(repo, tmp_path):
    assert export_screenshots(repo.list_incidents(), tmp_path / "shots") == 1  # missing file skipped, no crash
    s = repo.list_sessions()[0]
    assert export_annotated_video(s, tmp_path / "vid").exists()
    repo.update_session(s.id, annotated_video_path="/nope.mp4")
    assert export_annotated_video(repo.get_session(s.id), tmp_path / "vid2") is None


def test_html_report_contents_and_escaping(repo, tmp_path):
    sid = repo.list_sessions()[0].id
    html = export_report_html(repo, sid, tmp_path / "r.html").read_text()
    assert "not proof of academic misconduct" in html and "pending human review" in html
    assert "exam.mp4" in html and "Track #4" in html and "evidence file unavailable" in html
    assert "<script" not in html and "HYPERLINK" in html  # note is escaped text, not markup
