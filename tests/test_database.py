import sqlite3
from datetime import timedelta

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from examguard.database.db import Database
from examguard.database.models import Base, REVIEW_CONFIRMED, REVIEW_DISMISSED, REVIEW_PENDING, VideoSession, utcnow
from examguard.database.repository import IncidentFilter, InvalidReview, Repository


@pytest.fixture()
def repo(tmp_path):
    db = Database(tmp_path / "t.db")
    yield Repository(db)
    db.close()


def mk_incident(repo, sid=None, **kw):
    sid = sid or repo.create_session("a.mp4", "h" * 64)
    args = dict(session_id=sid, anon_track_id=3, event_type="PHONE_VISIBLE_NEAR_TRACK", first_ts=42.0,
                first_frame=420, confidence=0.7)
    args.update(kw)
    return sid, repo.create_incident(**args)


def test_migration_matches_models(tmp_path):
    db = Database(tmp_path / "m.db")
    insp = inspect(db.engine)
    for name, table in Base.metadata.tables.items():
        assert {c["name"] for c in insp.get_columns(name)} == {c.name for c in table.columns}, name
    assert "alembic_version" in insp.get_table_names()
    db.close()


def test_migration_is_idempotent_and_persists(tmp_path):
    p = tmp_path / "x.db"
    r1 = Repository(Database(p)); sid = r1.create_session("v.mp4", "a" * 64)
    r2 = Repository(Database(p))  # re-open => upgrade is a no-op, data intact
    assert r2.get_session(sid).filename == "v.mp4"


def test_foreign_keys_enforced(repo):
    with pytest.raises(IntegrityError):
        with repo.db.session() as s:
            s.execute(text("INSERT INTO tracks(video_session_id, anonymous_track_id, first_seen_timestamp,"
                           " last_seen_timestamp) VALUES (999, 1, 0, 0)"))


def test_check_constraints(repo):
    sid = repo.create_session("a.mp4", "h" * 64)
    with pytest.raises(IntegrityError):
        with repo.db.session() as s:
            s.execute(text("UPDATE video_sessions SET processing_status='BOGUS' WHERE id=:i"), {"i": sid})
    sid, iid = mk_incident(repo, sid)
    with pytest.raises(IntegrityError):
        with repo.db.session() as s:
            s.execute(text("UPDATE incidents SET review_status='GUILTY' WHERE id=:i"), {"i": iid})


def test_unique_track_per_session_and_update(repo):
    sid = repo.create_session("a.mp4", "h" * 64)
    a = repo.upsert_track(sid, 1, 2.0, 3.0)
    b = repo.upsert_track(sid, 1, 1.0, 9.0)
    assert a == b and repo.count_tracks(sid) == 1


def test_transaction_rolls_back_on_error(repo):
    with pytest.raises(RuntimeError):
        with repo.db.session() as s:
            s.add(VideoSession(filename="x", file_hash="y" * 64)); s.flush(); raise RuntimeError
    assert repo.list_sessions() == []


def test_incident_persist_and_default_status(repo):
    sid, iid = mk_incident(repo, screenshot_path="/x/a.jpg", evidence={"frames": [1, 2]})
    inc = repo.get_incident(iid)
    assert inc.review_status == REVIEW_PENDING and inc.screenshot_path == "/x/a.jpg"
    assert inc.track_id is not None and inc.session.filename == "a.mp4"


def test_review_workflow_history_and_validation(repo):
    _, iid = mk_incident(repo)
    repo.review(iid, REVIEW_DISMISSED, "phone was a calculator")
    repo.review(iid, REVIEW_CONFIRMED, "  second look  ")
    assert repo.get_incident(iid).review_status == REVIEW_CONFIRMED
    hist = repo.reviews_for(iid)
    assert [h.review_decision for h in hist] == [REVIEW_DISMISSED, REVIEW_CONFIRMED]
    assert hist[1].reviewer_note == "second look"
    with pytest.raises(InvalidReview):
        repo.review(iid, REVIEW_PENDING)
    with pytest.raises(InvalidReview):
        repo.review(iid, "STUDENT_GUILTY")
    with pytest.raises(KeyError):
        repo.review(9999, REVIEW_DISMISSED)


def test_review_survives_restart(tmp_path):
    p = tmp_path / "r.db"
    r = Repository(Database(p)); _, iid = mk_incident(r); r.review(iid, REVIEW_DISMISSED, "n")
    r2 = Repository(Database(p))
    assert r2.get_incident(iid).review_status == REVIEW_DISMISSED
    assert r2.reviews_for(iid)[0].reviewer_note == "n"


def test_filters_and_stats(repo):
    sid, a = mk_incident(repo)
    _, b = mk_incident(repo, sid, anon_track_id=7, event_type="MONITORED_REGION_EXIT")
    repo.review(a, REVIEW_DISMISSED)
    assert len(repo.list_incidents()) == 2
    assert [i.id for i in repo.list_incidents(IncidentFilter(event_type="MONITORED_REGION_EXIT"))] == [b]
    assert [i.id for i in repo.list_incidents(IncidentFilter(track_id=3))] == [a]
    assert [i.id for i in repo.list_incidents(IncidentFilter(review_status=REVIEW_PENDING))] == [b]
    assert len(repo.list_incidents(IncidentFilter(session_id=sid))) == 2
    today = utcnow()
    assert len(repo.list_incidents(IncidentFilter(date_from=today - timedelta(days=1), date_to=today))) == 2
    assert repo.list_incidents(IncidentFilter(date_to=today - timedelta(days=3))) == []
    st = repo.stats()
    assert st["videos"] == 1 and st["incidents"] == 2 and st["pending"] == 1 and st["dismissed"] == 1


def test_retention_and_cascade_delete(repo):
    sid, iid = mk_incident(repo, screenshot_path="/e/1.jpg", raw_screenshot_path="/e/1_raw.jpg")
    repo.update_session(sid, created_at=utcnow() - timedelta(days=40))
    paths = repo.delete_older_than(30)
    assert set(paths) == {"/e/1.jpg", "/e/1_raw.jpg"}
    assert repo.list_incidents() == [] and repo.list_sessions() == []
    assert repo.delete_older_than(0) == []


def test_confidence_clamped(repo):
    _, iid = mk_incident(repo, confidence=1.7)
    assert repo.get_incident(iid).confidence == 1.0
