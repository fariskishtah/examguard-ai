"""Persistence operations used by the pipeline and the UI."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from examguard.database.db import Database
from examguard.database.models import (REVIEW_DECISIONS, REVIEW_PENDING, Incident, IncidentReview, Track,
                                       VideoSession, utcnow)


class InvalidReview(ValueError):
    pass


@dataclass
class IncidentFilter:
    session_id: int | None = None
    event_type: str | None = None
    review_status: str | None = None
    track_id: int | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    text: str | None = None


class Repository:
    def __init__(self, db: Database):
        self.db = db

    # ---- sessions ------------------------------------------------------
    def create_session(self, filename: str, file_hash: str, **meta) -> int:
        with self.db.session() as s:
            vs = VideoSession(filename=Path(filename).name[:255], file_hash=file_hash, **meta)
            s.add(vs)
            s.flush()
            return vs.id

    def update_session(self, session_id: int, **fields) -> None:
        with self.db.session() as s:
            vs = s.get(VideoSession, session_id)
            if vs is None:
                raise KeyError(session_id)
            for k, v in fields.items():
                setattr(vs, k, v)

    def get_session(self, session_id: int) -> VideoSession | None:
        with self.db.session() as s:
            return s.get(VideoSession, session_id)

    def list_sessions(self) -> list[VideoSession]:
        with self.db.session() as s:
            return list(s.scalars(select(VideoSession).order_by(VideoSession.id.desc())))

    # ---- tracks --------------------------------------------------------
    def upsert_track(self, session_id: int, anon_id: int, first_ts: float, last_ts: float) -> int:
        with self.db.session() as s:
            t = s.scalar(select(Track).where(Track.video_session_id == session_id,
                                             Track.anonymous_track_id == anon_id))
            if t is None:
                t = Track(video_session_id=session_id, anonymous_track_id=anon_id,
                          first_seen_timestamp=first_ts, last_seen_timestamp=last_ts)
                s.add(t)
            else:
                t.first_seen_timestamp = min(t.first_seen_timestamp, first_ts)
                t.last_seen_timestamp = max(t.last_seen_timestamp, last_ts)
            s.flush()
            return t.id

    def save_tracks(self, session_id: int, spans: dict[int, tuple[float, float]]) -> None:
        for anon, (a, b) in spans.items():
            self.upsert_track(session_id, anon, a, b)

    def count_tracks(self, session_id: int) -> int:
        with self.db.session() as s:
            return s.scalar(select(func.count()).select_from(Track).where(Track.video_session_id == session_id)) or 0

    # ---- incidents -----------------------------------------------------
    def create_incident(self, session_id: int, anon_track_id: int, event_type: str, first_ts: float,
                        first_frame: int, confidence: float, person_conf: float | None = None,
                        screenshot_path: str | None = None, raw_screenshot_path: str | None = None,
                        region_id: str | None = None, evidence: dict | None = None) -> int:
        with self.db.session() as s:
            tid = s.scalar(select(Track.id).where(Track.video_session_id == session_id,
                                                  Track.anonymous_track_id == anon_track_id))
            if tid is None:
                t = Track(video_session_id=session_id, anonymous_track_id=anon_track_id,
                          first_seen_timestamp=first_ts, last_seen_timestamp=first_ts)
                s.add(t)
                s.flush()
                tid = t.id
            inc = Incident(video_session_id=session_id, track_id=tid, anonymous_track_id=anon_track_id,
                           event_type=event_type, first_detected_at=first_ts, last_detected_at=first_ts,
                           first_frame=first_frame, confidence=max(0.0, min(1.0, confidence)),
                           person_confidence=person_conf, screenshot_path=screenshot_path,
                           raw_screenshot_path=raw_screenshot_path, region_id=region_id,
                           evidence_json=json.dumps(evidence) if evidence else None)
            s.add(inc)
            s.flush()
            return inc.id

    def update_incident(self, incident_id: int, **fields) -> None:
        if "evidence" in fields:
            fields["evidence_json"] = json.dumps(fields.pop("evidence"))
        if "confidence" in fields:
            fields["confidence"] = max(0.0, min(1.0, fields["confidence"]))
        with self.db.session() as s:
            inc = s.get(Incident, incident_id)
            if inc is None:
                raise KeyError(incident_id)
            for k, v in fields.items():
                setattr(inc, k, v)

    def get_incident(self, incident_id: int) -> Incident | None:
        with self.db.session() as s:
            inc = s.get(Incident, incident_id)
            if inc:
                _ = inc.reviews, inc.session  # load before session closes
            return inc

    def _filtered(self, f: IncidentFilter):
        q = select(Incident)
        if f.session_id is not None: q = q.where(Incident.video_session_id == f.session_id)
        if f.event_type: q = q.where(Incident.event_type == f.event_type)
        if f.review_status: q = q.where(Incident.review_status == f.review_status)
        if f.track_id is not None: q = q.where(Incident.anonymous_track_id == f.track_id)
        if f.date_from: q = q.where(Incident.created_at >= f.date_from)
        if f.date_to: q = q.where(Incident.created_at < f.date_to + timedelta(days=1))
        if f.text:
            like = f"%{f.text}%"
            q = q.where(or_(Incident.event_type.ilike(like), Incident.review_status.ilike(like),
                            Incident.region_id.ilike(like)))
        return q

    def list_incidents(self, f: IncidentFilter | None = None, limit: int = 1000) -> list[Incident]:
        with self.db.session() as s:
            q = self._filtered(f or IncidentFilter()).order_by(Incident.created_at.desc(), Incident.id.desc()).limit(limit)
            rows = list(s.scalars(q))
            for r in rows:
                _ = r.session.filename
            return rows

    # ---- review --------------------------------------------------------
    def review(self, incident_id: int, decision: str, note: str | None = None) -> int:
        if decision not in REVIEW_DECISIONS:
            raise InvalidReview(f"decision must be one of {REVIEW_DECISIONS}, got {decision!r}")
        with self.db.session() as s:
            inc = s.get(Incident, incident_id)
            if inc is None:
                raise KeyError(incident_id)
            r = IncidentReview(incident_id=incident_id, review_decision=decision,
                               reviewer_note=(note or "").strip()[:2000] or None)
            s.add(r)
            inc.review_status = decision
            inc.updated_at = utcnow()
            s.flush()
            return r.id

    def reviews_for(self, incident_id: int) -> list[IncidentReview]:
        with self.db.session() as s:
            return list(s.scalars(select(IncidentReview).where(IncidentReview.incident_id == incident_id)
                                  .order_by(IncidentReview.id)))

    # ---- stats / retention ----------------------------------------------
    def stats(self) -> dict:
        with self.db.session() as s:
            def cnt(*where):
                return s.scalar(select(func.count()).select_from(Incident).where(*where)) or 0
            last = s.scalar(select(VideoSession).order_by(VideoSession.id.desc()).limit(1))
            return {
                "videos": s.scalar(select(func.count()).select_from(VideoSession)) or 0,
                "incidents": cnt(),
                "pending": cnt(Incident.review_status == REVIEW_PENDING),
                "dismissed": cnt(Incident.review_status == "DISMISSED"),
                "confirmed": cnt(Incident.review_status == "CONFIRMED_OBSERVATION"),
                "needs_more": cnt(Incident.review_status == "NEEDS_MORE_EVIDENCE"),
                "last_session": (last.filename, last.processing_status, last.created_at) if last else None,
            }

    def delete_older_than(self, days: int) -> list[str]:
        """Delete sessions (and cascaded incidents) older than `days`; return evidence paths to remove."""
        if days <= 0:
            return []
        cutoff = utcnow() - timedelta(days=days)
        paths: list[str] = []
        with self.db.session() as s:
            for vs in s.scalars(select(VideoSession).where(VideoSession.created_at < cutoff)):
                for inc in vs.incidents:
                    paths += [p for p in (inc.screenshot_path, inc.raw_screenshot_path) if p]
                if vs.annotated_video_path:
                    paths.append(vs.annotated_video_path)
                s.delete(vs)
        return paths

    def delete_session(self, session_id: int) -> list[str]:
        paths: list[str] = []
        with self.db.session() as s:
            vs = s.get(VideoSession, session_id)
            if vs:
                for inc in vs.incidents:
                    paths += [p for p in (inc.screenshot_path, inc.raw_screenshot_path) if p]
                if vs.annotated_video_path:
                    paths.append(vs.annotated_video_path)
                s.delete(vs)
        return paths
