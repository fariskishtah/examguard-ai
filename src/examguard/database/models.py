"""SQLAlchemy models. No student names/identities are stored - only anonymous track ids."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String, Text,
                        UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

REVIEW_PENDING = "PENDING_REVIEW"
REVIEW_CONFIRMED = "CONFIRMED_OBSERVATION"
REVIEW_DISMISSED = "DISMISSED"
REVIEW_MORE = "NEEDS_MORE_EVIDENCE"
REVIEW_STATUSES = (REVIEW_PENDING, REVIEW_CONFIRMED, REVIEW_DISMISSED, REVIEW_MORE)
REVIEW_DECISIONS = (REVIEW_CONFIRMED, REVIEW_DISMISSED, REVIEW_MORE)  # a reviewer cannot "decide" pending
SESSION_STATUSES = ("PENDING", "PROCESSING", "COMPLETED", "STOPPED", "FAILED")
INCIDENT_STATES = ("ACTIVE", "RESOLVED")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC (SQLite has no tz)


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    pass


class VideoSession(Base):
    __tablename__ = "video_sessions"
    __table_args__ = (CheckConstraint(_in("processing_status", SESSION_STATUSES), name="ck_session_status"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime)
    processing_completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    frame_count: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    processing_status: Mapped[str] = mapped_column(String(16), default="PENDING")
    detector_version: Mapped[str | None] = mapped_column(String(255))
    tracker_version: Mapped[str | None] = mapped_column(String(64))
    settings_json: Mapped[str | None] = mapped_column(Text)
    annotated_video_path: Mapped[str | None] = mapped_column(Text)
    tracks: Mapped[list["Track"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    incidents: Mapped[list["Incident"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Track(Base):
    __tablename__ = "tracks"
    __table_args__ = (UniqueConstraint("video_session_id", "anonymous_track_id", name="uq_track_per_session"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    video_session_id: Mapped[int] = mapped_column(ForeignKey("video_sessions.id", ondelete="CASCADE"), index=True)
    anonymous_track_id: Mapped[int] = mapped_column(Integer)
    first_seen_timestamp: Mapped[float] = mapped_column(Float)
    last_seen_timestamp: Mapped[float] = mapped_column(Float)
    session: Mapped[VideoSession] = relationship(back_populates="tracks")


class Incident(Base):
    __tablename__ = "incidents"
    __table_args__ = (
        CheckConstraint(_in("review_status", REVIEW_STATUSES), name="ck_incident_review_status"),
        CheckConstraint(_in("lifecycle_state", INCIDENT_STATES), name="ck_incident_state"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_incident_conf"),
        Index("ix_incidents_type_status", "event_type", "review_status"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    video_session_id: Mapped[int] = mapped_column(ForeignKey("video_sessions.id", ondelete="CASCADE"), index=True)
    track_id: Mapped[int | None] = mapped_column(ForeignKey("tracks.id", ondelete="SET NULL"), index=True)
    anonymous_track_id: Mapped[int] = mapped_column(Integer)  # denormalised for easy filtering
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    first_detected_at: Mapped[float] = mapped_column(Float)   # video timestamp (s)
    last_detected_at: Mapped[float] = mapped_column(Float)    # video timestamp (s)
    first_frame: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)
    person_confidence: Mapped[float | None] = mapped_column(Float)
    lifecycle_state: Mapped[str] = mapped_column(String(12), default="ACTIVE")
    review_status: Mapped[str] = mapped_column(String(24), default=REVIEW_PENDING, index=True)
    region_id: Mapped[str | None] = mapped_column(String(64))
    screenshot_path: Mapped[str | None] = mapped_column(Text)       # annotated review copy
    raw_screenshot_path: Mapped[str | None] = mapped_column(Text)   # unannotated evidence frame
    evidence_json: Mapped[str | None] = mapped_column(Text)         # supporting frame indices etc.
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    session: Mapped[VideoSession] = relationship(back_populates="incidents")
    reviews: Mapped[list["IncidentReview"]] = relationship(
        back_populates="incident", cascade="all, delete-orphan", order_by="IncidentReview.id")


class IncidentReview(Base):
    __tablename__ = "incident_reviews"
    __table_args__ = (CheckConstraint(_in("review_decision", REVIEW_DECISIONS), name="ck_review_decision"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    review_decision: Mapped[str] = mapped_column(String(24))
    reviewer_note: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    incident: Mapped[Incident] = relationship(back_populates="reviews")
