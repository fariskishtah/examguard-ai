"""initial schema

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

REVIEW_STATUSES = "'PENDING_REVIEW','CONFIRMED_OBSERVATION','DISMISSED','NEEDS_MORE_EVIDENCE'"
DECISIONS = "'CONFIRMED_OBSERVATION','DISMISSED','NEEDS_MORE_EVIDENCE'"
SESSION_STATUSES = "'PENDING','PROCESSING','COMPLETED','STOPPED','FAILED'"


def upgrade() -> None:
    op.create_table(
        "video_sessions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("processing_started_at", sa.DateTime),
        sa.Column("processing_completed_at", sa.DateTime),
        sa.Column("duration_seconds", sa.Float),
        sa.Column("frame_count", sa.Integer),
        sa.Column("fps", sa.Float),
        sa.Column("width", sa.Integer),
        sa.Column("height", sa.Integer),
        sa.Column("processing_status", sa.String(16), nullable=False),
        sa.Column("detector_version", sa.String(255)),
        sa.Column("tracker_version", sa.String(64)),
        sa.Column("settings_json", sa.Text),
        sa.Column("annotated_video_path", sa.Text),
        sa.CheckConstraint(f"processing_status IN ({SESSION_STATUSES})", name="ck_session_status"),
    )
    op.create_index("ix_video_sessions_file_hash", "video_sessions", ["file_hash"])
    op.create_table(
        "tracks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("video_session_id", sa.Integer, sa.ForeignKey("video_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("anonymous_track_id", sa.Integer, nullable=False),
        sa.Column("first_seen_timestamp", sa.Float, nullable=False),
        sa.Column("last_seen_timestamp", sa.Float, nullable=False),
        sa.UniqueConstraint("video_session_id", "anonymous_track_id", name="uq_track_per_session"),
    )
    op.create_index("ix_tracks_video_session_id", "tracks", ["video_session_id"])
    op.create_table(
        "incidents",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("video_session_id", sa.Integer, sa.ForeignKey("video_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("track_id", sa.Integer, sa.ForeignKey("tracks.id", ondelete="SET NULL")),
        sa.Column("anonymous_track_id", sa.Integer, nullable=False),
        sa.Column("event_type", sa.String(48), nullable=False),
        sa.Column("first_detected_at", sa.Float, nullable=False),
        sa.Column("last_detected_at", sa.Float, nullable=False),
        sa.Column("first_frame", sa.Integer, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("person_confidence", sa.Float),
        sa.Column("lifecycle_state", sa.String(12), nullable=False),
        sa.Column("review_status", sa.String(24), nullable=False),
        sa.Column("region_id", sa.String(64)),
        sa.Column("screenshot_path", sa.Text),
        sa.Column("raw_screenshot_path", sa.Text),
        sa.Column("evidence_json", sa.Text),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
        sa.CheckConstraint(f"review_status IN ({REVIEW_STATUSES})", name="ck_incident_review_status"),
        sa.CheckConstraint("lifecycle_state IN ('ACTIVE','RESOLVED')", name="ck_incident_state"),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_incident_conf"),
    )
    op.create_index("ix_incidents_video_session_id", "incidents", ["video_session_id"])
    op.create_index("ix_incidents_track_id", "incidents", ["track_id"])
    op.create_index("ix_incidents_event_type", "incidents", ["event_type"])
    op.create_index("ix_incidents_review_status", "incidents", ["review_status"])
    op.create_index("ix_incidents_created_at", "incidents", ["created_at"])
    op.create_index("ix_incidents_type_status", "incidents", ["event_type", "review_status"])
    op.create_table(
        "incident_reviews",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("incident_id", sa.Integer, sa.ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("review_decision", sa.String(24), nullable=False),
        sa.Column("reviewer_note", sa.Text),
        sa.Column("reviewed_at", sa.DateTime, nullable=False),
        sa.CheckConstraint(f"review_decision IN ({DECISIONS})", name="ck_review_decision"),
    )
    op.create_index("ix_incident_reviews_incident_id", "incident_reviews", ["incident_id"])


def downgrade() -> None:
    for t in ("incident_reviews", "incidents", "tracks", "video_sessions"):
        op.drop_table(t)
