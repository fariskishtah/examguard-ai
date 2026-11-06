"""Engine/session factory. SQLite with foreign keys enforced; schema managed by Alembic."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def make_engine(db_path: Path | str) -> Engine:
    url = "sqlite://" if str(db_path) == ":memory:" else f"sqlite:///{db_path}"
    engine = create_engine(url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

    return engine


def upgrade_to_head(db_path: Path | str) -> None:
    from alembic import command
    from alembic.config import Config

    if str(db_path) == ":memory:":
        raise ValueError("migrations need a file database")
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")


class Database:
    def __init__(self, db_path: Path | str):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        upgrade_to_head(db_path)
        self.path = db_path
        self.engine = make_engine(db_path)
        self._factory = sessionmaker(self.engine, expire_on_commit=False)
        try:
            db_path.chmod(0o600)
        except OSError:
            pass

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional scope: commit on success, rollback on error."""
        s = self._factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def close(self) -> None:
        self.engine.dispose()
