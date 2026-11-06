from alembic import context
from sqlalchemy import create_engine, event

config = context.config


def run() -> None:
    engine = create_engine(config.get_main_option("sqlalchemy.url"))
    event.listen(engine, "connect", lambda c, _: c.execute("PRAGMA foreign_keys=ON"))
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=None, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


run()
