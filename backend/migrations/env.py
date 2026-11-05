"""Alembic environment: uses the application's settings and ORM metadata."""
from __future__ import annotations

import sys
from pathlib import Path

from alembic import context

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

from app.core.config import get_settings  # noqa: E402
from app.database.session import Base, engine  # noqa: E402
from app.models import entities  # noqa: E402,F401  (registers every table on Base.metadata)

config = context.config
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(url=get_settings().database_url, target_metadata=target_metadata,
                      literal_binds=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    conn = config.attributes.get("connection")  # tests may pass an open Connection
    if conn is not None:
        _run(conn)
        return
    with engine.connect() as c:
        _run(c)


def _run(conn) -> None:
    # render_as_batch lets ALTER TABLE work on SQLite as well as PostgreSQL
    context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
