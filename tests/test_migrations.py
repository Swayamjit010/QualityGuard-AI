"""The Alembic migrations must build exactly the schema the ORM declares (no drift) and be reversible."""
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

alembic = pytest.importorskip("alembic")
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _cfg(conn):
    cfg = Config(str(ROOT / "backend" / "alembic.ini"))
    cfg.attributes["connection"] = conn
    return cfg


def test_upgrade_matches_orm_and_downgrades(tmp_path):
    from app.database.session import Base

    eng = create_engine(f"sqlite:///{(tmp_path / 'mig.db').as_posix()}")
    with eng.begin() as conn:
        command.upgrade(_cfg(conn), "head")
    tables = set(inspect(eng).get_table_names())
    assert set(Base.metadata.tables) <= tables and "alembic_version" in tables
    with eng.begin() as conn:
        command.check(_cfg(conn))  # raises if the models and the migrations differ
    with eng.begin() as conn:
        command.downgrade(_cfg(conn), "base")
    assert set(inspect(eng).get_table_names()) <= {"alembic_version"}
