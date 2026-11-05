"""Idempotent startup initialisation: schema, roles/permissions, initial admin, profiles, KB, registry sync."""
from __future__ import annotations

import json
import logging
import secrets

from app.core.config import ROOT, get_settings
from app.database.session import Base, SessionLocal, engine
from app.models import entities as E
from app.security.auth import PERMISSIONS, ROLES, hash_password

log = logging.getLogger("qualityguard.bootstrap")

DEMO_USERS = [
    ("manager@qualityguard.local", "Quality Manager (demo)", "quality_manager"),
    ("inspector@qualityguard.local", "Inspector (demo)", "inspector"),
    ("mlengineer@qualityguard.local", "ML Engineer (demo)", "ml_engineer"),
    ("viewer@qualityguard.local", "Viewer (demo)", "viewer"),
]


def run_migrations() -> None:
    """Bring the schema to the latest Alembic revision. A database created earlier with create_all (no
    alembic_version table but existing tables) is stamped at the initial revision first."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect as sa_inspect

    cfg = Config(str(ROOT / "backend" / "alembic.ini"))
    tables = set(sa_inspect(engine).get_table_names())
    if tables and "alembic_version" not in tables:
        command.stamp(cfg, "0001")
    command.upgrade(cfg, "head")


def add_missing_nullable_columns() -> list[str]:
    """create_all() creates new tables but never alters existing ones. For databases created that way, add any
    column the models declare that the table lacks - only if it is NULLABLE (purely additive, no data change).
    Unique indexes for such columns are created too. Anything else must go through Alembic migrations."""
    from sqlalchemy import inspect as sa_inspect, text
    insp = sa_inspect(engine)
    existing = set(insp.get_table_names())
    added = []
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing:
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have or not col.nullable:
                    continue
                ctype = col.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ctype}'))
                added.append(f"{table.name}.{col.name}")
                for idx in table.indexes:
                    if [c.name for c in idx.columns] == [col.name]:
                        uniq = "UNIQUE " if idx.unique else ""
                        conn.execute(text(f'CREATE {uniq}INDEX IF NOT EXISTS "{idx.name}" ON "{table.name}" ("{col.name}")'))
    return added


def init_db(create_all: bool = True):
    s = get_settings()
    if s.schema_mode == "migrations":
        run_migrations()
    elif create_all:
        Base.metadata.create_all(engine)
        added = add_missing_nullable_columns()
        if added:
            import logging
            logging.getLogger("qualityguard").info("schema: added nullable columns %s", added)
    db = SessionLocal()
    try:
        perms = {}
        for code, desc in PERMISSIONS.items():
            p = db.query(E.Permission).filter_by(code=code).first() or E.Permission(code=code, description=desc)
            db.add(p)
            perms[code] = p
        db.flush()
        roles = {}
        for name, (title, codes) in ROLES.items():
            r = db.query(E.Role).filter_by(name=name).first() or E.Role(name=name, title=title)
            r.permissions = [perms[c] for c in codes]
            db.add(r)
            roles[name] = r
        db.flush()
        creds_file = s.storage_dir / "INITIAL_CREDENTIALS.txt"
        if not db.query(E.User).filter_by(email=s.default_admin_email).first():
            pw = s.default_admin_password or ("Qg-" + secrets.token_urlsafe(12) + "9a")
            db.add(E.User(email=s.default_admin_email, full_name="Administrator", password_hash=hash_password(pw),
                          role_id=roles["admin"].id))
            lines = [f"admin  {s.default_admin_email}  {pw}"]
            if s.seed_demo_users:
                for email, name, role in DEMO_USERS:
                    dpw = "Qg-" + secrets.token_urlsafe(10) + "7b"
                    db.add(E.User(email=email, full_name=name, password_hash=hash_password(dpw), role_id=roles[role].id))
                    lines.append(f"{role}  {email}  {dpw}")
            creds_file.write_text("QualityGuard AI initial accounts (change these passwords; delete this file):\n" +
                                  "\n".join(lines) + "\n")
            log.warning("Initial accounts created. Credentials written to %s", creds_file)
        db.commit()

        from app.services import datasets, model_registry, profiles
        profiles.seed_profiles(db)
        kb = ROOT / "configs" / "knowledge_base.json"
        if kb.exists() and not db.query(E.KnowledgeEntry).count():
            for e in json.loads(kb.read_text(encoding="utf-8")):
                db.add(E.KnowledgeEntry(**e))
            db.commit()
        if not db.get(E.SystemSetting, "alert_rules"):
            from app.services.alerts import DEFAULT_RULES
            db.add(E.SystemSetting(key="alert_rules", value=DEFAULT_RULES))
        for key, val in {"report_preferences": {"company_name": "QualityGuard AI", "include_heatmaps": True},
                         "notification_preferences": {"webhook_enabled": bool(s.webhook_url), "events": ["alert"]},
                         "camera_settings": {"capture_interval_ms": 1500, "resolution": "1280x720"},
                         "storage_settings": {"retention_days": s.data_retention_days}}.items():
            if not db.get(E.SystemSetting, key):
                db.add(E.SystemSetting(key=key, value=val))
        db.commit()
        datasets.sync(db)
        model_registry.sync(db)
        model_registry.bootstrap_deployments(db)
    finally:
        db.close()
