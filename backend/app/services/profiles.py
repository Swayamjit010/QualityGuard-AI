"""Product profiles: JSON files in configs/profiles seed the DB; the DB is the source of truth after."""
from __future__ import annotations

import copy
import json

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import ROOT
from app.models.entities import Product, ProductProfile, ProfileRevision, User

PROFILE_DIR = ROOT / "configs" / "profiles"
REQUIRED = {"id", "name", "models", "acceptance_rules", "severity_rules"}
SEV = {"none", "low", "medium", "high", "critical"}


def validate_profile(cfg: dict) -> None:
    missing = REQUIRED - set(cfg)
    if missing:
        raise HTTPException(422, f"Profile missing fields: {sorted(missing)}")
    acc = cfg["acceptance_rules"]
    if acc.get("fail_at_severity", "medium") not in SEV - {"none"}:
        raise HTTPException(422, "acceptance_rules.fail_at_severity must be low|medium|high|critical")
    for band in cfg["severity_rules"].get("area_levels", []):
        if band.get("level") not in SEV:
            raise HTTPException(422, f"Invalid severity level in area_levels: {band}")
        if band.get("max") is not None and not (0 <= float(band["max"]) <= 1):
            raise HTTPException(422, "area_levels max must be a ratio in [0,1]")
    cal = cfg.get("calibration", {}).get("mm_per_pixel")
    if cal is not None and float(cal) <= 0:
        raise HTTPException(422, "calibration.mm_per_pixel must be positive or null")
    m = cfg["models"]
    if not m.get("classifiers") and not m.get("anomaly"):
        raise HTTPException(422, "Profile must enable at least one model (classifiers or anomaly)")


def seed_profiles(db: Session):
    for f in sorted(PROFILE_DIR.glob("*.json")):
        cfg = json.loads(f.read_text(encoding="utf-8"))
        if db.get(ProductProfile, cfg["id"]):
            continue
        if not db.get(Product, cfg["id"]):
            db.add(Product(id=cfg["id"], name=cfg["name"], description=cfg.get("description", "")))
            db.flush()
        db.add(ProductProfile(id=cfg["id"], product_id=cfg["id"], version=cfg.get("version", 1), config=cfg))
        db.flush()  # no ORM relationship links revision->profile, so force insert order for the FK
        db.add(ProfileRevision(profile_id=cfg["id"], version=cfg.get("version", 1), config=cfg, reason="seed"))
    db.commit()


def get_profile(db: Session, profile_id: str) -> ProductProfile:
    p = db.get(ProductProfile, profile_id)
    if not p or not p.is_active:
        raise HTTPException(404, f"Product profile '{profile_id}' not found")
    return p


def update_profile(db: Session, profile_id: str, new_cfg: dict, user: User | None, reason: str) -> ProductProfile:
    p = get_profile(db, profile_id)
    cfg = copy.deepcopy(new_cfg)
    cfg["id"] = profile_id
    validate_profile(cfg)
    p.version += 1
    cfg["version"] = p.version
    p.config = cfg
    db.add(ProfileRevision(profile_id=profile_id, version=p.version, config=cfg, reason=reason,
                           user_id=user.id if user else None))
    db.commit()
    return p
