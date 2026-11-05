from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.session import get_db
from app.models.entities import ApiKey, Role, User, utcnow
from app.security.auth import (create_token, get_current_user, hash_password, limiter, new_api_key, require, revoke,
                               validate_password_strength, verify_password)
from app.services.audit import audit

EmailStr = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True, max_length=255,
                                            pattern=r"^[^@\s]+@[^@\s]+\.[A-Za-z0-9-]+$")]

Username = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=32,
                                             pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]

router = APIRouter(prefix="/api", tags=["auth & users"])
settings = get_settings()


class LoginIn(BaseModel):
    # the account's email OR its username (field kept as "email" for API compatibility)
    email: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=200)


class RegisterIn(BaseModel):
    email: EmailStr
    username: Username
    password: str = Field(min_length=1, max_length=200)
    full_name: str = Field("", max_length=255)


class UserIn(BaseModel):
    email: EmailStr
    full_name: str = ""
    password: str | None = None
    role: str = "viewer"
    is_active: bool = True


def user_out(u: User) -> dict:
    return {"id": u.id, "email": u.email, "username": u.username, "full_name": u.full_name, "role": u.role.name,
            "role_title": u.role.title,
            "permissions": sorted(u.permission_codes), "is_active": u.is_active, "preferences": u.preferences or {},
            "created_at": u.created_at.isoformat() + "Z",
            "last_login_at": u.last_login_at.isoformat() + "Z" if u.last_login_at else None}


@router.post("/auth/login")
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    limiter.check(f"login:{request.client.host if request.client else '?'}", settings.login_rate_limit_per_minute)
    ident = body.email.strip().lower()
    q = db.query(User)
    u = (q.filter(User.email == ident) if "@" in ident else q.filter(func.lower(User.username) == ident)).first()
    if not u or not verify_password(body.password, u.password_hash) or not u.is_active:
        audit(db, None, "auth.login_failed", "user", ident[:255], request=request)
        raise HTTPException(401, "Invalid email/username or password")
    token, exp = create_token(u)
    u.last_login_at = utcnow()
    audit(db, u, "auth.login", "user", u.id, request=request)
    return {"access_token": token, "token_type": "bearer", "expires_at": exp.isoformat(), "user": user_out(u)}


@router.post("/auth/register", status_code=201)
def register(body: RegisterIn, request: Request, db: Session = Depends(get_db)):
    if not settings.allow_registration:
        raise HTTPException(403, "Self-registration is disabled")
    limiter.check(f"register:{request.client.host if request.client else '?'}", 10)
    validate_password_strength(body.password)
    if db.query(User).filter(User.email == body.email.lower()).first():
        raise HTTPException(409, "Email already registered")
    if db.query(User).filter(func.lower(User.username) == body.username.lower()).first():
        raise HTTPException(409, "Username already taken")
    u = User(email=body.email.lower(), username=body.username, full_name=body.full_name or body.username,
             password_hash=hash_password(body.password),
             role_id=db.query(Role).filter_by(name="viewer").one().id)
    db.add(u)
    db.commit()
    audit(db, u, "auth.register", "user", u.id, request=request)
    return user_out(u)


@router.post("/auth/logout")
def logout(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = getattr(request.state, "token_payload", None)
    if p:
        revoke(db, p)
    audit(db, user, "auth.logout", "user", user.id, request=request)
    return {"ok": True}


@router.get("/auth/me")
def me(user: User = Depends(get_current_user)):
    return user_out(user)


@router.put("/auth/me/preferences")
def prefs(body: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    u = db.get(User, user.id)
    u.preferences = {**(u.preferences or {}), **{k: v for k, v in body.items() if k in ("theme", "language", "density",
                                                                                      "default_profile")}}
    db.commit()
    return user_out(u)


class PasswordIn(BaseModel):
    current_password: str
    new_password: str


@router.post("/auth/me/password")
def change_password(body: PasswordIn, request: Request, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    u = db.get(User, user.id)
    if not verify_password(body.current_password, u.password_hash):
        raise HTTPException(400, "Current password incorrect")
    validate_password_strength(body.new_password)
    u.password_hash = hash_password(body.new_password)
    audit(db, u, "auth.password_changed", "user", u.id, request=request)
    return {"ok": True}


# ---------------------------------------------------------------- users & roles
@router.get("/users")
def list_users(_: User = Depends(require("manage_users")), db: Session = Depends(get_db)):
    return [user_out(u) for u in db.query(User).order_by(User.id).all()]


@router.get("/roles")
def roles(_: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return [{"name": r.name, "title": r.title, "permissions": sorted(p.code for p in r.permissions)}
            for r in db.query(Role).order_by(Role.id).all()]


@router.post("/users", status_code=201)
def create_user(body: UserIn, request: Request, admin: User = Depends(require("manage_users")),
                db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email.lower()).first():
        raise HTTPException(409, "Email already exists")
    role = db.query(Role).filter_by(name=body.role).first()
    if not role:
        raise HTTPException(422, "Unknown role")
    if not body.password:
        raise HTTPException(422, "Password required")
    validate_password_strength(body.password)
    u = User(email=body.email.lower(), full_name=body.full_name, password_hash=hash_password(body.password),
             role_id=role.id, is_active=body.is_active)
    db.add(u)
    db.commit()
    audit(db, admin, "user.create", "user", u.id, {"email": u.email, "role": role.name}, request)
    return user_out(u)


@router.put("/users/{uid}")
def update_user(uid: int, body: UserIn, request: Request, admin: User = Depends(require("manage_users")),
                db: Session = Depends(get_db)):
    u = db.get(User, uid)
    if not u:
        raise HTTPException(404, "User not found")
    role = db.query(Role).filter_by(name=body.role).first()
    if not role:
        raise HTTPException(422, "Unknown role")
    if u.id == admin.id and (role.name != "admin" or not body.is_active):
        raise HTTPException(409, "You cannot demote or deactivate your own account")
    changes = {"role": [u.role.name, role.name], "active": [u.is_active, body.is_active]}
    u.full_name, u.role_id, u.is_active = body.full_name, role.id, body.is_active
    if body.password:
        validate_password_strength(body.password)
        u.password_hash = hash_password(body.password)
        changes["password_reset"] = True
    db.commit()
    db.refresh(u)
    audit(db, admin, "user.update", "user", u.id, changes, request)
    return user_out(u)


@router.delete("/users/{uid}")
def deactivate_user(uid: int, request: Request, admin: User = Depends(require("manage_users")),
                    db: Session = Depends(get_db)):
    u = db.get(User, uid)
    if not u:
        raise HTTPException(404, "User not found")
    if u.id == admin.id:
        raise HTTPException(409, "You cannot deactivate yourself")
    u.is_active = False
    db.commit()
    audit(db, admin, "user.deactivate", "user", u.id, request=request)
    return {"ok": True}


# ---------------------------------------------------------------- API keys (Mode 7: API inspection)
@router.get("/api-keys")
def list_keys(user: User = Depends(require("manage_api_keys")), db: Session = Depends(get_db)):
    q = db.query(ApiKey) if "manage_users" in user.permission_codes else db.query(ApiKey).filter_by(user_id=user.id)
    return [{"id": k.id, "name": k.name, "prefix": k.prefix, "user": k.user.email, "revoked": k.revoked,
             "created_at": k.created_at.isoformat() + "Z",
             "last_used_at": k.last_used_at.isoformat() + "Z" if k.last_used_at else None} for k in q.all()]


@router.post("/api-keys", status_code=201)
def create_key(body: dict, request: Request, user: User = Depends(require("manage_api_keys")),
               db: Session = Depends(get_db)):
    raw, prefix, h = new_api_key()
    k = ApiKey(name=str(body.get("name", "integration"))[:128], prefix=prefix, key_hash=h, user_id=user.id)
    db.add(k)
    db.commit()
    audit(db, user, "api_key.create", "api_key", k.id, {"name": k.name}, request)
    return {"id": k.id, "key": raw, "note": "Store this key now; it is not shown again. Send it as the X-API-Key header."}


@router.delete("/api-keys/{kid}")
def revoke_key(kid: int, request: Request, user: User = Depends(require("manage_api_keys")),
               db: Session = Depends(get_db)):
    k = db.get(ApiKey, kid)
    if not k or (k.user_id != user.id and "manage_users" not in user.permission_codes):
        raise HTTPException(404, "Key not found")
    k.revoked = True
    db.commit()
    audit(db, user, "api_key.revoke", "api_key", k.id, request=request)
    return {"ok": True}
