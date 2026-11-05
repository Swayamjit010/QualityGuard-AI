"""Authentication (JWT bearer + API keys), RBAC and rate limiting."""
from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.session import get_db
from app.models.entities import ApiKey, RevokedToken, User, utcnow

settings = get_settings()
ALGO = "HS256"
bearer = HTTPBearer(auto_error=False)

PERMISSIONS = {
    "view": "View dashboards, inspections and analytics",
    "inspect": "Run inspections",
    "review": "Submit inspection feedback / reviews",
    "reports": "Generate and download reports",
    "export": "Export data (CSV/JSON/feedback datasets)",
    "manage_alerts": "Acknowledge/resolve alerts",
    "manage_profiles": "Edit product profiles and quality rules",
    "train": "Start/stop training jobs",
    "manage_models": "Register/evaluate/retire model versions",
    "approve_models": "Approve and deploy model versions, rollback",
    "manage_datasets": "Import/split datasets",
    "manage_users": "Manage users and roles",
    "manage_settings": "Change system settings",
    "view_logs": "View system logs and health",
    "manage_api_keys": "Create API keys for external integrations",
}

ROLES = {
    "admin": ("Administrator", list(PERMISSIONS)),
    "quality_manager": ("Quality Manager", ["view", "inspect", "review", "reports", "export", "manage_alerts",
                                            "manage_profiles", "approve_models", "view_logs"]),
    "inspector": ("Inspector", ["view", "inspect", "review", "reports", "manage_alerts"]),
    "ml_engineer": ("ML Engineer", ["view", "inspect", "reports", "export", "train", "manage_models",
                                    "manage_datasets", "view_logs", "manage_api_keys"]),
    "viewer": ("Viewer", ["view"]),
}


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode()[:72], bcrypt.gensalt(12)).decode()


def verify_password(pw: str, h: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode()[:72], h.encode())
    except ValueError:
        return False


def validate_password_strength(pw: str) -> None:
    if len(pw) < 10 or pw.lower() == pw or not any(c.isdigit() for c in pw):
        raise HTTPException(422, "Password must be >= 10 characters and include upper-case, lower-case and a digit.")


def create_token(user: User) -> tuple[str, datetime]:
    exp = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_minutes)
    payload = {"sub": str(user.id), "role": user.role.name, "exp": exp, "iat": datetime.now(timezone.utc),
               "jti": secrets.token_hex(16)}
    return jwt.encode(payload, settings.secret_key, algorithm=ALGO), exp


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[ALGO])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Session expired")
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid token")


def revoke(db: Session, payload: dict):
    db.merge(RevokedToken(jti=payload["jti"], expires_at=datetime.fromtimestamp(payload["exp"], timezone.utc)
                          .replace(tzinfo=None)))
    db.commit()


def new_api_key() -> tuple[str, str, str]:
    raw = "qg_" + secrets.token_urlsafe(32)
    return raw, raw[:10], hash_password(raw)


def _user_from_api_key(db: Session, raw: str) -> User | None:
    for k in db.query(ApiKey).filter(ApiKey.prefix == raw[:10], ApiKey.revoked.is_(False)).all():
        if verify_password(raw, k.key_hash):
            k.last_used_at = utcnow()
            db.commit()
            return k.user
    return None


def get_current_user(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(bearer),
                     db: Session = Depends(get_db)) -> User:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        user = _user_from_api_key(db, api_key)
        if not user or not user.is_active:
            raise HTTPException(401, "Invalid API key")
        request.state.auth = "api_key"
        request.state.user = user
        return user
    token = creds.credentials if creds else request.query_params.get("token")  # token param for <img>/downloads
    if not token:
        raise HTTPException(401, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    payload = decode_token(token)
    if db.get(RevokedToken, payload.get("jti", "")):
        raise HTTPException(401, "Session has been logged out")
    user = db.get(User, int(payload["sub"]))
    if not user or not user.is_active:
        raise HTTPException(401, "User inactive or not found")
    request.state.auth = "jwt"
    request.state.token_payload = payload
    request.state.user = user
    return user


def require(*perms: str):
    def dep(user: User = Depends(get_current_user)) -> User:
        missing = [p for p in perms if p not in user.permission_codes]
        if missing:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Missing permission: {', '.join(missing)}")
        return user
    return dep


class RateLimiter:
    """Sliding-window in-memory limiter (per process). Use a shared store (Redis) when scaling out."""

    def __init__(self):
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str, limit: int, window: float = 60.0):
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                raise HTTPException(429, "Rate limit exceeded; retry later", headers={"Retry-After": str(int(window))})
            q.append(now)


limiter = RateLimiter()
