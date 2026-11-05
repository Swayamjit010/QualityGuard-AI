from __future__ import annotations

from fastapi import Request
from sqlalchemy.orm import Session

from app.models.entities import AuditEvent, User


def audit(db: Session, user: User | None, action: str, entity_type: str = "", entity_id: str = "",
          details: dict | None = None, request: Request | None = None, commit: bool = True):
    ev = AuditEvent(user_id=user.id if user else None, user_email=user.email if user else "system",
                    action=action, entity_type=entity_type, entity_id=str(entity_id), details=details or {},
                    ip=(request.client.host if request and request.client else ""))
    db.add(ev)
    if commit:
        db.commit()
    return ev
