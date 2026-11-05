"""Access requests: a self-registered user (Viewer) asks for inspection rights; an administrator approves or denies.

POST /api/access-requests                     signed-in user: request the Inspector role (one pending request at a time)
GET  /api/access-requests/mine                signed-in user: own requests, newest first
GET  /api/access-requests?status=pending      admin (manage_users): list requests
POST /api/access-requests/{id}/decision       admin (manage_users): {"action": "approve"|"deny", "note": "..."}

Approving changes the user's role to Inspector (view + inspect + review + reports + alerts). It never lowers the role
of a user who already has more rights. Every step is written to the audit trail.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.entities import AccessRequest, Role, User, utcnow
from app.security.auth import get_current_user, require
from app.services.audit import audit
from app.services.events import bus

router = APIRouter(prefix="/api/access-requests", tags=["access requests"])
GRANTED_ROLE = "inspector"


class RequestIn(BaseModel):
    reason: str = Field("", max_length=1000)


class DecisionIn(BaseModel):
    action: Literal["approve", "deny"]
    note: str = Field("", max_length=1000)


def out(r: AccessRequest) -> dict:
    return {"id": r.id, "status": r.status, "requested_role": r.requested_role, "reason": r.reason,
            "created_at": r.created_at.isoformat() + "Z",
            "decided_at": r.decided_at.isoformat() + "Z" if r.decided_at else None,
            "decided_by": r.decider.email if r.decider else None, "decision_note": r.decision_note,
            "user": {"id": r.user.id, "email": r.user.email, "username": r.user.username,
                     "full_name": r.user.full_name, "role": r.user.role.name, "role_title": r.user.role.title,
                     "created_at": r.user.created_at.isoformat() + "Z"}}


@router.post("", status_code=201)
def create(body: RequestIn, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if "inspect" in user.permission_codes:
        raise HTTPException(409, "You already have permission to run inspections")
    if db.query(AccessRequest).filter_by(user_id=user.id, status="pending").first():
        raise HTTPException(409, "You already have a pending request; an administrator will review it")
    r = AccessRequest(user_id=user.id, requested_role=GRANTED_ROLE, reason=body.reason.strip())
    db.add(r)
    db.commit()
    db.refresh(r)
    audit(db, user, "access.request", "access_request", r.id, {"role": GRANTED_ROLE}, request)
    bus.publish("access_request", {"id": r.id, "status": "pending", "user_id": user.id})
    return out(r)


@router.get("/mine")
def mine(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(AccessRequest).filter_by(user_id=user.id).order_by(AccessRequest.created_at.desc()).limit(20)
    return [out(r) for r in rows]


@router.get("")
def list_requests(status: str | None = Query(None, pattern="^(pending|approved|denied)$"),
                  _: User = Depends(require("manage_users")), db: Session = Depends(get_db)):
    q = db.query(AccessRequest)
    if status:
        q = q.filter(AccessRequest.status == status)
    rows = q.order_by(AccessRequest.status != "pending", AccessRequest.created_at.desc()).limit(500).all()
    return {"items": [out(r) for r in rows],
            "pending": db.query(AccessRequest).filter_by(status="pending").count()}


@router.post("/{rid}/decision")
def decide(rid: int, body: DecisionIn, request: Request, admin: User = Depends(require("manage_users")),
           db: Session = Depends(get_db)):
    r = db.get(AccessRequest, rid)
    if not r:
        raise HTTPException(404, "Access request not found")
    if r.status != "pending":
        raise HTTPException(409, f"Request was already {r.status}")
    r.status = "approved" if body.action == "approve" else "denied"
    r.decided_at, r.decided_by, r.decision_note = utcnow(), admin.id, body.note.strip()
    changed = None
    if r.status == "approved":
        target = db.query(Role).filter_by(name=r.requested_role).one()
        # only upgrade users who cannot inspect yet; never lower the role of someone with more rights
        if "inspect" not in r.user.permission_codes:
            changed = {"from": r.user.role.name, "to": target.name}
            r.user.role_id = target.id
    db.commit()
    db.refresh(r)
    audit(db, admin, f"access.{r.status}", "access_request", r.id,
          {"user": r.user.email, "role_change": changed, "note": r.decision_note}, request)
    bus.publish("access_request", {"id": r.id, "status": r.status, "user_id": r.user_id})
    return out(r)
