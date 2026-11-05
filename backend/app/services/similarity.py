"""Embedding-based similar-defect retrieval (cosine similarity over stored penultimate/global embeddings).

Visual similarity is NOT evidence of an identical cause; the UI states this next to results."""
from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from app.models.entities import Inspection


def similar(db: Session, insp: Inspection, k: int = 8, only_defective: bool = False) -> list[dict]:
    if insp.embedding is None:
        return []
    q = np.frombuffer(insp.embedding, np.float32)
    qn = q / (np.linalg.norm(q) + 1e-8)
    cands = db.query(Inspection.id, Inspection.embedding, Inspection.status, Inspection.defect_class,
                     Inspection.severity, Inspection.created_at, Inspection.profile_id, Inspection.human_decision) \
        .filter(Inspection.embedding_model == insp.embedding_model, Inspection.id != insp.id,
                Inspection.embedding.isnot(None))
    if only_defective:
        cands = cands.filter(Inspection.defect_present.is_(True))
    rows = cands.order_by(Inspection.created_at.desc()).limit(20000).all()
    if not rows:
        return []
    M = np.stack([np.frombuffer(r.embedding, np.float32) for r in rows])
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    s = M @ qn
    idx = np.argsort(-s)[:k]
    return [{"id": rows[i].id, "similarity": float(s[i]), "status": rows[i].status, "defect_class": rows[i].defect_class,
             "severity": rows[i].severity, "created_at": rows[i].created_at.isoformat() + "Z",
             "profile_id": rows[i].profile_id, "human_decision": rows[i].human_decision} for i in idx]
