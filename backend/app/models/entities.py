"""Normalised relational schema (SQLite for dev, PostgreSQL-compatible types only)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, LargeBinary, String,
                        Table, Text, UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def uid() -> str:
    return uuid.uuid4().hex


role_permissions = Table(
    "role_permissions", Base.metadata,
    Column("role_id", ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
    Column("permission_id", ForeignKey("permissions.id", ondelete="CASCADE"), primary_key=True),
)


class Permission(Base):
    __tablename__ = "permissions"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(64), unique=True)
    description: Mapped[str] = mapped_column(String(255), default="")


class Role(Base):
    __tablename__ = "roles"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    title: Mapped[str] = mapped_column(String(128), default="")
    permissions: Mapped[list[Permission]] = relationship(secondary=role_permissions, lazy="selectin")


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    # chosen at self-registration; nullable because accounts created earlier (or by an admin) may not have one
    username: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, nullable=True)
    full_name: Mapped[str] = mapped_column(String(255), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id"))
    role: Mapped[Role] = relationship(lazy="selectin")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    preferences: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    @property
    def permission_codes(self) -> set[str]:
        return {p.code for p in self.role.permissions}


class AccessRequest(Base):
    """A self-registered user asking an administrator for more permissions (e.g. the Inspector role)."""
    __tablename__ = "access_requests"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    user: Mapped[User] = relationship(foreign_keys=[user_id], lazy="selectin")
    requested_role: Mapped[str] = mapped_column(String(32), default="inspector")
    reason: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)  # pending|approved|denied
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    decided_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    decider: Mapped[User | None] = relationship(foreign_keys=[decided_by], lazy="selectin")
    decision_note: Mapped[str] = mapped_column(Text, default="")


class RevokedToken(Base):
    __tablename__ = "revoked_tokens"
    jti: Mapped[str] = mapped_column(String(64), primary_key=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    prefix: Mapped[str] = mapped_column(String(16), index=True)
    key_hash: Mapped[str] = mapped_column(String(255))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    user: Mapped[User] = relationship(lazy="selectin")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


# ------------------------------------------------------------------ products / production
class Product(Base):
    __tablename__ = "products"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ProductProfile(Base):
    __tablename__ = "product_profiles"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id"))
    version: Mapped[int] = mapped_column(Integer, default=1)
    config: Mapped[dict] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    product: Mapped[Product] = relationship(lazy="selectin")


class ProfileRevision(Base):
    """Immutable history of profile configurations (supports model rollback + audit)."""
    __tablename__ = "profile_revisions"
    id: Mapped[int] = mapped_column(primary_key=True)
    profile_id: Mapped[str] = mapped_column(ForeignKey("product_profiles.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    config: Mapped[dict] = mapped_column(JSON)
    reason: Mapped[str] = mapped_column(String(255), default="")
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ProductionLine(Base):
    __tablename__ = "production_lines"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    location: Mapped[str] = mapped_column(String(255), default="")
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False)


class ProductionBatch(Base):
    __tablename__ = "production_batches"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(255))
    profile_id: Mapped[str | None] = mapped_column(ForeignKey("product_profiles.id"), nullable=True)
    line_id: Mapped[str | None] = mapped_column(ForeignKey("production_lines.id"), nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="batch")
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False)


# ------------------------------------------------------------------ datasets
class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    title: Mapped[str] = mapped_column(String(255), default="")
    task: Mapped[str] = mapped_column(String(64), default="classification")
    layout: Mapped[str] = mapped_column(String(64), default="")
    source: Mapped[str] = mapped_column(Text, default="")
    license: Mapped[str] = mapped_column(String(255), default="")
    path: Mapped[str] = mapped_column(String(512))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    versions: Mapped[list["DatasetVersion"]] = relationship(back_populates="dataset", lazy="selectin",
                                                            order_by="DatasetVersion.created_at")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    dataset: Mapped[Dataset] = relationship(back_populates="versions")
    version: Mapped[str] = mapped_column(String(32))
    manifest_path: Mapped[str] = mapped_column(String(512))
    num_samples: Mapped[int] = mapped_column(Integer)
    split_counts: Mapped[dict] = mapped_column(JSON, default=dict)
    statistics: Mapped[dict] = mapped_column(JSON, default=dict)
    split_policy: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    __table_args__ = (UniqueConstraint("dataset_id", "version"),)


class ImageAsset(Base):
    __tablename__ = "images"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    path: Mapped[str] = mapped_column(String(512))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    original_filename: Mapped[str] = mapped_column(String(255), default="")
    dataset_version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ImageAnnotation(Base):
    __tablename__ = "image_annotations"
    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32))  # label | bbox | mask | polygon
    label: Mapped[str] = mapped_column(String(128), default="")
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    mask_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="dataset")  # dataset | inspector
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ------------------------------------------------------------------ models
class MLModel(Base):
    __tablename__ = "ml_models"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)   # e.g. neu_cls-efficientnet_b0
    task: Mapped[str] = mapped_column(String(64))                 # classification | anomaly_detection
    architecture: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text, default="")
    versions: Mapped[list["ModelVersion"]] = relationship(back_populates="model", lazy="selectin",
                                                          order_by="ModelVersion.created_at")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("ml_models.id", ondelete="CASCADE"))
    model: Mapped[MLModel] = relationship(back_populates="versions", lazy="selectin")
    run_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    version: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="evaluation")
    # development|training|evaluation|validated|approved|deployed|retired|failed
    dataset_name: Mapped[str] = mapped_column(String(128), default="")
    dataset_version: Mapped[str] = mapped_column(String(32), default="")
    input_size: Mapped[int] = mapped_column(Integer, default=224)
    class_labels: Mapped[list] = mapped_column(JSON, default=list)
    checkpoint_path: Mapped[str] = mapped_column(String(512))
    checkpoint_size_mb: Mapped[float] = mapped_column(Float, default=0)
    preprocess: Mapped[dict] = mapped_column(JSON, default=dict)
    training_config: Mapped[dict] = mapped_column(JSON, default=dict)
    approved_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    metrics: Mapped[list["EvaluationMetric"]] = relationship(back_populates="model_version", lazy="selectin")


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"), index=True)
    model_version: Mapped[ModelVersion] = relationship(back_populates="metrics")
    split: Mapped[str] = mapped_column(String(32), default="test")
    name: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    kind: Mapped[str] = mapped_column(String(32), default="classification")
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="queued")  # queued|running|completed|failed|stopped
    message: Mapped[str] = mapped_column(Text, default="")
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    log_path: Mapped[str] = mapped_column(String(512), default="")
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ------------------------------------------------------------------ inspections
class Inspection(Base):
    __tablename__ = "inspections"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    profile_id: Mapped[str] = mapped_column(ForeignKey("product_profiles.id"), index=True)
    profile_version: Mapped[int] = mapped_column(Integer, default=1)
    batch_id: Mapped[str | None] = mapped_column(ForeignKey("production_batches.id", ondelete="SET NULL"),
                                                 nullable=True, index=True)
    line_id: Mapped[str | None] = mapped_column(ForeignKey("production_lines.id"), nullable=True)
    image_id: Mapped[str | None] = mapped_column(ForeignKey("images.id"), nullable=True)
    product_serial: Mapped[str] = mapped_column(String(128), default="")
    source: Mapped[str] = mapped_column(String(32), default="single")
    filename: Mapped[str] = mapped_column(String(255), default="")
    artifact_dir: Mapped[str] = mapped_column(String(512), default="")
    status: Mapped[str] = mapped_column(String(32), index=True)
    defect_present: Mapped[bool] = mapped_column(Boolean, default=False)
    defect_class: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    severity: Mapped[str] = mapped_column(String(16), default="none", index=True)
    quality_index: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    anomaly_score_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    reliability: Mapped[str] = mapped_column(String(16), default="")
    unfamiliar: Mapped[bool] = mapped_column(Boolean, default=False)
    manual_review_required: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    review_status: Mapped[str] = mapped_column(String(16), default="not_required", index=True)
    human_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0)
    models_used: Mapped[list] = mapped_column(JSON, default=list)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    ground_truth: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    video_timestamp_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_brightness: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_contrast: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_sharpness: Mapped[float | None] = mapped_column(Float, nullable=True)

    predictions: Mapped[list["Prediction"]] = relationship(cascade="all, delete-orphan", lazy="selectin")
    anomaly: Mapped["AnomalyResult | None"] = relationship(cascade="all, delete-orphan", lazy="selectin",
                                                           uselist=False)
    segmentation: Mapped["SegmentationResult | None"] = relationship(cascade="all, delete-orphan", lazy="selectin",
                                                                     uselist=False)
    severity_assessment: Mapped["SeverityAssessment | None"] = relationship(cascade="all, delete-orphan",
                                                                            lazy="selectin", uselist=False)
    feedback: Mapped[list["HumanFeedback"]] = relationship(cascade="all, delete-orphan", lazy="selectin",
                                                           order_by="HumanFeedback.created_at")
    __table_args__ = (Index("ix_insp_profile_created", "profile_id", "created_at"),)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.id", ondelete="CASCADE"), index=True)
    model_run_id: Mapped[str] = mapped_column(String(128))
    predicted_class: Mapped[str] = mapped_column(String(128))
    confidence: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict] = mapped_column(JSON, default=dict)
    ood_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_ood: Mapped[bool] = mapped_column(Boolean, default=False)


class AnomalyResult(Base):
    __tablename__ = "anomaly_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.id", ondelete="CASCADE"), unique=True)
    model_run_id: Mapped[str] = mapped_column(String(128))
    scores: Mapped[dict] = mapped_column(JSON)
    thresholds: Mapped[dict] = mapped_column(JSON)
    is_anomalous: Mapped[bool] = mapped_column(Boolean)
    engines_disagree: Mapped[bool] = mapped_column(Boolean)
    novelty_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    novelty_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)


class SegmentationResult(Base):
    __tablename__ = "segmentation_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.id", ondelete="CASCADE"), unique=True)
    method: Mapped[str] = mapped_column(String(64))
    num_regions: Mapped[int] = mapped_column(Integer)
    total_area_px: Mapped[float] = mapped_column(Float)
    area_ratio: Mapped[float] = mapped_column(Float)
    calibrated: Mapped[bool] = mapped_column(Boolean, default=False)
    measurements: Mapped[dict] = mapped_column(JSON)


class SeverityAssessment(Base):
    __tablename__ = "severity_assessments"
    id: Mapped[int] = mapped_column(primary_key=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.id", ondelete="CASCADE"), unique=True)
    ai_level: Mapped[str] = mapped_column(String(16))
    basis: Mapped[list] = mapped_column(JSON, default=list)
    rule_status: Mapped[str] = mapped_column(String(32))
    rules_applied: Mapped[list] = mapped_column(JSON, default=list)
    human_level: Mapped[str | None] = mapped_column(String(16), nullable=True)


class HumanFeedback(Base):
    __tablename__ = "human_feedback"
    id: Mapped[int] = mapped_column(primary_key=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.id", ondelete="CASCADE"), index=True)
    reviewer_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    reviewer: Mapped[User] = relationship(lazy="selectin")
    verdict: Mapped[str] = mapped_column(String(16))           # confirm | reject
    corrected_class: Mapped[str | None] = mapped_column(String(128), nullable=True)
    corrected_severity: Mapped[str | None] = mapped_column(String(16), nullable=True)
    final_decision: Mapped[str] = mapped_column(String(16))    # accept | reject
    location_confirmed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    difficult: Mapped[bool] = mapped_column(Boolean, default=False)
    comment: Mapped[str] = mapped_column(Text, default="")
    annotation_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    exported: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


# ------------------------------------------------------------------ operations
class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    # critical_defect|repeated_defect|unfamiliar_pattern|model_drift|inspection_failure|review_required
    level: Mapped[str] = mapped_column(String(16), default="warning")  # info|warning|critical
    title: Mapped[str] = mapped_column(String(255))
    message: Mapped[str] = mapped_column(Text, default="")
    profile_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    inspection_id: Mapped[str | None] = mapped_column(ForeignKey("inspections.id", ondelete="SET NULL"),
                                                      nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)  # open|acknowledged|resolved
    acknowledged_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    resolved_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    data: Mapped[dict] = mapped_column(JSON, default=dict)


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(255))
    format: Mapped[str] = mapped_column(String(8), default="pdf")
    path: Mapped[str] = mapped_column(String(512), default="")
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="ready")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    inspection_id: Mapped[str | None] = mapped_column(ForeignKey("inspections.id", ondelete="SET NULL"),
                                                      nullable=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    user_email: Mapped[str] = mapped_column(String(255), default="")
    action: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(64), default="")
    entity_id: Mapped[str] = mapped_column(String(128), default="")
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    ip: Mapped[str] = mapped_column(String(64), default="")


class SystemSetting(Base):
    __tablename__ = "system_settings"
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Job(Base):
    """Generic background job (batch, video, report, simulation, drift check)."""
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0)
    message: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class KnowledgeEntry(Base):
    __tablename__ = "knowledge_base"
    id: Mapped[int] = mapped_column(primary_key=True)
    defect_class: Mapped[str] = mapped_column(String(128), index=True)
    profile_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    title: Mapped[str] = mapped_column(String(255))
    visual_description: Mapped[str] = mapped_column(Text, default="")
    characteristics: Mapped[list] = mapped_column(JSON, default=list)
    inspection_guidance: Mapped[list] = mapped_column(JSON, default=list)
    corrective_actions: Mapped[list] = mapped_column(JSON, default=list)  # [{text, verified, verified_by}]
    source: Mapped[str] = mapped_column(Text, default="")
    example_images: Mapped[list] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class DriftCheck(Base):
    __tablename__ = "drift_checks"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    profile_id: Mapped[str] = mapped_column(String(64), index=True)
    window: Mapped[int] = mapped_column(Integer)
    data_drift: Mapped[bool] = mapped_column(Boolean)
    prediction_drift: Mapped[bool] = mapped_column(Boolean)
    performance_degradation: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    details: Mapped[dict] = mapped_column(JSON)
