"""SQLAlchemy ORM models for the AumOS Model Registry.

All tables use the `reg_` prefix. Tenant-scoped tables extend AumOSModel
which supplies id (UUID), tenant_id, created_at, and updated_at columns.
"""

import uuid
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aumos_common.database import AumOSModel, Base


class Model(AumOSModel):
    """Registered AI/ML model entity.

    Represents a named model owned by a tenant. Versions are attached
    separately via the ModelVersion relationship.

    Table: reg_models
    """

    __tablename__ = "reg_models"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_reg_models_tenant_name"),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="e.g. classification, regression, llm, embedding",
    )
    framework: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="e.g. pytorch, tensorflow, sklearn, transformers",
    )
    created_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    tags: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    versions: Mapped[list["ModelVersion"]] = relationship(
        "ModelVersion",
        back_populates="model",
        cascade="all, delete-orphan",
        order_by="desc(ModelVersion.version)",
    )


class ModelVersion(Base):
    """A specific version of a registered model.

    Captures training provenance, hyperparameters, evaluation metrics,
    artifact storage location, ML-BOM, cost data, and lineage pointer.

    Table: reg_model_versions
    """

    __tablename__ = "reg_model_versions"
    __table_args__ = (
        UniqueConstraint("model_id", "version", name="uq_reg_versions_model_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reg_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    stage: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="development",
        comment="development | staging | production | archived",
    )
    artifact_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    training_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    hyperparameters: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    parent_model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="UUID of the base model this version was fine-tuned from",
    )
    ml_bom: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="CycloneDX-format ML Bill of Materials",
    )
    training_cost: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    from sqlalchemy import func
    from sqlalchemy import DateTime

    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    model: Mapped["Model"] = relationship("Model", back_populates="versions")
    deployments: Mapped[list["ModelDeployment"]] = relationship(
        "ModelDeployment",
        back_populates="model_version",
        cascade="all, delete-orphan",
    )


class ModelDeployment(Base):
    """An active or historical deployment of a model version.

    Tracks the environment, endpoint, status, and running inference costs
    for a particular model version deployment.

    Table: reg_model_deployments
    """

    __tablename__ = "reg_model_deployments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reg_model_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    environment: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="dev | staging | production",
    )
    endpoint_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        comment="pending | active | inactive | failed",
    )
    inference_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    inference_cost: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False, default=Decimal("0.00")
    )

    from sqlalchemy import func
    from sqlalchemy import DateTime

    deployed_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_inference: Mapped[str | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    model_version: Mapped["ModelVersion"] = relationship(
        "ModelVersion", back_populates="deployments"
    )


class Experiment(AumOSModel):
    """An MLflow-compatible experiment grouping multiple runs.

    Table: reg_experiments
    """

    __tablename__ = "reg_experiments"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "name", name="uq_reg_experiments_tenant_name"
        ),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    runs: Mapped[list["ExperimentRun"]] = relationship(
        "ExperimentRun",
        back_populates="experiment",
        cascade="all, delete-orphan",
        order_by="desc(ExperimentRun.started_at)",
    )


class ExperimentRun(Base):
    """A single training run within an experiment.

    Captures parameters, metrics, artifact references, and timing
    for one execution of a training job.

    Table: reg_experiment_runs
    """

    __tablename__ = "reg_experiment_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reg_experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="running",
        comment="running | finished | failed | killed",
    )
    parameters: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    artifacts: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    from sqlalchemy import func
    from sqlalchemy import DateTime

    started_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)

    experiment: Mapped["Experiment"] = relationship(
        "Experiment", back_populates="runs"
    )


# ---------------------------------------------------------------------------
# Model Provenance Chain (P1.2)
# ---------------------------------------------------------------------------


class ModelProvenanceChain(AumOSModel):
    """Cryptographic chain of custody for an AI model asset.

    Records the complete provenance lifecycle from training dataset through
    deployment and inference. Each link in the chain is signed with Ed25519
    and SHA-256 hash-linked to the previous link, making the chain tamper-evident.

    Table: reg_model_provenance_chains
    """

    __tablename__ = "reg_model_provenance_chains"

    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="ID of the model whose provenance this chain tracks",
    )
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Specific model version if applicable",
    )
    chain_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        comment="active | sealed | revoked",
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether the full chain has been cryptographically verified",
    )
    head_link_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="SHA-256 hash of the most recently appended link",
    )
    metadata_: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        name="metadata",
        comment="Arbitrary context metadata for the chain",
    )

    links: Mapped[list["ProvenanceChainLink"]] = relationship(
        "ProvenanceChainLink",
        back_populates="chain",
        cascade="all, delete-orphan",
        order_by="ProvenanceChainLink.sequence_number",
    )


class ProvenanceChainLink(AumOSModel):
    """A single tamper-evident link in a model provenance chain.

    Each link records one stage in the model lifecycle (e.g., training dataset
    ingestion, training run, validation). Links are hash-chained: the
    ``previous_link_hash`` of each link equals the ``link_hash`` of its
    predecessor, forming an immutable chain. The ``signature`` field holds
    an Ed25519 signature over the canonical JSON of the link payload.

    Table: reg_provenance_chain_links
    """

    __tablename__ = "reg_provenance_chain_links"

    chain_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reg_model_provenance_chains.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="1-based position in the chain; must be monotonically increasing",
    )
    link_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment=(
            "training_dataset | training_run | model_artifact | "
            "validation_results | approval_decision | deployment_record | inference_log"
        ),
    )
    payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        comment="Structured data describing this lifecycle stage",
    )
    link_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="SHA-256 hash of (previous_link_hash + canonical JSON payload)",
    )
    previous_link_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="Hash of the predecessor link; NULL for the genesis link",
    )
    signature: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Base64-encoded Ed25519 signature over the canonical link JSON",
    )
    signed_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        comment="UUID of the public key record (reg_provenance_public_keys) used to sign",
    )
    actor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User/service that created this link",
    )

    chain: Mapped["ModelProvenanceChain"] = relationship(
        "ModelProvenanceChain", back_populates="links"
    )


class ProvenancePublicKey(AumOSModel):
    """Registered Ed25519 public key used to verify provenance chain signatures.

    Tenants register one or more Ed25519 public keys. Each key is identified
    by a UUID. Chain link signatures reference ``ProvenancePublicKey.id``.

    Table: reg_provenance_public_keys
    """

    __tablename__ = "reg_provenance_public_keys"

    key_label: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable label for the key (e.g., 'mlops-pipeline-prod')",
    )
    public_key_pem: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="PEM-encoded Ed25519 public key",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether this key is currently trusted for signing",
    )
    revoked_at: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="ISO-8601 timestamp of revocation, if applicable",
    )
