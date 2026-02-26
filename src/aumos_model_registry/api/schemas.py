"""Pydantic request and response schemas for the AumOS Model Registry API.

All API inputs and outputs are represented as Pydantic v2 models.
Never return raw dicts from endpoints — always use these schemas.
"""

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field, HttpUrl


# ---------------------------------------------------------------------------
# Shared / common
# ---------------------------------------------------------------------------

class PaginationMeta(BaseModel):
    """Pagination metadata embedded in list responses."""

    page: int
    page_size: int
    total: int
    total_pages: int


# ---------------------------------------------------------------------------
# Model schemas
# ---------------------------------------------------------------------------

class ModelCreateRequest(BaseModel):
    """Request body for registering a new model."""

    name: str = Field(..., min_length=1, max_length=255, description="Unique model name within the tenant")
    description: str | None = Field(None, description="Long-form description")
    model_type: str | None = Field(
        None,
        description="Broad category: classification | regression | llm | embedding | other",
    )
    framework: str | None = Field(
        None,
        description="Training framework: pytorch | tensorflow | sklearn | transformers | jax | other",
    )
    tags: dict[str, str] = Field(default_factory=dict, description="Arbitrary key/value metadata")


class ModelResponse(BaseModel):
    """Response schema for a single model."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    description: str | None
    model_type: str | None
    framework: str | None
    created_by: uuid.UUID
    tags: dict
    created_at: datetime
    updated_at: datetime
    version_count: int = Field(0, description="Number of registered versions")

    model_config = {"from_attributes": True}


class ModelListResponse(BaseModel):
    """Paginated list of models."""

    items: list[ModelResponse]
    pagination: PaginationMeta


class ModelSearchResponse(BaseModel):
    """Search results for models."""

    items: list[ModelResponse]
    query: str
    total: int


# ---------------------------------------------------------------------------
# Version schemas
# ---------------------------------------------------------------------------

class TrainingDataSchema(BaseModel):
    """Training data provenance descriptor."""

    datasets: list[dict] = Field(
        default_factory=list,
        description="List of dataset descriptors with name, version, source, records, license",
    )
    split: dict[str, float] | None = Field(
        None,
        description="Train/val/test split ratios",
    )


class VersionCreateRequest(BaseModel):
    """Request body for creating a new model version."""

    artifact_uri: str | None = Field(
        None,
        description="S3/MinIO URI where model artifact is stored",
    )
    training_data: TrainingDataSchema | None = Field(
        None,
        description="Training data provenance",
    )
    hyperparameters: dict[str, float | int | str | bool] | None = Field(
        None,
        description="Training hyperparameters",
    )
    metrics: dict[str, float | int] | None = Field(
        None,
        description="Evaluation metrics from training",
    )
    parent_model_id: uuid.UUID | None = Field(
        None,
        description="Base model UUID if this version was fine-tuned",
    )
    training_cost_usd: Decimal | None = Field(
        None,
        ge=Decimal("0"),
        description="Total training compute cost in USD",
    )
    size_bytes: int | None = Field(
        None,
        ge=0,
        description="Artifact size in bytes",
    )
    generate_bom: bool = Field(
        True,
        description="Auto-generate CycloneDX ML-BOM on creation",
    )


class VersionResponse(BaseModel):
    """Response schema for a single model version."""

    id: uuid.UUID
    model_id: uuid.UUID
    version: int
    stage: str
    artifact_uri: str | None
    training_data: dict | None
    hyperparameters: dict | None
    metrics: dict | None
    parent_model_id: uuid.UUID | None
    training_cost: Decimal | None
    size_bytes: int | None
    ml_bom: dict | None
    created_at: datetime

    model_config = {"from_attributes": True}


class StageTransitionRequest(BaseModel):
    """Request to transition a model version to a new stage."""

    stage: str = Field(
        ...,
        description="Target stage: staging | production | archived",
        pattern="^(staging|production|archived|development)$",
    )
    reason: str | None = Field(
        None,
        description="Human-readable reason for the transition (audit trail)",
    )


# ---------------------------------------------------------------------------
# Deployment schemas
# ---------------------------------------------------------------------------

class DeploymentCreateRequest(BaseModel):
    """Request body for deploying a model version."""

    environment: str = Field(
        ...,
        description="Target environment: dev | staging | production",
        pattern="^(dev|staging|production)$",
    )
    endpoint_url: str | None = Field(
        None,
        description="Inference endpoint URL (may be populated post-deploy by orchestrator)",
    )


class DeploymentResponse(BaseModel):
    """Response schema for a single deployment."""

    id: uuid.UUID
    model_version_id: uuid.UUID
    tenant_id: uuid.UUID
    environment: str | None
    endpoint_url: str | None
    status: str | None
    inference_count: int
    inference_cost: Decimal
    deployed_at: datetime
    last_inference: datetime | None

    model_config = {"from_attributes": True}


class RollbackRequest(BaseModel):
    """Request to roll back a deployment."""

    reason: str | None = Field(
        None,
        description="Reason for rollback (stored in audit log)",
    )


# ---------------------------------------------------------------------------
# Lineage schemas
# ---------------------------------------------------------------------------

class LineageEdge(BaseModel):
    """A directed edge in the model lineage graph."""

    from_model_id: str
    to_model_id: str
    to_version_id: str
    relationship: str


class LineageVersionNode(BaseModel):
    """A model version node in the lineage graph."""

    version_id: str
    version: int
    stage: str
    parent_model_id: str | None


class LineageResponse(BaseModel):
    """Lineage graph for a model."""

    model_id: str
    versions: list[LineageVersionNode]
    edges: list[LineageEdge]


# ---------------------------------------------------------------------------
# Cost schemas
# ---------------------------------------------------------------------------

class CostBreakdownResponse(BaseModel):
    """Cost attribution breakdown for a model version."""

    model_version_id: str
    training_cost_usd: Decimal
    inference_cost_usd: Decimal
    storage_cost_monthly_usd: Decimal
    total_cost_usd: Decimal
    currency: str = "USD"


# ---------------------------------------------------------------------------
# Experiment schemas
# ---------------------------------------------------------------------------

class ExperimentCreateRequest(BaseModel):
    """Request body for creating a new experiment."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None


class ExperimentResponse(BaseModel):
    """Response schema for an experiment."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class RunCreateRequest(BaseModel):
    """Request body for starting an experiment run."""

    parameters: dict[str, float | int | str | bool] = Field(
        default_factory=dict,
        description="Initial hyperparameter values",
    )


class RunUpdateRequest(BaseModel):
    """Request body for logging metrics/artifacts to a run."""

    metrics: dict[str, float | int] | None = Field(
        None,
        description="Metric name → value pairs to log",
    )
    artifacts: list[dict] | None = Field(
        None,
        description="List of artifact descriptors {name, uri, type}",
    )
    status: str | None = Field(
        None,
        description="Terminal status: finished | failed | killed",
        pattern="^(finished|failed|killed)$",
    )


class RunResponse(BaseModel):
    """Response schema for an experiment run."""

    id: uuid.UUID
    experiment_id: uuid.UUID
    tenant_id: uuid.UUID
    status: str
    parameters: dict
    metrics: dict
    artifacts: list
    started_at: datetime
    ended_at: datetime | None

    model_config = {"from_attributes": True}
