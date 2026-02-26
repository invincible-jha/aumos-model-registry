"""Protocol (interface) definitions for the Model Registry service.

Defines abstract contracts between the service layer and adapters,
enabling dependency injection and test doubles without coupling to
concrete implementations.
"""

import uuid
from decimal import Decimal
from typing import Protocol, runtime_checkable

from aumos_model_registry.core.models import (
    Experiment,
    ExperimentRun,
    Model,
    ModelDeployment,
    ModelVersion,
)


@runtime_checkable
class IModelRepository(Protocol):
    """Contract for model persistence operations."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        name: str,
        created_by: uuid.UUID,
        description: str | None,
        model_type: str | None,
        framework: str | None,
        tags: dict,
    ) -> Model:
        """Create a new model record."""
        ...

    async def get_by_id(self, model_id: uuid.UUID, tenant_id: uuid.UUID) -> Model | None:
        """Fetch a model by its UUID within a tenant scope."""
        ...

    async def get_by_name(self, name: str, tenant_id: uuid.UUID) -> Model | None:
        """Fetch a model by tenant-scoped name."""
        ...

    async def list_all(
        self,
        tenant_id: uuid.UUID,
        page: int,
        page_size: int,
        model_type: str | None,
        framework: str | None,
    ) -> tuple[list[Model], int]:
        """Return paginated models with optional filters. Returns (items, total)."""
        ...

    async def search(
        self, tenant_id: uuid.UUID, query: str, limit: int
    ) -> list[Model]:
        """Full-text / semantic search over model name, description, and tags."""
        ...

    async def update_tags(
        self, model_id: uuid.UUID, tenant_id: uuid.UUID, tags: dict
    ) -> Model:
        """Replace the model's tags JSONB with the provided dict."""
        ...

    async def delete(self, model_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
        """Soft-delete (or hard-delete) a model and cascade to versions."""
        ...


@runtime_checkable
class IVersionRepository(Protocol):
    """Contract for model version persistence operations."""

    async def create(
        self,
        model_id: uuid.UUID,
        artifact_uri: str | None,
        training_data: dict | None,
        hyperparameters: dict | None,
        metrics: dict | None,
        parent_model_id: uuid.UUID | None,
        training_cost: Decimal | None,
        size_bytes: int | None,
    ) -> ModelVersion:
        """Create a new version and auto-increment version number."""
        ...

    async def get_by_id(self, version_id: uuid.UUID) -> ModelVersion | None:
        """Fetch a version by its UUID."""
        ...

    async def get_by_model_and_number(
        self, model_id: uuid.UUID, version: int
    ) -> ModelVersion | None:
        """Fetch a specific version number for a model."""
        ...

    async def list_by_model(self, model_id: uuid.UUID) -> list[ModelVersion]:
        """Return all versions for a model, newest first."""
        ...

    async def transition_stage(
        self, version_id: uuid.UUID, new_stage: str
    ) -> ModelVersion:
        """Update the lifecycle stage of a version."""
        ...

    async def set_ml_bom(
        self, version_id: uuid.UUID, ml_bom: dict
    ) -> ModelVersion:
        """Attach a CycloneDX ML-BOM JSON payload to a version."""
        ...


@runtime_checkable
class IDeploymentRepository(Protocol):
    """Contract for deployment tracking operations."""

    async def create(
        self,
        model_version_id: uuid.UUID,
        tenant_id: uuid.UUID,
        environment: str,
        endpoint_url: str | None,
    ) -> ModelDeployment:
        """Record a new deployment."""
        ...

    async def get_by_id(self, deployment_id: uuid.UUID) -> ModelDeployment | None:
        """Fetch a deployment by UUID."""
        ...

    async def list_by_version(self, model_version_id: uuid.UUID) -> list[ModelDeployment]:
        """Return all deployments for a version."""
        ...

    async def list_by_tenant(
        self, tenant_id: uuid.UUID, environment: str | None
    ) -> list[ModelDeployment]:
        """Return all deployments for a tenant, optionally filtered by environment."""
        ...

    async def update_status(
        self, deployment_id: uuid.UUID, status: str
    ) -> ModelDeployment:
        """Update deployment status."""
        ...

    async def increment_inference_count(
        self,
        deployment_id: uuid.UUID,
        count: int,
        cost_delta: Decimal,
    ) -> None:
        """Atomically increment inference counter and accumulated cost."""
        ...


@runtime_checkable
class IExperimentRepository(Protocol):
    """Contract for experiment and run persistence."""

    async def create_experiment(
        self,
        tenant_id: uuid.UUID,
        name: str,
        description: str | None,
    ) -> Experiment:
        """Create a new experiment."""
        ...

    async def get_experiment(
        self, experiment_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Experiment | None:
        """Fetch an experiment."""
        ...

    async def list_experiments(
        self, tenant_id: uuid.UUID
    ) -> list[Experiment]:
        """Return all experiments for a tenant."""
        ...

    async def create_run(
        self,
        experiment_id: uuid.UUID,
        tenant_id: uuid.UUID,
        parameters: dict,
    ) -> ExperimentRun:
        """Start a new experiment run."""
        ...

    async def get_run(self, run_id: uuid.UUID) -> ExperimentRun | None:
        """Fetch a run by UUID."""
        ...

    async def list_runs(self, experiment_id: uuid.UUID) -> list[ExperimentRun]:
        """Return all runs for an experiment."""
        ...

    async def update_run(
        self,
        run_id: uuid.UUID,
        metrics: dict | None,
        artifacts: list | None,
        status: str | None,
    ) -> ExperimentRun:
        """Update metrics, artifacts, and/or status on a run."""
        ...


@runtime_checkable
class IArtifactStorage(Protocol):
    """Contract for binary artifact storage (MinIO / S3-compatible)."""

    async def upload_artifact(
        self,
        tenant_id: uuid.UUID,
        model_id: uuid.UUID,
        version: int,
        object_name: str,
        data: bytes,
        content_type: str,
    ) -> str:
        """Upload an artifact and return its canonical URI."""
        ...

    async def get_presigned_download_url(
        self, artifact_uri: str, expiry_seconds: int = 3600
    ) -> str:
        """Return a time-limited presigned URL for artifact download."""
        ...

    async def delete_artifacts(
        self, tenant_id: uuid.UUID, model_id: uuid.UUID, version: int
    ) -> None:
        """Delete all artifacts for a model version."""
        ...
