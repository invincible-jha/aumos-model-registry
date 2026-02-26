"""SQLAlchemy repository implementations for the AumOS Model Registry.

Implements the repository interfaces defined in core/interfaces.py using
SQLAlchemy 2.0 async ORM. All repositories extend BaseRepository from
aumos-common which provides standard CRUD scaffolding and RLS session handling.
"""

import uuid
from decimal import Decimal

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.database import BaseRepository
from aumos_common.observability import get_logger

from aumos_model_registry.core.models import (
    Experiment,
    ExperimentRun,
    Model,
    ModelDeployment,
    ModelVersion,
)

logger = get_logger(__name__)


class ModelRepository(BaseRepository[Model]):
    """Persistence operations for registered AI/ML models.

    Table: reg_models
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an async SQLAlchemy session.

        Args:
            session: Async database session (RLS context already set by middleware).
        """
        super().__init__(session, Model)

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
        """Create and persist a new model record.

        Args:
            tenant_id: Owning tenant UUID.
            name: Unique model name within the tenant.
            created_by: User UUID performing the registration.
            description: Optional long-form description.
            model_type: Broad category (classification, llm, embedding, …).
            framework: Training framework (pytorch, tensorflow, …).
            tags: Arbitrary key/value metadata.

        Returns:
            Newly created and persisted Model ORM instance.
        """
        model = Model(
            tenant_id=tenant_id,
            name=name,
            created_by=created_by,
            description=description,
            model_type=model_type,
            framework=framework,
            tags=tags,
        )
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        logger.debug("Model created in DB", model_id=str(model.id), name=name)
        return model

    async def get_by_id(self, model_id: uuid.UUID, tenant_id: uuid.UUID) -> Model | None:
        """Fetch a model by UUID within a tenant scope.

        Args:
            model_id: Model UUID.
            tenant_id: Requesting tenant (enforces isolation).

        Returns:
            Model ORM instance or None if not found.
        """
        result = await self._session.execute(
            select(Model).where(
                Model.id == model_id,
                Model.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str, tenant_id: uuid.UUID) -> Model | None:
        """Fetch a model by tenant-scoped name.

        Args:
            name: Model name to look up.
            tenant_id: Owning tenant.

        Returns:
            Model ORM instance or None if not found.
        """
        result = await self._session.execute(
            select(Model).where(
                Model.name == name,
                Model.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_all(
        self,
        tenant_id: uuid.UUID,
        page: int,
        page_size: int,
        model_type: str | None,
        framework: str | None,
    ) -> tuple[list[Model], int]:
        """Return paginated models with optional filters.

        Args:
            tenant_id: Requesting tenant.
            page: 1-based page number.
            page_size: Results per page.
            model_type: Optional filter by model type.
            framework: Optional filter by framework.

        Returns:
            Tuple of (model_list, total_count).
        """
        base_query = select(Model).where(Model.tenant_id == tenant_id)
        if model_type is not None:
            base_query = base_query.where(Model.model_type == model_type)
        if framework is not None:
            base_query = base_query.where(Model.framework == framework)

        count_result = await self._session.execute(
            select(func.count()).select_from(base_query.subquery())
        )
        total: int = count_result.scalar_one()

        offset = (page - 1) * page_size
        items_result = await self._session.execute(
            base_query.order_by(Model.created_at.desc()).offset(offset).limit(page_size)
        )
        items = list(items_result.scalars().all())
        return items, total

    async def search(
        self, tenant_id: uuid.UUID, query: str, limit: int
    ) -> list[Model]:
        """Full-text search over model name, description, and tags.

        Uses PostgreSQL ILIKE for case-insensitive substring matching on name
        and description, and a JSONB text cast for tag values.

        Args:
            tenant_id: Requesting tenant.
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of matching Model instances.
        """
        pattern = f"%{query}%"
        result = await self._session.execute(
            select(Model)
            .where(
                Model.tenant_id == tenant_id,
                or_(
                    Model.name.ilike(pattern),
                    Model.description.ilike(pattern),
                    func.cast(Model.tags, type_=None).cast(str).ilike(pattern),
                ),
            )
            .order_by(Model.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_tags(
        self, model_id: uuid.UUID, tenant_id: uuid.UUID, tags: dict
    ) -> Model:
        """Replace the model's tags JSONB with the provided dict.

        Args:
            model_id: Target model UUID.
            tenant_id: Owning tenant (for isolation check).
            tags: New tags dict to persist.

        Returns:
            Updated Model ORM instance.
        """
        await self._session.execute(
            update(Model)
            .where(Model.id == model_id, Model.tenant_id == tenant_id)
            .values(tags=tags)
        )
        await self._session.flush()
        model = await self.get_by_id(model_id, tenant_id)
        assert model is not None  # noqa: S101 — already validated upstream
        return model

    async def delete(self, model_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
        """Hard-delete a model and cascade to versions.

        Args:
            model_id: Target model UUID.
            tenant_id: Owning tenant.
        """
        model = await self.get_by_id(model_id, tenant_id)
        if model is not None:
            await self._session.delete(model)
            await self._session.flush()


class ModelVersionRepository(BaseRepository[ModelVersion]):
    """Persistence operations for model versions.

    Table: reg_model_versions
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an async SQLAlchemy session.

        Args:
            session: Async database session.
        """
        super().__init__(session, ModelVersion)

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
        """Create a new version with auto-incremented version number.

        Args:
            model_id: Parent model UUID.
            artifact_uri: S3/MinIO URI of stored artifact.
            training_data: Provenance dict describing training datasets.
            hyperparameters: Training hyperparameters used.
            metrics: Evaluation metrics from training run.
            parent_model_id: Base model UUID if fine-tuned.
            training_cost: Compute cost for this training run.
            size_bytes: Artifact size for storage cost tracking.

        Returns:
            Newly created ModelVersion ORM instance.
        """
        next_version_result = await self._session.execute(
            select(func.coalesce(func.max(ModelVersion.version), 0) + 1).where(
                ModelVersion.model_id == model_id
            )
        )
        next_version: int = next_version_result.scalar_one()

        version = ModelVersion(
            model_id=model_id,
            version=next_version,
            artifact_uri=artifact_uri,
            training_data=training_data,
            hyperparameters=hyperparameters,
            metrics=metrics,
            parent_model_id=parent_model_id,
            training_cost=training_cost,
            size_bytes=size_bytes,
        )
        self._session.add(version)
        await self._session.flush()
        await self._session.refresh(version)
        return version

    async def get_by_id(self, version_id: uuid.UUID) -> ModelVersion | None:
        """Fetch a version by UUID.

        Args:
            version_id: Version UUID.

        Returns:
            ModelVersion ORM instance or None.
        """
        result = await self._session.execute(
            select(ModelVersion).where(ModelVersion.id == version_id)
        )
        return result.scalar_one_or_none()

    async def get_by_model_and_number(
        self, model_id: uuid.UUID, version: int
    ) -> ModelVersion | None:
        """Fetch a specific version number for a model.

        Args:
            model_id: Parent model UUID.
            version: Version number (integer).

        Returns:
            ModelVersion ORM instance or None.
        """
        result = await self._session.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == model_id,
                ModelVersion.version == version,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_model(self, model_id: uuid.UUID) -> list[ModelVersion]:
        """Return all versions for a model, newest first.

        Args:
            model_id: Parent model UUID.

        Returns:
            List of ModelVersion instances ordered by descending version number.
        """
        result = await self._session.execute(
            select(ModelVersion)
            .where(ModelVersion.model_id == model_id)
            .order_by(ModelVersion.version.desc())
        )
        return list(result.scalars().all())

    async def transition_stage(
        self, version_id: uuid.UUID, new_stage: str
    ) -> ModelVersion:
        """Update the lifecycle stage of a version.

        Args:
            version_id: Version UUID to update.
            new_stage: Target stage name.

        Returns:
            Updated ModelVersion ORM instance.
        """
        await self._session.execute(
            update(ModelVersion)
            .where(ModelVersion.id == version_id)
            .values(stage=new_stage)
        )
        await self._session.flush()
        version = await self.get_by_id(version_id)
        assert version is not None  # noqa: S101 — validated upstream
        return version

    async def set_ml_bom(
        self, version_id: uuid.UUID, ml_bom: dict
    ) -> ModelVersion:
        """Attach a CycloneDX ML-BOM JSON payload to a version.

        Args:
            version_id: Version UUID.
            ml_bom: CycloneDX-format ML-BOM dict.

        Returns:
            Updated ModelVersion with ml_bom field populated.
        """
        await self._session.execute(
            update(ModelVersion)
            .where(ModelVersion.id == version_id)
            .values(ml_bom=ml_bom)
        )
        await self._session.flush()
        version = await self.get_by_id(version_id)
        assert version is not None  # noqa: S101 — validated upstream
        return version


class ModelLineageRepository:
    """Cross-model lineage queries for the registry.

    Provides graph traversal utilities for fine-tuning lineage chains.
    Does not extend BaseRepository as it operates across multiple tables.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an async SQLAlchemy session.

        Args:
            session: Async database session.
        """
        self._session = session

    async def get_ancestors(
        self,
        model_id: uuid.UUID,
        max_depth: int = 10,
    ) -> list[dict]:
        """Traverse the fine-tuning lineage chain upward from a model.

        Follows parent_model_id pointers across versions to build the
        ancestor chain. Stops at max_depth to prevent infinite loops.

        Args:
            model_id: Starting model UUID.
            max_depth: Maximum number of ancestor hops to follow.

        Returns:
            List of ancestor dicts with model_id, version_id, and depth keys.
        """
        ancestors: list[dict] = []
        visited: set[str] = set()
        current_model_id: uuid.UUID | None = model_id
        depth = 0

        while current_model_id is not None and depth < max_depth:
            model_id_str = str(current_model_id)
            if model_id_str in visited:
                break
            visited.add(model_id_str)

            result = await self._session.execute(
                select(ModelVersion.parent_model_id, ModelVersion.id)
                .where(
                    ModelVersion.model_id == current_model_id,
                    ModelVersion.parent_model_id.is_not(None),
                )
                .order_by(ModelVersion.version.desc())
                .limit(1)
            )
            row = result.one_or_none()
            if row is None:
                break

            parent_model_id, version_id = row
            ancestors.append(
                {
                    "model_id": model_id_str,
                    "version_id": str(version_id),
                    "depth": depth,
                    "parent_model_id": str(parent_model_id) if parent_model_id else None,
                }
            )
            current_model_id = parent_model_id
            depth += 1

        return ancestors


class DeploymentRepository(BaseRepository[ModelDeployment]):
    """Persistence operations for model deployments.

    Table: reg_model_deployments
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an async SQLAlchemy session.

        Args:
            session: Async database session.
        """
        super().__init__(session, ModelDeployment)

    async def create(
        self,
        model_version_id: uuid.UUID,
        tenant_id: uuid.UUID,
        environment: str,
        endpoint_url: str | None,
    ) -> ModelDeployment:
        """Record a new deployment.

        Args:
            model_version_id: Version UUID being deployed.
            tenant_id: Owning tenant.
            environment: Target environment (dev, staging, production).
            endpoint_url: Inference endpoint URL if known.

        Returns:
            Newly created ModelDeployment ORM instance.
        """
        deployment = ModelDeployment(
            model_version_id=model_version_id,
            tenant_id=tenant_id,
            environment=environment,
            endpoint_url=endpoint_url,
            status="pending",
        )
        self._session.add(deployment)
        await self._session.flush()
        await self._session.refresh(deployment)
        return deployment

    async def get_by_id(self, deployment_id: uuid.UUID) -> ModelDeployment | None:
        """Fetch a deployment by UUID.

        Args:
            deployment_id: Deployment UUID.

        Returns:
            ModelDeployment ORM instance or None.
        """
        result = await self._session.execute(
            select(ModelDeployment).where(ModelDeployment.id == deployment_id)
        )
        return result.scalar_one_or_none()

    async def list_by_version(self, model_version_id: uuid.UUID) -> list[ModelDeployment]:
        """Return all deployments for a version.

        Args:
            model_version_id: Version UUID.

        Returns:
            List of ModelDeployment instances.
        """
        result = await self._session.execute(
            select(ModelDeployment)
            .where(ModelDeployment.model_version_id == model_version_id)
            .order_by(ModelDeployment.deployed_at.desc())
        )
        return list(result.scalars().all())

    async def list_by_tenant(
        self, tenant_id: uuid.UUID, environment: str | None
    ) -> list[ModelDeployment]:
        """Return all deployments for a tenant, optionally filtered by environment.

        Args:
            tenant_id: Requesting tenant.
            environment: Optional environment filter.

        Returns:
            List of ModelDeployment instances.
        """
        query = select(ModelDeployment).where(ModelDeployment.tenant_id == tenant_id)
        if environment is not None:
            query = query.where(ModelDeployment.environment == environment)
        result = await self._session.execute(
            query.order_by(ModelDeployment.deployed_at.desc())
        )
        return list(result.scalars().all())

    async def update_status(
        self, deployment_id: uuid.UUID, status: str
    ) -> ModelDeployment:
        """Update deployment status.

        Args:
            deployment_id: Deployment UUID.
            status: New status string (pending | active | inactive | failed).

        Returns:
            Updated ModelDeployment ORM instance.
        """
        await self._session.execute(
            update(ModelDeployment)
            .where(ModelDeployment.id == deployment_id)
            .values(status=status)
        )
        await self._session.flush()
        deployment = await self.get_by_id(deployment_id)
        assert deployment is not None  # noqa: S101 — validated upstream
        return deployment

    async def increment_inference_count(
        self,
        deployment_id: uuid.UUID,
        count: int,
        cost_delta: Decimal,
    ) -> None:
        """Atomically increment inference counter and accumulated cost.

        Args:
            deployment_id: Deployment UUID.
            count: Number of inferences to add.
            cost_delta: Cost amount to add in USD.
        """
        await self._session.execute(
            update(ModelDeployment)
            .where(ModelDeployment.id == deployment_id)
            .values(
                inference_count=ModelDeployment.inference_count + count,
                inference_cost=ModelDeployment.inference_cost + cost_delta,
            )
        )
        await self._session.flush()


class ExperimentRepository(BaseRepository[Experiment]):
    """Persistence operations for experiments and runs.

    Tables: reg_experiments, reg_experiment_runs
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an async SQLAlchemy session.

        Args:
            session: Async database session.
        """
        super().__init__(session, Experiment)

    async def create_experiment(
        self,
        tenant_id: uuid.UUID,
        name: str,
        description: str | None,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            tenant_id: Owning tenant.
            name: Unique experiment name within the tenant.
            description: Optional description.

        Returns:
            Newly created Experiment ORM instance.
        """
        experiment = Experiment(
            tenant_id=tenant_id,
            name=name,
            description=description,
        )
        self._session.add(experiment)
        await self._session.flush()
        await self._session.refresh(experiment)
        return experiment

    async def get_experiment(
        self, experiment_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Experiment | None:
        """Fetch an experiment by UUID within a tenant scope.

        Args:
            experiment_id: Experiment UUID.
            tenant_id: Requesting tenant.

        Returns:
            Experiment ORM instance or None.
        """
        result = await self._session.execute(
            select(Experiment).where(
                Experiment.id == experiment_id,
                Experiment.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_experiments(self, tenant_id: uuid.UUID) -> list[Experiment]:
        """Return all experiments for a tenant.

        Args:
            tenant_id: Requesting tenant.

        Returns:
            List of Experiment ORM instances.
        """
        result = await self._session.execute(
            select(Experiment)
            .where(Experiment.tenant_id == tenant_id)
            .order_by(Experiment.created_at.desc())
        )
        return list(result.scalars().all())

    async def create_run(
        self,
        experiment_id: uuid.UUID,
        tenant_id: uuid.UUID,
        parameters: dict,
    ) -> ExperimentRun:
        """Start a new experiment run.

        Args:
            experiment_id: Parent experiment UUID.
            tenant_id: Owning tenant.
            parameters: Initial hyperparameter values.

        Returns:
            Newly created ExperimentRun in 'running' status.
        """
        run = ExperimentRun(
            experiment_id=experiment_id,
            tenant_id=tenant_id,
            parameters=parameters,
            status="running",
        )
        self._session.add(run)
        await self._session.flush()
        await self._session.refresh(run)
        return run

    async def get_run(self, run_id: uuid.UUID) -> ExperimentRun | None:
        """Fetch a run by UUID.

        Args:
            run_id: Run UUID.

        Returns:
            ExperimentRun ORM instance or None.
        """
        result = await self._session.execute(
            select(ExperimentRun).where(ExperimentRun.id == run_id)
        )
        return result.scalar_one_or_none()

    async def list_runs(self, experiment_id: uuid.UUID) -> list[ExperimentRun]:
        """Return all runs for an experiment.

        Args:
            experiment_id: Parent experiment UUID.

        Returns:
            List of ExperimentRun instances ordered by started_at descending.
        """
        result = await self._session.execute(
            select(ExperimentRun)
            .where(ExperimentRun.experiment_id == experiment_id)
            .order_by(ExperimentRun.started_at.desc())
        )
        return list(result.scalars().all())

    async def update_run(
        self,
        run_id: uuid.UUID,
        metrics: dict | None,
        artifacts: list | None,
        status: str | None,
    ) -> ExperimentRun:
        """Update metrics, artifacts, and/or status on a run.

        Args:
            run_id: Run UUID to update.
            metrics: Dict of metric_name → value pairs (merged, not replaced).
            artifacts: List of artifact descriptors to append.
            status: Terminal status if run completed.

        Returns:
            Updated ExperimentRun ORM instance.
        """
        run = await self.get_run(run_id)
        assert run is not None  # noqa: S101 — validated upstream

        if metrics is not None:
            merged_metrics = {**run.metrics, **metrics}
            run.metrics = merged_metrics  # type: ignore[assignment]

        if artifacts is not None:
            run.artifacts = list(run.artifacts) + artifacts  # type: ignore[assignment]

        if status is not None:
            run.status = status  # type: ignore[assignment]

        await self._session.flush()
        await self._session.refresh(run)
        return run
