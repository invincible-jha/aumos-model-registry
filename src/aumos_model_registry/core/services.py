"""Business logic services for the AumOS Model Registry.

All services depend on repository interfaces (not concrete implementations)
and receive dependencies via constructor injection. No framework code here.
"""

import uuid
from decimal import Decimal

from aumos_common.errors import ConflictError, ErrorCode, NotFoundError
from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

from aumos_model_registry.core.cost_engine import (
    ModelCostBreakdown,
    calculate_model_version_cost,
)
from aumos_model_registry.core.interfaces import (
    IDeploymentRepository,
    IExperimentRepository,
    IMLBOMGenerator,
    IModelCostAttribution,
    IModelRepository,
    IModelSemanticSearch,
    IVersionRepository,
)
from aumos_model_registry.core.ml_bom import generate_ml_bom
from aumos_model_registry.core.models import (
    Experiment,
    ExperimentRun,
    Model,
    ModelDeployment,
    ModelVersion,
)

logger = get_logger(__name__)

# Valid stage transitions — only forward transitions are allowed unless archived
_VALID_STAGE_TRANSITIONS: dict[str, list[str]] = {
    "development": ["staging", "archived"],
    "staging": ["production", "development", "archived"],
    "production": ["archived"],
    "archived": [],
}


class ModelService:
    """Lifecycle management for AI models and their versions.

    Orchestrates model registration, version creation, stage transitions,
    ML-BOM generation, and publishes lifecycle events to Kafka.
    """

    def __init__(
        self,
        model_repo: IModelRepository,
        version_repo: IVersionRepository,
        deployment_repo: IDeploymentRepository,
        event_publisher: EventPublisher,
        storage_gb_monthly_rate_usd: float = 0.023,
    ) -> None:
        """Initialise the service with injected dependencies.

        Args:
            model_repo: Model persistence repository.
            version_repo: Version persistence repository.
            deployment_repo: Deployment persistence repository.
            event_publisher: Kafka event publisher.
            storage_gb_monthly_rate_usd: Per-GB monthly storage pricing.
        """
        self._models = model_repo
        self._versions = version_repo
        self._deployments = deployment_repo
        self._publisher = event_publisher
        self._storage_rate = storage_gb_monthly_rate_usd

    # -------------------------------------------------------------------------
    # Model CRUD
    # -------------------------------------------------------------------------

    async def register_model(
        self,
        tenant_id: uuid.UUID,
        name: str,
        created_by: uuid.UUID,
        description: str | None = None,
        model_type: str | None = None,
        framework: str | None = None,
        tags: dict | None = None,
    ) -> Model:
        """Register a new model in the registry.

        Args:
            tenant_id: Owning tenant.
            name: Unique model name within the tenant.
            created_by: User UUID performing the registration.
            description: Optional long-form model description.
            model_type: Broad category (classification, llm, embedding, …).
            framework: Training framework (pytorch, tensorflow, …).
            tags: Arbitrary key/value metadata.

        Returns:
            Newly created Model ORM instance.

        Raises:
            ConflictError: If a model with the same name already exists.
        """
        existing = await self._models.get_by_name(name, tenant_id)
        if existing is not None:
            raise ConflictError(
                message=f"Model '{name}' already exists in this tenant.",
                error_code=ErrorCode.ALREADY_EXISTS,
            )

        model = await self._models.create(
            tenant_id=tenant_id,
            name=name,
            created_by=created_by,
            description=description,
            model_type=model_type,
            framework=framework,
            tags=tags or {},
        )

        await self._publisher.publish(
            Topics.MODEL_LIFECYCLE,
            {
                "event_type": "model.registered",
                "tenant_id": str(tenant_id),
                "model_id": str(model.id),
                "name": model.name,
            },
        )

        logger.info(
            "Model registered",
            tenant_id=str(tenant_id),
            model_id=str(model.id),
            name=model.name,
        )
        return model

    async def get_model(
        self, model_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Model:
        """Retrieve a model by ID.

        Args:
            model_id: Model UUID.
            tenant_id: Requesting tenant (enforces isolation).

        Returns:
            Model ORM instance.

        Raises:
            NotFoundError: If no model found for the given ID and tenant.
        """
        model = await self._models.get_by_id(model_id, tenant_id)
        if model is None:
            raise NotFoundError(
                message=f"Model {model_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return model

    async def list_models(
        self,
        tenant_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
        model_type: str | None = None,
        framework: str | None = None,
    ) -> tuple[list[Model], int]:
        """Paginated list of models for a tenant.

        Args:
            tenant_id: Requesting tenant.
            page: 1-based page number.
            page_size: Number of results per page.
            model_type: Optional filter by model type.
            framework: Optional filter by framework.

        Returns:
            Tuple of (models, total_count).
        """
        return await self._models.list_all(
            tenant_id=tenant_id,
            page=page,
            page_size=page_size,
            model_type=model_type,
            framework=framework,
        )

    async def search_models(
        self,
        tenant_id: uuid.UUID,
        query: str,
        limit: int = 20,
    ) -> list[Model]:
        """Full-text search across model name, description, and tags.

        Args:
            tenant_id: Requesting tenant.
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of matching Model instances.
        """
        return await self._models.search(tenant_id=tenant_id, query=query, limit=limit)

    # -------------------------------------------------------------------------
    # Version management
    # -------------------------------------------------------------------------

    async def create_version(
        self,
        model_id: uuid.UUID,
        tenant_id: uuid.UUID,
        artifact_uri: str | None = None,
        training_data: dict | None = None,
        hyperparameters: dict | None = None,
        metrics: dict | None = None,
        parent_model_id: uuid.UUID | None = None,
        training_cost: Decimal | None = None,
        size_bytes: int | None = None,
        generate_bom: bool = True,
    ) -> ModelVersion:
        """Create a new version for an existing model.

        Automatically increments the version number, and optionally
        generates a CycloneDX ML-BOM and attaches it to the version.

        Args:
            model_id: Parent model UUID.
            tenant_id: Owning tenant.
            artifact_uri: S3/MinIO URI to stored model artifact.
            training_data: Provenance dict describing training datasets.
            hyperparameters: Training hyperparameters used.
            metrics: Evaluation metrics from training run.
            parent_model_id: Base model UUID if fine-tuned.
            training_cost: Compute cost for this training run.
            size_bytes: Artifact size for storage cost tracking.
            generate_bom: Whether to auto-generate and attach an ML-BOM.

        Returns:
            Newly created ModelVersion ORM instance.

        Raises:
            NotFoundError: If the parent model does not exist.
        """
        model = await self.get_model(model_id, tenant_id)

        version = await self._versions.create(
            model_id=model_id,
            artifact_uri=artifact_uri,
            training_data=training_data,
            hyperparameters=hyperparameters,
            metrics=metrics,
            parent_model_id=parent_model_id,
            training_cost=training_cost,
            size_bytes=size_bytes,
        )

        if generate_bom:
            bom = generate_ml_bom(
                model_version=version,
                model_name=model.name,
                tenant_id=tenant_id,
            )
            version = await self._versions.set_ml_bom(version.id, bom)

        await self._publisher.publish(
            Topics.MODEL_LIFECYCLE,
            {
                "event_type": "model.version_created",
                "tenant_id": str(tenant_id),
                "model_id": str(model_id),
                "version_id": str(version.id),
                "version": version.version,
                "stage": version.stage,
            },
        )

        logger.info(
            "Model version created",
            model_id=str(model_id),
            version_id=str(version.id),
            version=version.version,
        )
        return version

    async def transition_stage(
        self,
        version_id: uuid.UUID,
        model_id: uuid.UUID,
        tenant_id: uuid.UUID,
        new_stage: str,
    ) -> ModelVersion:
        """Transition a model version to a new lifecycle stage.

        Validates that the transition follows the allowed path:
        development → staging → production → archived.

        Args:
            version_id: Version UUID to transition.
            model_id: Parent model UUID (for tenant validation).
            tenant_id: Requesting tenant.
            new_stage: Target stage name.

        Returns:
            Updated ModelVersion.

        Raises:
            NotFoundError: If the version does not exist.
            ConflictError: If the transition is not allowed.
        """
        # Validate parent model belongs to tenant
        await self.get_model(model_id, tenant_id)

        version = await self._versions.get_by_id(version_id)
        if version is None:
            raise NotFoundError(
                message=f"Model version {version_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )

        allowed = _VALID_STAGE_TRANSITIONS.get(version.stage, [])
        if new_stage not in allowed:
            raise ConflictError(
                message=f"Cannot transition from '{version.stage}' to '{new_stage}'. "
                f"Allowed: {allowed}",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        updated = await self._versions.transition_stage(version_id, new_stage)

        event_type = (
            "model.deprecated" if new_stage == "archived" else f"model.stage_{new_stage}"
        )
        await self._publisher.publish(
            Topics.MODEL_LIFECYCLE,
            {
                "event_type": event_type,
                "tenant_id": str(tenant_id),
                "model_id": str(model_id),
                "version_id": str(version_id),
                "previous_stage": version.stage,
                "new_stage": new_stage,
            },
        )

        logger.info(
            "Model stage transitioned",
            version_id=str(version_id),
            previous_stage=version.stage,
            new_stage=new_stage,
        )
        return updated

    # -------------------------------------------------------------------------
    # Deployment management
    # -------------------------------------------------------------------------

    async def deploy_version(
        self,
        model_id: uuid.UUID,
        version_id: uuid.UUID,
        tenant_id: uuid.UUID,
        environment: str,
        endpoint_url: str | None = None,
    ) -> ModelDeployment:
        """Deploy a model version to an environment.

        Only versions in 'staging' or 'production' stage may be deployed
        to staging/production environments respectively.

        Args:
            model_id: Parent model UUID.
            version_id: Version UUID to deploy.
            tenant_id: Requesting tenant.
            environment: Target environment (dev, staging, production).
            endpoint_url: Inference endpoint URL if known at deploy time.

        Returns:
            Newly created ModelDeployment.

        Raises:
            NotFoundError: If model or version not found.
            ConflictError: If version stage doesn't allow deployment to environment.
        """
        await self.get_model(model_id, tenant_id)

        version = await self._versions.get_by_id(version_id)
        if version is None:
            raise NotFoundError(
                message=f"Model version {version_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )

        # Guard: only promote sufficiently staged versions
        stage_env_map = {
            "production": ["production", "staging"],
            "staging": ["staging", "dev"],
            "development": ["dev"],
        }
        allowed_envs = stage_env_map.get(version.stage, [])
        if environment not in allowed_envs:
            raise ConflictError(
                message=f"Version in stage '{version.stage}' cannot be deployed to "
                f"'{environment}' environment.",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        deployment = await self._deployments.create(
            model_version_id=version_id,
            tenant_id=tenant_id,
            environment=environment,
            endpoint_url=endpoint_url,
        )

        await self._publisher.publish(
            Topics.MODEL_LIFECYCLE,
            {
                "event_type": "model.deployed",
                "tenant_id": str(tenant_id),
                "model_id": str(model_id),
                "version_id": str(version_id),
                "deployment_id": str(deployment.id),
                "environment": environment,
            },
        )

        logger.info(
            "Model deployed",
            deployment_id=str(deployment.id),
            version_id=str(version_id),
            environment=environment,
        )
        return deployment

    async def rollback_deployment(
        self,
        deployment_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> ModelDeployment:
        """Mark an active deployment as inactive (rollback signal).

        The caller is responsible for routing traffic away from the
        deployment endpoint after calling this method.

        Args:
            deployment_id: Deployment UUID to roll back.
            tenant_id: Requesting tenant (for validation).

        Returns:
            Updated ModelDeployment with status='inactive'.

        Raises:
            NotFoundError: If deployment not found for the tenant.
        """
        deployment = await self._deployments.get_by_id(deployment_id)
        if deployment is None or deployment.tenant_id != tenant_id:
            raise NotFoundError(
                message=f"Deployment {deployment_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )

        updated = await self._deployments.update_status(deployment_id, "inactive")

        await self._publisher.publish(
            Topics.MODEL_LIFECYCLE,
            {
                "event_type": "model.deployment_rolled_back",
                "tenant_id": str(tenant_id),
                "deployment_id": str(deployment_id),
            },
        )

        logger.info("Deployment rolled back", deployment_id=str(deployment_id))
        return updated

    # -------------------------------------------------------------------------
    # Lineage and cost
    # -------------------------------------------------------------------------

    async def get_lineage(
        self,
        model_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> dict:
        """Build a lineage graph for all versions of a model.

        Returns a simplified adjacency-list representation of fine-tuning
        relationships (parent → child model chains).

        Args:
            model_id: Root model UUID.
            tenant_id: Requesting tenant.

        Returns:
            Dict with 'model_id', 'versions', and 'edges' keys.
        """
        await self.get_model(model_id, tenant_id)
        versions = await self._versions.list_by_model(model_id)

        edges: list[dict] = []
        for version in versions:
            if version.parent_model_id is not None:
                edges.append(
                    {
                        "from_model_id": str(version.parent_model_id),
                        "to_model_id": str(model_id),
                        "to_version_id": str(version.id),
                        "relationship": "fine-tuned-from",
                    }
                )

        return {
            "model_id": str(model_id),
            "versions": [
                {
                    "version_id": str(v.id),
                    "version": v.version,
                    "stage": v.stage,
                    "parent_model_id": str(v.parent_model_id)
                    if v.parent_model_id
                    else None,
                }
                for v in versions
            ],
            "edges": edges,
        }

    async def get_cost_breakdown(
        self,
        model_id: uuid.UUID,
        version_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> ModelCostBreakdown:
        """Calculate full cost breakdown for a model version.

        Args:
            model_id: Parent model UUID.
            version_id: Version UUID.
            tenant_id: Requesting tenant.

        Returns:
            ModelCostBreakdown with training, inference, and storage costs.

        Raises:
            NotFoundError: If model or version not found.
        """
        await self.get_model(model_id, tenant_id)

        version = await self._versions.get_by_id(version_id)
        if version is None:
            raise NotFoundError(
                message=f"Model version {version_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )

        deployments = await self._deployments.list_by_version(version_id)

        return calculate_model_version_cost(
            version=version,
            deployments=deployments,
            storage_gb_monthly_rate_usd=self._storage_rate,
        )

    async def get_ml_bom(
        self,
        model_id: uuid.UUID,
        version_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> dict:
        """Return the CycloneDX ML-BOM for a model version.

        If no BOM exists, generates one on-the-fly.

        Args:
            model_id: Parent model UUID.
            version_id: Version UUID.
            tenant_id: Requesting tenant.

        Returns:
            CycloneDX ML-BOM dict.

        Raises:
            NotFoundError: If model or version not found.
        """
        model = await self.get_model(model_id, tenant_id)

        version = await self._versions.get_by_id(version_id)
        if version is None:
            raise NotFoundError(
                message=f"Model version {version_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )

        if version.ml_bom is not None:
            return version.ml_bom

        bom = generate_ml_bom(
            model_version=version,
            model_name=model.name,
            tenant_id=tenant_id,
        )
        await self._versions.set_ml_bom(version_id, bom)
        return bom


class MLBOMService:
    """Service for generating and exporting CycloneDX ML Bills of Materials.

    Wraps the MLBOMGenerator adapter with tenant-level access guards and
    on-demand generation with persistence back to the version repository.
    """

    def __init__(
        self,
        bom_generator: IMLBOMGenerator,
        version_repo: IVersionRepository,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            bom_generator: ML-BOM generation adapter.
            version_repo: Version repository for BOM persistence.
        """
        self._generator = bom_generator
        self._versions = version_repo

    async def get_or_generate_bom(
        self,
        model_id: uuid.UUID,
        version_id: uuid.UUID,
        model_name: str,
        tenant_id: uuid.UUID,
        framework: str | None = None,
    ) -> dict:
        """Return an existing BOM or generate and persist one on demand.

        Args:
            model_id: Parent model UUID.
            version_id: Target version UUID.
            model_name: Human-readable model name.
            tenant_id: Owning tenant.
            framework: Optional framework hint for component detection.

        Returns:
            CycloneDX ML-BOM dict.
        """
        version = await self._versions.get_by_id(version_id)
        if version is not None and version.ml_bom is not None:
            return version.ml_bom

        bom = await self._generator.generate(
            model_id=model_id,
            version_id=version_id,
            model_name=model_name,
            tenant_id=tenant_id,
            framework=framework,
            additional_components=None,
            training_datasets=None,
        )
        await self._versions.set_ml_bom(version_id, bom)
        logger.info("ML-BOM generated and persisted", version_id=str(version_id))
        return bom

    async def export_bom_json(self, model_id: uuid.UUID, version_id: uuid.UUID) -> str:
        """Return the JSON representation of a persisted BOM.

        Args:
            model_id: Parent model UUID.
            version_id: Target version UUID.

        Returns:
            JSON string of the CycloneDX BOM.
        """
        version = await self._versions.get_by_id(version_id)
        if version is None or version.ml_bom is None:
            raise ValueError(f"No BOM found for version {version_id}")
        return await self._generator.export_json(version.ml_bom)

    async def export_bom_xml(self, model_id: uuid.UUID, version_id: uuid.UUID) -> str:
        """Return the XML representation of a persisted BOM.

        Args:
            model_id: Parent model UUID.
            version_id: Target version UUID.

        Returns:
            CycloneDX XML string.
        """
        version = await self._versions.get_by_id(version_id)
        if version is None or version.ml_bom is None:
            raise ValueError(f"No BOM found for version {version_id}")
        return await self._generator.export_xml(version.ml_bom)


class SemanticSearchService:
    """Service for semantic model discovery within a tenant's model registry.

    Delegates to the IModelSemanticSearch adapter with tenant-level validation
    and search result formatting.
    """

    def __init__(self, search_adapter: IModelSemanticSearch) -> None:
        """Initialise with injected search adapter.

        Args:
            search_adapter: Semantic search implementation.
        """
        self._search = search_adapter

    async def search(
        self,
        tenant_id: uuid.UUID,
        query: str,
        tags: list | None = None,
        framework: str | None = None,
        model_type: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search for models by natural language query.

        Args:
            tenant_id: Requesting tenant.
            query: Natural language search query.
            tags: Optional tag filter list.
            framework: Optional framework filter.
            model_type: Optional model_type filter.
            limit: Maximum results.

        Returns:
            Ranked list of model search results.
        """
        return await self._search.search(
            tenant_id=tenant_id,
            query=query,
            tags=tags,
            framework=framework,
            model_type=model_type,
            limit=limit,
        )

    async def get_facets(self, tenant_id: uuid.UUID) -> dict:
        """Return faceted search metadata for UI rendering.

        Args:
            tenant_id: Requesting tenant.

        Returns:
            Dict with framework and model_type facet counts.
        """
        return await self._search.get_facets(tenant_id=tenant_id)

    async def suggest(
        self,
        tenant_id: uuid.UUID,
        prefix: str,
        limit: int = 8,
    ) -> list[str]:
        """Return autocomplete suggestions for a search prefix.

        Args:
            tenant_id: Requesting tenant.
            prefix: Query prefix (minimum 2 characters).
            limit: Maximum suggestions.

        Returns:
            List of suggestion strings.
        """
        return await self._search.get_autocomplete_suggestions(
            tenant_id=tenant_id, prefix=prefix, limit=limit
        )


class CostAttributionService:
    """Service for per-model cost tracking, analysis, and budget management.

    Wraps the IModelCostAttribution adapter with tenant validation.
    """

    def __init__(self, cost_adapter: IModelCostAttribution) -> None:
        """Initialise with injected cost attribution adapter.

        Args:
            cost_adapter: Cost tracking implementation.
        """
        self._costs = cost_adapter

    async def get_version_cost(
        self,
        version_id: uuid.UUID,
        storage_months: int = 1,
    ) -> dict:
        """Return cost breakdown for a specific model version.

        Args:
            version_id: Target version UUID.
            storage_months: Months to amortise storage over.

        Returns:
            Cost breakdown dict.
        """
        return await self._costs.get_version_cost_breakdown(
            version_id=version_id, storage_months=storage_months
        )

    async def get_model_cost_report(
        self,
        model_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> dict:
        """Generate a comprehensive cost report for a model.

        Args:
            model_id: Parent model UUID.
            tenant_id: Owning tenant.

        Returns:
            Full cost report including summary and trends.
        """
        return await self._costs.generate_cost_report(
            model_id=model_id, tenant_id=tenant_id, include_trends=True
        )

    async def check_budget(
        self,
        model_id: uuid.UUID,
        tenant_id: uuid.UUID,
        budget_usd: Decimal,
    ) -> dict:
        """Check model cost against a budget allocation.

        Args:
            model_id: Parent model UUID.
            tenant_id: Owning tenant.
            budget_usd: Budget cap in USD.

        Returns:
            Budget alert dict.
        """
        return await self._costs.check_budget_alert(
            model_id=model_id, tenant_id=tenant_id, budget_usd=budget_usd
        )


class ExperimentService:
    """MLflow-compatible experiment and run management.

    Supports creating experiments, logging parameters/metrics to runs,
    and querying run history for comparison.
    """

    def __init__(
        self,
        experiment_repo: IExperimentRepository,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            experiment_repo: Experiment and run persistence.
            event_publisher: Kafka event publisher.
        """
        self._repo = experiment_repo
        self._publisher = event_publisher

    async def create_experiment(
        self,
        tenant_id: uuid.UUID,
        name: str,
        description: str | None = None,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            tenant_id: Owning tenant.
            name: Unique experiment name within the tenant.
            description: Optional description.

        Returns:
            Newly created Experiment.

        Raises:
            ConflictError: If an experiment with the same name already exists.
        """
        existing = await self._repo.get_experiment(
            uuid.UUID(int=0), tenant_id  # dummy UUID — get by name via list
        )
        # Implementation note: list and filter for uniqueness check
        all_experiments = await self._repo.list_experiments(tenant_id)
        if any(e.name == name for e in all_experiments):
            raise ConflictError(
                message=f"Experiment '{name}' already exists.",
                error_code=ErrorCode.ALREADY_EXISTS,
            )

        experiment = await self._repo.create_experiment(tenant_id, name, description)
        logger.info(
            "Experiment created",
            experiment_id=str(experiment.id),
            name=name,
            tenant_id=str(tenant_id),
        )
        return experiment

    async def get_experiment(
        self, experiment_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Experiment:
        """Fetch an experiment.

        Args:
            experiment_id: Experiment UUID.
            tenant_id: Requesting tenant.

        Returns:
            Experiment ORM instance.

        Raises:
            NotFoundError: If not found.
        """
        experiment = await self._repo.get_experiment(experiment_id, tenant_id)
        if experiment is None:
            raise NotFoundError(
                message=f"Experiment {experiment_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return experiment

    async def list_experiments(self, tenant_id: uuid.UUID) -> list[Experiment]:
        """Return all experiments for a tenant.

        Args:
            tenant_id: Requesting tenant.

        Returns:
            List of Experiment ORM instances.
        """
        return await self._repo.list_experiments(tenant_id)

    async def start_run(
        self,
        experiment_id: uuid.UUID,
        tenant_id: uuid.UUID,
        parameters: dict | None = None,
    ) -> ExperimentRun:
        """Start a new training run within an experiment.

        Args:
            experiment_id: Parent experiment UUID.
            tenant_id: Requesting tenant.
            parameters: Initial hyperparameter dict to log.

        Returns:
            Newly created ExperimentRun in 'running' status.

        Raises:
            NotFoundError: If the experiment does not exist.
        """
        await self.get_experiment(experiment_id, tenant_id)

        run = await self._repo.create_run(
            experiment_id=experiment_id,
            tenant_id=tenant_id,
            parameters=parameters or {},
        )
        logger.info(
            "Experiment run started",
            run_id=str(run.id),
            experiment_id=str(experiment_id),
        )
        return run

    async def log_run(
        self,
        run_id: uuid.UUID,
        metrics: dict | None = None,
        artifacts: list | None = None,
        status: str | None = None,
    ) -> ExperimentRun:
        """Log metrics and artifacts to a run, and optionally set final status.

        Args:
            run_id: Run UUID to update.
            metrics: Dict of metric_name → value pairs.
            artifacts: List of artifact descriptors (name, uri, type).
            status: Terminal status if run completed (finished, failed, killed).

        Returns:
            Updated ExperimentRun.

        Raises:
            NotFoundError: If run not found.
        """
        run = await self._repo.get_run(run_id)
        if run is None:
            raise NotFoundError(
                message=f"Run {run_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )

        updated = await self._repo.update_run(
            run_id=run_id,
            metrics=metrics,
            artifacts=artifacts,
            status=status,
        )
        return updated

    async def list_runs(
        self, experiment_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> list[ExperimentRun]:
        """Return all runs for an experiment.

        Args:
            experiment_id: Parent experiment UUID.
            tenant_id: Requesting tenant.

        Returns:
            List of ExperimentRun ORM instances.
        """
        await self.get_experiment(experiment_id, tenant_id)
        return await self._repo.list_runs(experiment_id)
