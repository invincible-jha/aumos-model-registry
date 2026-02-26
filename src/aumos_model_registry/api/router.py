"""FastAPI router for the AumOS Model Registry API.

All routes are thin: they validate inputs via Pydantic schemas, delegate
business logic to ModelService or ExperimentService, and return Pydantic
response schemas. No domain logic lives in this module.
"""

import math
import uuid

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import get_current_tenant, get_current_user
from aumos_common.database import get_db_session
from aumos_common.errors import NotFoundError, ErrorCode
from aumos_common.observability import get_logger

from aumos_model_registry.adapters.repositories import (
    DeploymentRepository,
    ExperimentRepository,
    ModelRepository,
    ModelVersionRepository,
)
from aumos_model_registry.api.schemas import (
    CostBreakdownResponse,
    DeploymentCreateRequest,
    DeploymentResponse,
    ExperimentCreateRequest,
    ExperimentResponse,
    LineageResponse,
    ModelCreateRequest,
    ModelListResponse,
    ModelResponse,
    ModelSearchResponse,
    PaginationMeta,
    RollbackRequest,
    RunCreateRequest,
    RunResponse,
    RunUpdateRequest,
    StageTransitionRequest,
    VersionCreateRequest,
    VersionResponse,
)
from aumos_model_registry.core.services import ExperimentService, ModelService

logger = get_logger(__name__)

router = APIRouter(tags=["Model Registry"])


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _model_service(
    session: AsyncSession = Depends(get_db_session),
    request: Request = None,  # type: ignore[assignment]
) -> ModelService:
    """Build a ModelService with repository and event publisher dependencies.

    Args:
        session: SQLAlchemy async session (injected by FastAPI).
        request: FastAPI request (provides app.state for Kafka publisher).

    Returns:
        Configured ModelService instance.
    """
    model_repo = ModelRepository(session)
    version_repo = ModelVersionRepository(session)
    deployment_repo = DeploymentRepository(session)
    publisher = request.app.state.kafka_publisher
    return ModelService(
        model_repo=model_repo,
        version_repo=version_repo,
        deployment_repo=deployment_repo,
        event_publisher=publisher,
    )


def _experiment_service(
    session: AsyncSession = Depends(get_db_session),
    request: Request = None,  # type: ignore[assignment]
) -> ExperimentService:
    """Build an ExperimentService with repository and event publisher dependencies.

    Args:
        session: SQLAlchemy async session (injected by FastAPI).
        request: FastAPI request (provides app.state for Kafka publisher).

    Returns:
        Configured ExperimentService instance.
    """
    experiment_repo = ExperimentRepository(session)
    publisher = request.app.state.kafka_publisher
    return ExperimentService(
        experiment_repo=experiment_repo,
        event_publisher=publisher,
    )


# ---------------------------------------------------------------------------
# Model CRUD endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/models",
    response_model=ModelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new model",
)
async def register_model(
    body: ModelCreateRequest,
    tenant=Depends(get_current_tenant),
    user=Depends(get_current_user),
    service: ModelService = Depends(_model_service),
) -> ModelResponse:
    """Register a new AI/ML model in the registry.

    Creates a new model record scoped to the requesting tenant. The model
    name must be unique within the tenant.

    Args:
        body: Model creation request payload.
        tenant: Authenticated tenant context (from JWT).
        user: Authenticated user context (from JWT).
        service: Injected ModelService.

    Returns:
        Newly registered model resource.
    """
    model = await service.register_model(
        tenant_id=tenant.tenant_id,
        name=body.name,
        created_by=user.user_id,
        description=body.description,
        model_type=body.model_type,
        framework=body.framework,
        tags=body.tags,
    )
    return ModelResponse.model_validate(model)


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List models",
)
async def list_models(
    page: int = 1,
    page_size: int = 20,
    model_type: str | None = None,
    framework: str | None = None,
    tenant=Depends(get_current_tenant),
    service: ModelService = Depends(_model_service),
) -> ModelListResponse:
    """Return a paginated list of models for the authenticated tenant.

    Args:
        page: 1-based page number (default 1).
        page_size: Results per page (default 20, max 100).
        model_type: Optional filter by model type.
        framework: Optional filter by training framework.
        tenant: Authenticated tenant context.
        service: Injected ModelService.

    Returns:
        Paginated list of model resources.
    """
    page_size = min(page_size, 100)
    models, total = await service.list_models(
        tenant_id=tenant.tenant_id,
        page=page,
        page_size=page_size,
        model_type=model_type,
        framework=framework,
    )
    total_pages = max(1, math.ceil(total / page_size))
    return ModelListResponse(
        items=[ModelResponse.model_validate(m) for m in models],
        pagination=PaginationMeta(
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
        ),
    )


@router.get(
    "/models/{model_id}",
    response_model=ModelResponse,
    summary="Get model details",
)
async def get_model(
    model_id: uuid.UUID,
    tenant=Depends(get_current_tenant),
    service: ModelService = Depends(_model_service),
) -> ModelResponse:
    """Retrieve a single model by ID.

    Args:
        model_id: Model UUID path parameter.
        tenant: Authenticated tenant context.
        service: Injected ModelService.

    Returns:
        Model resource.
    """
    model = await service.get_model(model_id, tenant.tenant_id)
    return ModelResponse.model_validate(model)


@router.put(
    "/models/{model_id}",
    response_model=ModelResponse,
    summary="Update model metadata",
)
async def update_model(
    model_id: uuid.UUID,
    body: ModelCreateRequest,
    tenant=Depends(get_current_tenant),
    service: ModelService = Depends(_model_service),
) -> ModelResponse:
    """Update a model's tags and metadata.

    Currently supports updating tags only. Name and framework are immutable
    after registration.

    Args:
        model_id: Model UUID path parameter.
        body: Updated model fields.
        tenant: Authenticated tenant context.
        service: Injected ModelService.

    Returns:
        Updated model resource.
    """
    model = await service.get_model(model_id, tenant.tenant_id)
    model_repo = service._models  # noqa: SLF001
    updated = await model_repo.update_tags(model_id, tenant.tenant_id, body.tags)
    return ModelResponse.model_validate(updated)


@router.delete(
    "/models/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete model",
)
async def delete_model(
    model_id: uuid.UUID,
    tenant=Depends(get_current_tenant),
    service: ModelService = Depends(_model_service),
) -> None:
    """Delete a model and all its versions.

    This is a hard delete — all version records, deployments, and ML-BOMs
    associated with the model are permanently removed.

    Args:
        model_id: Model UUID path parameter.
        tenant: Authenticated tenant context.
        service: Injected ModelService.
    """
    await service.get_model(model_id, tenant.tenant_id)
    model_repo = service._models  # noqa: SLF001
    await model_repo.delete(model_id, tenant.tenant_id)


# ---------------------------------------------------------------------------
# Model version endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/models/{model_id}/versions",
    response_model=VersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create model version",
)
async def create_model_version(
    model_id: uuid.UUID,
    body: VersionCreateRequest,
    tenant=Depends(get_current_tenant),
    service: ModelService = Depends(_model_service),
) -> VersionResponse:
    """Create a new version for an existing model.

    Automatically increments the version number. Optionally generates a
    CycloneDX ML-BOM and attaches it to the version record.

    Args:
        model_id: Parent model UUID path parameter.
        body: Version creation request payload.
        tenant: Authenticated tenant context.
        service: Injected ModelService.

    Returns:
        Newly created model version resource.
    """
    version = await service.create_version(
        model_id=model_id,
        tenant_id=tenant.tenant_id,
        artifact_uri=body.artifact_uri,
        training_data=body.training_data.model_dump() if body.training_data else None,
        hyperparameters=body.hyperparameters,
        metrics=body.metrics,
        parent_model_id=body.parent_model_id,
        training_cost=body.training_cost_usd,
        size_bytes=body.size_bytes,
        generate_bom=body.generate_bom,
    )
    return VersionResponse.model_validate(version)


@router.get(
    "/models/{model_id}/versions",
    response_model=list[VersionResponse],
    summary="List model versions",
)
async def list_model_versions(
    model_id: uuid.UUID,
    tenant=Depends(get_current_tenant),
    service: ModelService = Depends(_model_service),
) -> list[VersionResponse]:
    """Return all versions for a model, newest first.

    Args:
        model_id: Parent model UUID path parameter.
        tenant: Authenticated tenant context.
        service: Injected ModelService.

    Returns:
        List of model version resources.
    """
    await service.get_model(model_id, tenant.tenant_id)
    version_repo = service._versions  # noqa: SLF001
    versions = await version_repo.list_by_model(model_id)
    return [VersionResponse.model_validate(v) for v in versions]


@router.patch(
    "/models/{model_id}/versions/{version_id}/stage",
    response_model=VersionResponse,
    summary="Transition model version stage",
)
async def transition_model_stage(
    model_id: uuid.UUID,
    version_id: uuid.UUID,
    body: StageTransitionRequest,
    tenant=Depends(get_current_tenant),
    service: ModelService = Depends(_model_service),
) -> VersionResponse:
    """Transition a model version to a new lifecycle stage.

    Allowed transitions: development → staging → production → archived.
    Reverse transitions (e.g., production → staging) are not permitted.

    Args:
        model_id: Parent model UUID path parameter.
        version_id: Version UUID path parameter.
        body: Stage transition request with target stage.
        tenant: Authenticated tenant context.
        service: Injected ModelService.

    Returns:
        Updated model version resource.
    """
    version = await service.transition_stage(
        version_id=version_id,
        model_id=model_id,
        tenant_id=tenant.tenant_id,
        new_stage=body.stage,
    )
    return VersionResponse.model_validate(version)


# ---------------------------------------------------------------------------
# Lineage endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/models/{model_id}/lineage",
    response_model=LineageResponse,
    summary="Get model lineage graph",
)
async def get_model_lineage(
    model_id: uuid.UUID,
    tenant=Depends(get_current_tenant),
    service: ModelService = Depends(_model_service),
) -> LineageResponse:
    """Return the fine-tuning lineage graph for a model.

    Shows all versions and their parent model relationships as a directed
    graph. Useful for understanding model provenance in fine-tuning chains.

    Args:
        model_id: Root model UUID path parameter.
        tenant: Authenticated tenant context.
        service: Injected ModelService.

    Returns:
        Lineage graph with version nodes and directed edges.
    """
    lineage = await service.get_lineage(model_id, tenant.tenant_id)
    return LineageResponse(**lineage)


# ---------------------------------------------------------------------------
# ML-BOM endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/models/{model_id}/versions/{version_id}/bom",
    response_model=dict,
    summary="Get CycloneDX ML-BOM",
)
async def get_ml_bom(
    model_id: uuid.UUID,
    version_id: uuid.UUID,
    tenant=Depends(get_current_tenant),
    service: ModelService = Depends(_model_service),
) -> dict:
    """Retrieve the CycloneDX ML Bill of Materials for a model version.

    If no BOM has been generated yet, one is produced on-the-fly and
    persisted for subsequent requests.

    Args:
        model_id: Parent model UUID path parameter.
        version_id: Version UUID path parameter.
        tenant: Authenticated tenant context.
        service: Injected ModelService.

    Returns:
        CycloneDX 1.5 ML-BOM JSON document.
    """
    return await service.get_ml_bom(model_id, version_id, tenant.tenant_id)


# ---------------------------------------------------------------------------
# Cost endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/models/{model_id}/versions/{version_id}/cost",
    response_model=CostBreakdownResponse,
    summary="Get cost attribution",
)
async def get_cost_breakdown(
    model_id: uuid.UUID,
    version_id: uuid.UUID,
    tenant=Depends(get_current_tenant),
    service: ModelService = Depends(_model_service),
) -> CostBreakdownResponse:
    """Return the full cost breakdown for a model version.

    Aggregates training cost (from version record), inference cost
    (accumulated from all deployments), and estimated monthly storage cost.

    Args:
        model_id: Parent model UUID path parameter.
        version_id: Version UUID path parameter.
        tenant: Authenticated tenant context.
        service: Injected ModelService.

    Returns:
        Cost breakdown with training, inference, storage, and total amounts.
    """
    breakdown = await service.get_cost_breakdown(model_id, version_id, tenant.tenant_id)
    return CostBreakdownResponse(
        model_version_id=breakdown.model_version_id,
        training_cost_usd=breakdown.training_cost_usd,
        inference_cost_usd=breakdown.inference_cost_usd,
        storage_cost_monthly_usd=breakdown.storage_cost_monthly_usd,
        total_cost_usd=breakdown.total_cost_usd,
    )


# ---------------------------------------------------------------------------
# Semantic search endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/models/search",
    response_model=ModelSearchResponse,
    summary="Semantic model search",
)
async def search_models(
    body: dict,
    tenant=Depends(get_current_tenant),
    service: ModelService = Depends(_model_service),
) -> ModelSearchResponse:
    """Search for models by name, description, or tags.

    Performs a case-insensitive full-text search across model name,
    description, and tags JSONB fields. Future versions will support
    pgvector semantic similarity search.

    Args:
        body: JSON body with 'query' and optional 'limit' fields.
        tenant: Authenticated tenant context.
        service: Injected ModelService.

    Returns:
        Search results with matching models.
    """
    query: str = body.get("query", "")
    limit: int = min(int(body.get("limit", 20)), 100)
    models = await service.search_models(
        tenant_id=tenant.tenant_id,
        query=query,
        limit=limit,
    )
    return ModelSearchResponse(
        items=[ModelResponse.model_validate(m) for m in models],
        query=query,
        total=len(models),
    )


# ---------------------------------------------------------------------------
# Deployment endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/models/{model_id}/versions/{version_id}/deployments",
    response_model=DeploymentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Deploy a model version",
)
async def deploy_model_version(
    model_id: uuid.UUID,
    version_id: uuid.UUID,
    body: DeploymentCreateRequest,
    tenant=Depends(get_current_tenant),
    service: ModelService = Depends(_model_service),
) -> DeploymentResponse:
    """Deploy a model version to a target environment.

    Creates a deployment record and publishes a model.deployed event.
    The version must be in the correct stage for the target environment.

    Args:
        model_id: Parent model UUID path parameter.
        version_id: Version UUID path parameter.
        body: Deployment creation request.
        tenant: Authenticated tenant context.
        service: Injected ModelService.

    Returns:
        Newly created deployment resource.
    """
    deployment = await service.deploy_version(
        model_id=model_id,
        version_id=version_id,
        tenant_id=tenant.tenant_id,
        environment=body.environment,
        endpoint_url=body.endpoint_url,
    )
    return DeploymentResponse.model_validate(deployment)


@router.post(
    "/deployments/{deployment_id}/rollback",
    response_model=DeploymentResponse,
    summary="Roll back a deployment",
)
async def rollback_deployment(
    deployment_id: uuid.UUID,
    body: RollbackRequest,
    tenant=Depends(get_current_tenant),
    service: ModelService = Depends(_model_service),
) -> DeploymentResponse:
    """Roll back an active deployment by marking it inactive.

    The caller is responsible for routing traffic away from the endpoint.
    A model.deployment_rolled_back event is published to Kafka.

    Args:
        deployment_id: Deployment UUID path parameter.
        body: Optional rollback reason for the audit log.
        tenant: Authenticated tenant context.
        service: Injected ModelService.

    Returns:
        Updated deployment resource with status='inactive'.
    """
    deployment = await service.rollback_deployment(deployment_id, tenant.tenant_id)
    return DeploymentResponse.model_validate(deployment)


# ---------------------------------------------------------------------------
# Experiment endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/experiments",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create experiment",
)
async def create_experiment(
    body: ExperimentCreateRequest,
    tenant=Depends(get_current_tenant),
    service: ExperimentService = Depends(_experiment_service),
) -> ExperimentResponse:
    """Create a new MLflow-compatible experiment.

    Args:
        body: Experiment creation request.
        tenant: Authenticated tenant context.
        service: Injected ExperimentService.

    Returns:
        Newly created experiment resource.
    """
    experiment = await service.create_experiment(
        tenant_id=tenant.tenant_id,
        name=body.name,
        description=body.description,
    )
    return ExperimentResponse.model_validate(experiment)


@router.get(
    "/experiments",
    response_model=list[ExperimentResponse],
    summary="List experiments",
)
async def list_experiments(
    tenant=Depends(get_current_tenant),
    service: ExperimentService = Depends(_experiment_service),
) -> list[ExperimentResponse]:
    """Return all experiments for the authenticated tenant.

    Args:
        tenant: Authenticated tenant context.
        service: Injected ExperimentService.

    Returns:
        List of experiment resources.
    """
    experiments = await service.list_experiments(tenant.tenant_id)
    return [ExperimentResponse.model_validate(e) for e in experiments]


@router.post(
    "/experiments/{experiment_id}/runs",
    response_model=RunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start experiment run",
)
async def start_run(
    experiment_id: uuid.UUID,
    body: RunCreateRequest,
    tenant=Depends(get_current_tenant),
    service: ExperimentService = Depends(_experiment_service),
) -> RunResponse:
    """Start a new training run within an experiment.

    Args:
        experiment_id: Parent experiment UUID path parameter.
        body: Run creation request with initial parameters.
        tenant: Authenticated tenant context.
        service: Injected ExperimentService.

    Returns:
        Newly created run resource in 'running' status.
    """
    run = await service.start_run(
        experiment_id=experiment_id,
        tenant_id=tenant.tenant_id,
        parameters=body.parameters,
    )
    return RunResponse.model_validate(run)


@router.patch(
    "/runs/{run_id}",
    response_model=RunResponse,
    summary="Log metrics/artifacts to run",
)
async def log_run(
    run_id: uuid.UUID,
    body: RunUpdateRequest,
    tenant=Depends(get_current_tenant),
    service: ExperimentService = Depends(_experiment_service),
) -> RunResponse:
    """Log metrics, artifacts, and/or terminal status to a run.

    Args:
        run_id: Run UUID path parameter.
        body: Run update payload with metrics, artifacts, or status.
        tenant: Authenticated tenant context (used for logging only).
        service: Injected ExperimentService.

    Returns:
        Updated run resource.
    """
    run = await service.log_run(
        run_id=run_id,
        metrics=body.metrics,
        artifacts=body.artifacts,
        status=body.status,
    )
    return RunResponse.model_validate(run)


@router.get(
    "/experiments/{experiment_id}/runs",
    response_model=list[RunResponse],
    summary="List experiment runs",
)
async def list_runs(
    experiment_id: uuid.UUID,
    tenant=Depends(get_current_tenant),
    service: ExperimentService = Depends(_experiment_service),
) -> list[RunResponse]:
    """Return all runs for an experiment.

    Args:
        experiment_id: Parent experiment UUID path parameter.
        tenant: Authenticated tenant context.
        service: Injected ExperimentService.

    Returns:
        List of run resources.
    """
    runs = await service.list_runs(experiment_id, tenant.tenant_id)
    return [RunResponse.model_validate(r) for r in runs]
