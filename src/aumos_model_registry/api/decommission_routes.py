"""FastAPI routes for the model decommission API.

All routes are thin: validate inputs via Pydantic schemas, delegate to
DecommissionSignalCollector or DecommissionWorkflowManager, and return
Pydantic response schemas. No domain logic lives here.

Endpoints:
    GET  /models/decommission/candidates
    POST /models/{model_id}/decommission/initiate
    GET  /models/{model_id}/decommission/workflow/{workflow_id}
    POST /models/{model_id}/decommission/abort
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from aumos_common.auth import get_current_tenant, get_current_user
from aumos_common.observability import get_logger

from aumos_model_registry.api.decommission_schemas import (
    AbortDecommissionRequest,
    DecommissionCandidateResponse,
    DecommissionWorkflowResponse,
    InitiateDecommissionRequest,
    SignalDetail,
)
from aumos_model_registry.decommission.signal_collector import (
    AUTO_DECOMMISSION_THRESHOLD,
    REVIEW_THRESHOLD,
    DecommissionScore,
    DecommissionSignalCollector,
)
from aumos_model_registry.decommission.workflow_manager import (
    DecommissionWorkflow,
    DecommissionWorkflowManager,
    _InMemoryWorkflowRepository,
)

logger = get_logger(__name__)

decommission_router = APIRouter(tags=["Model Decommissioning"])

# ---------------------------------------------------------------------------
# Module-level singletons (replace with DI container in production)
# ---------------------------------------------------------------------------

_signal_collector = DecommissionSignalCollector()
_workflow_manager = DecommissionWorkflowManager(
    workflow_repo=_InMemoryWorkflowRepository(),
)


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_signal_collector() -> DecommissionSignalCollector:
    """Return the configured DecommissionSignalCollector.

    Returns:
        Configured DecommissionSignalCollector instance.
    """
    return _signal_collector


def _get_workflow_manager() -> DecommissionWorkflowManager:
    """Return the configured DecommissionWorkflowManager.

    Returns:
        Configured DecommissionWorkflowManager instance.
    """
    return _workflow_manager


# ---------------------------------------------------------------------------
# Helpers: domain → schema conversion
# ---------------------------------------------------------------------------


def _score_to_candidate_response(score: DecommissionScore) -> DecommissionCandidateResponse:
    """Convert a DecommissionScore domain object to a candidate response schema.

    Args:
        score: The computed DecommissionScore.

    Returns:
        DecommissionCandidateResponse Pydantic model.
    """
    signal_details = {
        name: SignalDetail(
            raw=values["raw"],
            weight=values["weight"],
            weighted_contribution=values["weighted_contribution"],
        )
        for name, values in score.signal_scores.items()
    }
    return DecommissionCandidateResponse(
        model_id=score.model_id,
        model_name=score.model_name,
        tenant_id=score.tenant_id,
        composite_score=score.composite_score,
        recommendation=score.recommendation,  # type: ignore[arg-type]
        should_flag_for_review=score.should_flag_for_review,
        should_auto_decommission=score.should_auto_decommission,
        primary_signal=score.primary_signal,
        signal_scores=signal_details,
        evaluated_at=score.evaluated_at,
    )


def _workflow_to_response(workflow: DecommissionWorkflow) -> DecommissionWorkflowResponse:
    """Convert a DecommissionWorkflow domain object to a workflow response schema.

    Args:
        workflow: The DecommissionWorkflow to convert.

    Returns:
        DecommissionWorkflowResponse Pydantic model.
    """
    return DecommissionWorkflowResponse(
        workflow_id=workflow.workflow_id,
        model_id=workflow.model_id,
        model_name=workflow.model_name,
        tenant_id=workflow.tenant_id,
        state=workflow.state.value,  # type: ignore[arg-type]
        trigger_reason=workflow.trigger_reason,
        initiated_by=workflow.initiated_by,
        traffic_percentage=workflow.traffic_percentage,
        drain_start_at=workflow.drain_start_at,
        drain_end_at=workflow.drain_end_at,
        drain_steps_completed=workflow.drain_steps_completed,
        state_history=workflow.state_history,
        created_at=workflow.created_at,
        updated_at=workflow.updated_at,
        completed_at=workflow.completed_at,
        aborted_at=workflow.aborted_at,
        abort_reason=workflow.abort_reason,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@decommission_router.get(
    "/models/decommission/candidates",
    response_model=list[DecommissionCandidateResponse],
    summary="List decommission candidates",
)
async def list_decommission_candidates(
    model_ids: list[str] = Query(
        default=[],
        description="Comma-separated list of model_id:model_name:tenant_id tuples to score. "
        "Format: 'uuid|name|tenant'. If empty, returns an empty list.",
    ),
    min_score: float = Query(
        default=REVIEW_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Only return models with composite_score >= min_score (default: review threshold 0.7)",
    ),
    tenant=Depends(get_current_tenant),
    collector: DecommissionSignalCollector = Depends(_get_signal_collector),
) -> list[DecommissionCandidateResponse]:
    """Score a batch of model versions and return decommission candidates.

    For each model in the request, collects five weighted signals (drift, cost trend,
    traffic decline, newer model availability, compliance expiry) and computes a
    composite decommission score. Returns only models at or above min_score.

    Args:
        model_ids: List of 'uuid|name|tenant' strings identifying models to score.
        min_score: Minimum composite score threshold for inclusion (default: 0.7).
        tenant: Authenticated tenant context.
        collector: Injected DecommissionSignalCollector.

    Returns:
        List of DecommissionCandidateResponse objects sorted by composite_score desc.
    """
    if not model_ids:
        return []

    # Parse 'uuid|name|tenant_id' tuples from query params
    models_to_score: list[tuple[str, str, str]] = []
    for entry in model_ids:
        parts = entry.split("|")
        if len(parts) != 3:  # noqa: PLR2004
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid model_ids entry '{entry}'. Expected format: 'uuid|name|tenant_id'.",
            )
        model_id, model_name, tenant_id = parts
        # Enforce tenant isolation: only score models belonging to the caller's tenant
        if tenant_id != tenant.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Model '{model_id}' belongs to tenant '{tenant_id}', "
                f"not the authenticated tenant '{tenant.tenant_id}'.",
            )
        models_to_score.append((model_id, model_name, tenant_id))

    scores = await collector.score_models_batch(models_to_score)

    candidates = [
        _score_to_candidate_response(score)
        for score in scores
        if score.composite_score >= min_score
    ]

    logger.info(
        "decommission_candidates_listed",
        tenant_id=tenant.tenant_id,
        total_scored=len(scores),
        candidates_returned=len(candidates),
        min_score=min_score,
    )
    return candidates


@decommission_router.post(
    "/models/{model_id}/decommission/initiate",
    response_model=DecommissionWorkflowResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Initiate decommission workflow",
)
async def initiate_decommission(
    model_id: str,
    body: InitiateDecommissionRequest,
    tenant=Depends(get_current_tenant),
    user=Depends(get_current_user),
    manager: DecommissionWorkflowManager = Depends(_get_workflow_manager),
) -> DecommissionWorkflowResponse:
    """Start the decommission workflow for a model version.

    Transitions the model from ACTIVE to FLAGGED_FOR_REVIEW. Raises 409 if an
    active decommission workflow already exists for this model.

    Args:
        model_id: Model version UUID path parameter.
        body: Initiate decommission request payload.
        tenant: Authenticated tenant context.
        user: Authenticated user context.
        manager: Injected DecommissionWorkflowManager.

    Returns:
        Newly created DecommissionWorkflowResponse in FLAGGED_FOR_REVIEW state.
    """
    try:
        workflow = await manager.initiate_decommission(
            model_id=model_id,
            model_name=model_id,  # Use model_id as name stub until registry lookup is wired
            tenant_id=tenant.tenant_id,
            trigger_reason=body.trigger_reason,
            initiated_by=user.user_id,
            drain_days=body.drain_days,
            metadata=body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    logger.info(
        "decommission_initiated_via_api",
        model_id=model_id,
        workflow_id=workflow.workflow_id,
        tenant_id=tenant.tenant_id,
        initiated_by=user.user_id,
    )
    return _workflow_to_response(workflow)


@decommission_router.get(
    "/models/{model_id}/decommission/workflow/{workflow_id}",
    response_model=DecommissionWorkflowResponse,
    summary="Get decommission workflow status",
)
async def get_decommission_workflow(
    model_id: str,
    workflow_id: str,
    tenant=Depends(get_current_tenant),
    manager: DecommissionWorkflowManager = Depends(_get_workflow_manager),
) -> DecommissionWorkflowResponse:
    """Return the current status of a decommission workflow.

    Args:
        model_id: Model version UUID path parameter (used for audit logging).
        workflow_id: Decommission workflow UUID path parameter.
        tenant: Authenticated tenant context.
        manager: Injected DecommissionWorkflowManager.

    Returns:
        DecommissionWorkflowResponse with current state and history.
    """
    try:
        workflow = await manager.get_workflow_status(
            workflow_id=workflow_id,
            tenant_id=tenant.tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    # Validate that this workflow belongs to the model in the URL path
    if workflow.model_id != model_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' does not belong to model '{model_id}'.",
        )

    return _workflow_to_response(workflow)


@decommission_router.post(
    "/models/{model_id}/decommission/abort",
    response_model=DecommissionWorkflowResponse,
    summary="Abort decommission workflow",
)
async def abort_decommission(
    model_id: str,
    body: AbortDecommissionRequest,
    tenant=Depends(get_current_tenant),
    manager: DecommissionWorkflowManager = Depends(_get_workflow_manager),
) -> DecommissionWorkflowResponse:
    """Abort the active decommission workflow for a model and return it to ACTIVE.

    Aborting from TRAFFIC_DRAINING or VALIDATION requires confirmed=True in the
    request body, as this restores traffic and requires infrastructure coordination.

    Args:
        model_id: Model version UUID path parameter.
        body: Abort request with reason and optional confirmation flag.
        tenant: Authenticated tenant context.
        manager: Injected DecommissionWorkflowManager.

    Returns:
        Updated DecommissionWorkflowResponse in ACTIVE state.
    """
    # Find the active workflow for this model
    active_workflow = await manager._repo.get_active_for_model(  # noqa: SLF001
        model_id=model_id,
        tenant_id=tenant.tenant_id,
    )
    if active_workflow is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active decommission workflow found for model '{model_id}'.",
        )

    try:
        workflow = await manager.abort_workflow(
            workflow_id=active_workflow.workflow_id,
            tenant_id=tenant.tenant_id,
            abort_reason=body.abort_reason,
            confirmed=body.confirmed,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    logger.warning(
        "decommission_aborted_via_api",
        model_id=model_id,
        workflow_id=workflow.workflow_id,
        tenant_id=tenant.tenant_id,
        abort_reason=body.abort_reason,
    )
    return _workflow_to_response(workflow)


__all__ = ["decommission_router"]
