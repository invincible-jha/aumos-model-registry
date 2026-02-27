"""Tests for DecommissionWorkflowManager.

Covers:
  - initiate_decommission creates workflow in FLAGGED_FOR_REVIEW
  - Duplicate initiation raises ValueError (409 upstream)
  - State machine transitions: advance_to_migration_planning, start_traffic_drain,
    advance_traffic_drain (step + auto-transition), complete_validation, delete_model
  - abort_workflow from all states
  - Abort requires confirmation from TRAFFIC_DRAINING / VALIDATION
  - Traffic percentage reduces correctly per step
  - get_workflow_status returns current state
  - list_workflows filters by state
"""

from __future__ import annotations

import pytest

from aumos_model_registry.decommission.workflow_manager import (
    DecommissionState,
    DecommissionWorkflowManager,
    _InMemoryWorkflowRepository,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo() -> _InMemoryWorkflowRepository:
    """Return a fresh in-memory workflow repository."""
    return _InMemoryWorkflowRepository()


@pytest.fixture
def manager(repo: _InMemoryWorkflowRepository) -> DecommissionWorkflowManager:
    """Return a workflow manager with 3 drain steps for faster tests."""
    return DecommissionWorkflowManager(
        workflow_repo=repo,
        default_drain_days=3,
        drain_steps=3,
    )


@pytest.fixture
def sample_params() -> dict[str, str]:
    """Return common initiate parameters."""
    return {
        "model_id": "model-uuid-001",
        "model_name": "Test CTGAN Model",
        "tenant_id": "tenant-test",
        "trigger_reason": "Score exceeded auto-decommission threshold of 0.9",
        "initiated_by": "system",
    }


# ---------------------------------------------------------------------------
# initiate_decommission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initiate_creates_flagged_for_review(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """Initiating decommission creates a workflow in FLAGGED_FOR_REVIEW."""
    workflow = await manager.initiate_decommission(**sample_params)
    assert workflow.state == DecommissionState.FLAGGED_FOR_REVIEW
    assert workflow.workflow_id is not None
    assert workflow.model_id == sample_params["model_id"]
    assert workflow.tenant_id == sample_params["tenant_id"]


@pytest.mark.asyncio
async def test_initiate_sets_trigger_reason(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """trigger_reason and initiated_by are stored on the workflow."""
    workflow = await manager.initiate_decommission(**sample_params)
    assert workflow.trigger_reason == sample_params["trigger_reason"]
    assert workflow.initiated_by == sample_params["initiated_by"]


@pytest.mark.asyncio
async def test_initiate_traffic_starts_at_100(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """Traffic starts at 100% on initiation."""
    workflow = await manager.initiate_decommission(**sample_params)
    assert workflow.traffic_percentage == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_initiate_duplicate_raises_value_error(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """Initiating a second workflow for the same model raises ValueError."""
    await manager.initiate_decommission(**sample_params)
    with pytest.raises(ValueError, match="already has an active decommission workflow"):
        await manager.initiate_decommission(**sample_params)


@pytest.mark.asyncio
async def test_initiate_records_state_transition_history(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """A state transition from ACTIVE → FLAGGED_FOR_REVIEW is recorded in history."""
    workflow = await manager.initiate_decommission(**sample_params)
    assert len(workflow.state_history) == 1
    assert workflow.state_history[0]["from"] == "ACTIVE"
    assert workflow.state_history[0]["to"] == "FLAGGED_FOR_REVIEW"


# ---------------------------------------------------------------------------
# advance_to_migration_planning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advance_to_migration_planning(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """FLAGGED_FOR_REVIEW → MIGRATION_PLANNING transition succeeds."""
    workflow = await manager.initiate_decommission(**sample_params)
    updated = await manager.advance_to_migration_planning(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
        migration_notes="Planning migration to model-v2",
    )
    assert updated.state == DecommissionState.MIGRATION_PLANNING


@pytest.mark.asyncio
async def test_advance_to_migration_planning_wrong_state_raises(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """Transitioning from ACTIVE (not FLAGGED_FOR_REVIEW) raises ValueError."""
    # Create a workflow and manually put it in MIGRATION_PLANNING
    workflow = await manager.initiate_decommission(**sample_params)
    await manager.advance_to_migration_planning(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    # Try to advance again from MIGRATION_PLANNING → raises because expected FLAGGED
    with pytest.raises(ValueError, match="expected FLAGGED_FOR_REVIEW"):
        await manager.advance_to_migration_planning(
            workflow_id=workflow.workflow_id,
            tenant_id=sample_params["tenant_id"],
        )


# ---------------------------------------------------------------------------
# start_traffic_drain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_traffic_drain_sets_schedule(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """Starting traffic drain sets drain_start_at and drain_end_at."""
    workflow = await manager.initiate_decommission(**sample_params)
    await manager.advance_to_migration_planning(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    draining = await manager.start_traffic_drain(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    assert draining.state == DecommissionState.TRAFFIC_DRAINING
    assert draining.drain_start_at is not None
    assert draining.drain_end_at is not None
    assert draining.drain_end_at > draining.drain_start_at


# ---------------------------------------------------------------------------
# advance_traffic_drain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advance_traffic_drain_reduces_percentage(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """Each drain step reduces traffic_percentage by 100/drain_steps %."""
    workflow = await manager.initiate_decommission(**sample_params)
    await manager.advance_to_migration_planning(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    await manager.start_traffic_drain(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    # With drain_steps=3, each step reduces by 100/3 ≈ 33.33%
    step1 = await manager.advance_traffic_drain(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    expected_pct = 100.0 - (100.0 / 3)
    assert step1.traffic_percentage == pytest.approx(expected_pct, abs=0.01)
    assert step1.drain_steps_completed == 1


@pytest.mark.asyncio
async def test_advance_traffic_drain_auto_transitions_to_validation(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """After drain_steps steps, traffic reaches 0% and auto-transitions to VALIDATION."""
    workflow = await manager.initiate_decommission(**sample_params)
    await manager.advance_to_migration_planning(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    await manager.start_traffic_drain(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    # drain_steps=3 → 3 advance calls drain to 0%
    result = None
    for _ in range(3):
        result = await manager.advance_traffic_drain(
            workflow_id=workflow.workflow_id,
            tenant_id=sample_params["tenant_id"],
        )
    assert result is not None
    assert result.state == DecommissionState.VALIDATION
    assert result.traffic_percentage == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# complete_validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_validation_transitions_to_archived(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """VALIDATION → ARCHIVED transition succeeds and sets completed_at."""
    workflow = await manager.initiate_decommission(**sample_params)
    await manager.advance_to_migration_planning(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    await manager.start_traffic_drain(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    for _ in range(3):
        await manager.advance_traffic_drain(
            workflow_id=workflow.workflow_id,
            tenant_id=sample_params["tenant_id"],
        )
    archived = await manager.complete_validation(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
        validation_notes="All downstream traffic migrated. Checksums verified.",
    )
    assert archived.state == DecommissionState.ARCHIVED
    assert archived.completed_at is not None


# ---------------------------------------------------------------------------
# delete_model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_model_transitions_to_deleted(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """ARCHIVED → DELETED is the terminal transition."""
    workflow = await manager.initiate_decommission(**sample_params)
    await manager.advance_to_migration_planning(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    await manager.start_traffic_drain(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    for _ in range(3):
        await manager.advance_traffic_drain(
            workflow_id=workflow.workflow_id,
            tenant_id=sample_params["tenant_id"],
        )
    await manager.complete_validation(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    deleted = await manager.delete_model(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    assert deleted.state == DecommissionState.DELETED


# ---------------------------------------------------------------------------
# abort_workflow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abort_from_flagged_for_review(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """Abort from FLAGGED_FOR_REVIEW returns to ACTIVE without confirmation."""
    workflow = await manager.initiate_decommission(**sample_params)
    aborted = await manager.abort_workflow(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
        abort_reason="Signal was a false positive. Model is performing well.",
        confirmed=False,
    )
    assert aborted.state == DecommissionState.ACTIVE
    assert aborted.abort_reason == "Signal was a false positive. Model is performing well."
    assert aborted.aborted_at is not None
    assert aborted.traffic_percentage == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_abort_from_traffic_draining_requires_confirmed(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """Aborting from TRAFFIC_DRAINING without confirmed=True raises ValueError."""
    workflow = await manager.initiate_decommission(**sample_params)
    await manager.advance_to_migration_planning(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    await manager.start_traffic_drain(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    with pytest.raises(ValueError, match="requires confirmed=True"):
        await manager.abort_workflow(
            workflow_id=workflow.workflow_id,
            tenant_id=sample_params["tenant_id"],
            abort_reason="Reverting due to upstream dependency failure discovered.",
            confirmed=False,
        )


@pytest.mark.asyncio
async def test_abort_from_traffic_draining_with_confirmation(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """Aborting from TRAFFIC_DRAINING with confirmed=True succeeds."""
    workflow = await manager.initiate_decommission(**sample_params)
    await manager.advance_to_migration_planning(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    await manager.start_traffic_drain(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    aborted = await manager.abort_workflow(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
        abort_reason="Confirmed abort after traffic coordination with serving team.",
        confirmed=True,
    )
    assert aborted.state == DecommissionState.ACTIVE


@pytest.mark.asyncio
async def test_abort_from_archived_raises(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """Aborting from ARCHIVED state raises ValueError."""
    workflow = await manager.initiate_decommission(**sample_params)
    await manager.advance_to_migration_planning(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    await manager.start_traffic_drain(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    for _ in range(3):
        await manager.advance_traffic_drain(
            workflow_id=workflow.workflow_id,
            tenant_id=sample_params["tenant_id"],
        )
    await manager.complete_validation(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    with pytest.raises(ValueError, match="ARCHIVED"):
        await manager.abort_workflow(
            workflow_id=workflow.workflow_id,
            tenant_id=sample_params["tenant_id"],
            abort_reason="Attempting to undo archive.",
            confirmed=True,
        )


@pytest.mark.asyncio
async def test_abort_from_deleted_raises(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """Aborting from DELETED (terminal) state raises ValueError."""
    workflow = await manager.initiate_decommission(**sample_params)
    await manager.advance_to_migration_planning(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    await manager.start_traffic_drain(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    for _ in range(3):
        await manager.advance_traffic_drain(
            workflow_id=workflow.workflow_id,
            tenant_id=sample_params["tenant_id"],
        )
    await manager.complete_validation(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    await manager.delete_model(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    with pytest.raises(ValueError, match="DELETED"):
        await manager.abort_workflow(
            workflow_id=workflow.workflow_id,
            tenant_id=sample_params["tenant_id"],
            abort_reason="Too late to abort.",
            confirmed=True,
        )


# ---------------------------------------------------------------------------
# get_workflow_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_workflow_status_returns_current_state(
    manager: DecommissionWorkflowManager,
    sample_params: dict[str, str],
) -> None:
    """get_workflow_status returns the workflow with current state."""
    workflow = await manager.initiate_decommission(**sample_params)
    status = await manager.get_workflow_status(
        workflow_id=workflow.workflow_id,
        tenant_id=sample_params["tenant_id"],
    )
    assert status.workflow_id == workflow.workflow_id
    assert status.state == DecommissionState.FLAGGED_FOR_REVIEW


@pytest.mark.asyncio
async def test_get_workflow_status_not_found_raises(
    manager: DecommissionWorkflowManager,
) -> None:
    """get_workflow_status raises ValueError for unknown workflow IDs."""
    with pytest.raises(ValueError, match="No decommission workflow found"):
        await manager.get_workflow_status(
            workflow_id="nonexistent-id",
            tenant_id="tenant-x",
        )


# ---------------------------------------------------------------------------
# list_workflows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_workflows_returns_all_for_tenant(
    manager: DecommissionWorkflowManager,
) -> None:
    """list_workflows returns all workflows for the tenant."""
    for i in range(3):
        await manager.initiate_decommission(
            model_id=f"model-{i}",
            model_name=f"Model {i}",
            tenant_id="shared-tenant",
            trigger_reason="Automated scoring exceeded threshold for testing.",
            initiated_by="test",
        )
    workflows = await manager.list_workflows(tenant_id="shared-tenant")
    assert len(workflows) == 3


@pytest.mark.asyncio
async def test_list_workflows_filtered_by_state(
    manager: DecommissionWorkflowManager,
) -> None:
    """list_workflows with state filter returns only matching workflows."""
    for i in range(2):
        await manager.initiate_decommission(
            model_id=f"model-flagged-{i}",
            model_name=f"Flagged Model {i}",
            tenant_id="filter-tenant",
            trigger_reason="Automated scoring exceeded threshold for state filter test.",
            initiated_by="test",
        )
    workflows = await manager.list_workflows(
        tenant_id="filter-tenant",
        state=DecommissionState.FLAGGED_FOR_REVIEW,
    )
    assert len(workflows) == 2
    assert all(w.state == DecommissionState.FLAGGED_FOR_REVIEW for w in workflows)


@pytest.mark.asyncio
async def test_list_workflows_tenant_isolation(
    manager: DecommissionWorkflowManager,
) -> None:
    """list_workflows only returns workflows for the specified tenant."""
    await manager.initiate_decommission(
        model_id="model-a",
        model_name="Model A",
        tenant_id="tenant-alpha",
        trigger_reason="Automated threshold breach for isolation test.",
        initiated_by="test",
    )
    await manager.initiate_decommission(
        model_id="model-b",
        model_name="Model B",
        tenant_id="tenant-beta",
        trigger_reason="Automated threshold breach for isolation test.",
        initiated_by="test",
    )
    alpha_workflows = await manager.list_workflows(tenant_id="tenant-alpha")
    assert len(alpha_workflows) == 1
    assert alpha_workflows[0].tenant_id == "tenant-alpha"
