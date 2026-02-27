"""DecommissionWorkflowManager — state machine for model decommission lifecycle.

Implements the full decommission lifecycle as a deterministic state machine:

  ACTIVE → FLAGGED_FOR_REVIEW → MIGRATION_PLANNING → TRAFFIC_DRAINING
         → VALIDATION → ARCHIVED → DELETED

Traffic drain: linearly reduces traffic from 100% to 0% over the configured
drain period (default 7 days). Progress is tracked per-step and persisted
via the workflow repository.

Abort: allowed from any state before ARCHIVED. Returns to ACTIVE.

Key invariants:
  - Only forward transitions are allowed (no rewind after ARCHIVED)
  - DELETED is terminal — no further transitions
  - Traffic drain percentage is always 0% ≤ x ≤ 100%
  - Abort from FLAGGED_FOR_REVIEW or MIGRATION_PLANNING → ACTIVE (safe)
  - Abort from TRAFFIC_DRAINING or VALIDATION requires human confirmation
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# State machine definition
# ---------------------------------------------------------------------------


class DecommissionState(str, Enum):
    """States in the model decommission lifecycle.

    Transitions (forward only, except ABORT which returns to ACTIVE):
        ACTIVE → FLAGGED_FOR_REVIEW
        FLAGGED_FOR_REVIEW → MIGRATION_PLANNING | ACTIVE (abort)
        MIGRATION_PLANNING → TRAFFIC_DRAINING | ACTIVE (abort)
        TRAFFIC_DRAINING → VALIDATION | ACTIVE (abort, requires confirmation)
        VALIDATION → ARCHIVED | ACTIVE (abort, requires confirmation)
        ARCHIVED → DELETED
        DELETED → (terminal)
    """

    ACTIVE = "ACTIVE"
    FLAGGED_FOR_REVIEW = "FLAGGED_FOR_REVIEW"
    MIGRATION_PLANNING = "MIGRATION_PLANNING"
    TRAFFIC_DRAINING = "TRAFFIC_DRAINING"
    VALIDATION = "VALIDATION"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"


# Valid forward transitions
_ALLOWED_TRANSITIONS: dict[DecommissionState, list[DecommissionState]] = {
    DecommissionState.ACTIVE: [DecommissionState.FLAGGED_FOR_REVIEW],
    DecommissionState.FLAGGED_FOR_REVIEW: [
        DecommissionState.MIGRATION_PLANNING,
        DecommissionState.ACTIVE,  # abort
    ],
    DecommissionState.MIGRATION_PLANNING: [
        DecommissionState.TRAFFIC_DRAINING,
        DecommissionState.ACTIVE,  # abort
    ],
    DecommissionState.TRAFFIC_DRAINING: [
        DecommissionState.VALIDATION,
        DecommissionState.ACTIVE,  # abort (requires confirmation)
    ],
    DecommissionState.VALIDATION: [
        DecommissionState.ARCHIVED,
        DecommissionState.ACTIVE,  # abort (requires confirmation)
    ],
    DecommissionState.ARCHIVED: [DecommissionState.DELETED],
    DecommissionState.DELETED: [],  # terminal
}

# States from which abort requires explicit human confirmation
_ABORT_CONFIRMATION_REQUIRED: set[DecommissionState] = {
    DecommissionState.TRAFFIC_DRAINING,
    DecommissionState.VALIDATION,
}


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass
class DecommissionWorkflow:
    """Mutable workflow record tracking the decommission lifecycle of a model.

    Attributes:
        workflow_id: Unique workflow instance identifier (UUID string).
        model_id: UUID of the model version being decommissioned.
        model_name: Human-readable model name for logging.
        tenant_id: Owning tenant.
        state: Current lifecycle state.
        trigger_reason: Why decommissioning was initiated.
        initiated_by: User or system that initiated the workflow.
        traffic_percentage: Current traffic percentage (100 = full, 0 = none).
        drain_start_at: When traffic drain began (None if not started).
        drain_end_at: Scheduled end of traffic drain.
        drain_step_hours: Hours between each traffic drain step.
        drain_steps_completed: Number of drain steps completed.
        validation_started_at: When validation phase started.
        state_history: Chronological list of state transitions.
        metadata: Arbitrary context for auditing.
        created_at: When the workflow was created.
        updated_at: When the workflow was last updated.
        completed_at: When the workflow reached ARCHIVED or DELETED (None if ongoing).
        aborted_at: When the workflow was aborted (None if not aborted).
        abort_reason: Reason provided when aborting.
    """

    workflow_id: str
    model_id: str
    model_name: str
    tenant_id: str
    state: DecommissionState
    trigger_reason: str
    initiated_by: str
    traffic_percentage: float = 100.0
    drain_start_at: datetime | None = None
    drain_end_at: datetime | None = None
    drain_step_hours: float = 24.0
    drain_steps_completed: int = 0
    validation_started_at: datetime | None = None
    state_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    aborted_at: datetime | None = None
    abort_reason: str | None = None

    def record_transition(
        self,
        from_state: DecommissionState,
        to_state: DecommissionState,
        notes: str = "",
    ) -> None:
        """Append a state transition record to history.

        Args:
            from_state: Previous state.
            to_state: New state.
            notes: Optional human-readable notes for the transition.
        """
        self.state_history.append(
            {
                "from": from_state.value,
                "to": to_state.value,
                "at": datetime.now(timezone.utc).isoformat(),
                "notes": notes,
            }
        )
        self.updated_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Workflow repository protocol
# ---------------------------------------------------------------------------


class IDecommissionWorkflowRepository:
    """Protocol for persisting decommission workflow records."""

    async def create(self, workflow: DecommissionWorkflow) -> DecommissionWorkflow:
        """Persist a new workflow record.

        Args:
            workflow: The new workflow to persist.

        Returns:
            The persisted workflow.
        """
        raise NotImplementedError

    async def save(self, workflow: DecommissionWorkflow) -> None:
        """Update an existing workflow record.

        Args:
            workflow: The updated workflow.
        """
        raise NotImplementedError

    async def get_by_id(
        self,
        workflow_id: str,
        tenant_id: str,
    ) -> DecommissionWorkflow | None:
        """Load a workflow by its ID.

        Args:
            workflow_id: Workflow UUID string.
            tenant_id: Owning tenant.

        Returns:
            DecommissionWorkflow if found, None otherwise.
        """
        raise NotImplementedError

    async def get_active_for_model(
        self,
        model_id: str,
        tenant_id: str,
    ) -> DecommissionWorkflow | None:
        """Load the active workflow for a model (if any).

        Args:
            model_id: Model version UUID.
            tenant_id: Owning tenant.

        Returns:
            Active DecommissionWorkflow if one exists.
        """
        raise NotImplementedError

    async def list_by_tenant(
        self,
        tenant_id: str,
        state: DecommissionState | None = None,
        limit: int = 50,
    ) -> list[DecommissionWorkflow]:
        """List workflows for a tenant, optionally filtered by state.

        Args:
            tenant_id: Owning tenant.
            state: Optional state filter.
            limit: Maximum results.

        Returns:
            List of DecommissionWorkflow objects.
        """
        raise NotImplementedError


class _InMemoryWorkflowRepository(IDecommissionWorkflowRepository):
    """In-memory workflow repository for testing."""

    def __init__(self) -> None:
        self._store: dict[str, DecommissionWorkflow] = {}

    async def create(self, workflow: DecommissionWorkflow) -> DecommissionWorkflow:
        self._store[workflow.workflow_id] = workflow
        return workflow

    async def save(self, workflow: DecommissionWorkflow) -> None:
        self._store[workflow.workflow_id] = workflow

    async def get_by_id(
        self,
        workflow_id: str,
        tenant_id: str,
    ) -> DecommissionWorkflow | None:
        wf = self._store.get(workflow_id)
        return wf if wf and wf.tenant_id == tenant_id else None

    async def get_active_for_model(
        self,
        model_id: str,
        tenant_id: str,
    ) -> DecommissionWorkflow | None:
        for wf in self._store.values():
            if (
                wf.model_id == model_id
                and wf.tenant_id == tenant_id
                and wf.state not in (DecommissionState.DELETED, DecommissionState.ACTIVE)
            ):
                return wf
        return None

    async def list_by_tenant(
        self,
        tenant_id: str,
        state: DecommissionState | None = None,
        limit: int = 50,
    ) -> list[DecommissionWorkflow]:
        results = [
            wf
            for wf in self._store.values()
            if wf.tenant_id == tenant_id and (state is None or wf.state == state)
        ]
        return results[:limit]


# ---------------------------------------------------------------------------
# DecommissionWorkflowManager
# ---------------------------------------------------------------------------


class DecommissionWorkflowManager:
    """Manages the end-to-end decommission lifecycle for an AI model.

    Implements a deterministic state machine with the following stages:
        ACTIVE → FLAGGED_FOR_REVIEW → MIGRATION_PLANNING
               → TRAFFIC_DRAINING → VALIDATION → ARCHIVED → DELETED

    Traffic drain reduces traffic linearly from 100% to 0% over a
    configurable period (default 7 days = 168 hours). Each call to
    advance_traffic_drain() reduces traffic by one step.

    Args:
        workflow_repo: Repository for persisting workflow records.
        default_drain_days: Default traffic drain period in days.
        drain_steps: Number of equal steps to drain traffic over.
    """

    def __init__(
        self,
        workflow_repo: IDecommissionWorkflowRepository | None = None,
        default_drain_days: int = 7,
        drain_steps: int = 7,
    ) -> None:
        """Initialise the workflow manager."""
        self._repo = workflow_repo or _InMemoryWorkflowRepository()
        self._default_drain_days = default_drain_days
        self._drain_steps = max(1, drain_steps)

    # ------------------------------------------------------------------
    # Initiate decommission
    # ------------------------------------------------------------------

    async def initiate_decommission(
        self,
        model_id: str,
        model_name: str,
        tenant_id: str,
        trigger_reason: str,
        initiated_by: str = "system",
        drain_days: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DecommissionWorkflow:
        """Start the decommission workflow for a model.

        Transitions the model from ACTIVE to FLAGGED_FOR_REVIEW. If an
        active workflow already exists for this model, raises ValueError.

        Args:
            model_id: UUID of the model version to decommission.
            model_name: Human-readable name for logging.
            tenant_id: Owning tenant.
            trigger_reason: Why decommissioning was initiated.
            initiated_by: User or system triggering the workflow.
            drain_days: Override the default traffic drain period.
            metadata: Optional extra context for auditing.

        Returns:
            New DecommissionWorkflow in FLAGGED_FOR_REVIEW state.

        Raises:
            ValueError: If an active workflow already exists for this model.
        """
        existing = await self._repo.get_active_for_model(model_id, tenant_id)
        if existing is not None:
            raise ValueError(
                f"Model '{model_id}' already has an active decommission workflow "
                f"(workflow_id={existing.workflow_id}, state={existing.state.value}). "
                "Abort the existing workflow before initiating a new one."
            )

        effective_drain_days = drain_days or self._default_drain_days
        drain_step_hours = (effective_drain_days * 24) / self._drain_steps

        workflow = DecommissionWorkflow(
            workflow_id=str(uuid.uuid4()),
            model_id=model_id,
            model_name=model_name,
            tenant_id=tenant_id,
            state=DecommissionState.FLAGGED_FOR_REVIEW,
            trigger_reason=trigger_reason,
            initiated_by=initiated_by,
            traffic_percentage=100.0,
            drain_step_hours=drain_step_hours,
            metadata=metadata or {},
        )
        workflow.record_transition(
            from_state=DecommissionState.ACTIVE,
            to_state=DecommissionState.FLAGGED_FOR_REVIEW,
            notes=f"Initiated by {initiated_by}: {trigger_reason}",
        )

        await self._repo.create(workflow)

        logger.info(
            "decommission_workflow_initiated",
            workflow_id=workflow.workflow_id,
            model_id=model_id,
            model_name=model_name,
            tenant_id=tenant_id,
            trigger_reason=trigger_reason,
            initiated_by=initiated_by,
        )
        return workflow

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    async def advance_to_migration_planning(
        self,
        workflow_id: str,
        tenant_id: str,
        migration_notes: str = "",
    ) -> DecommissionWorkflow:
        """Transition from FLAGGED_FOR_REVIEW to MIGRATION_PLANNING.

        Args:
            workflow_id: Workflow UUID string.
            tenant_id: Owning tenant.
            migration_notes: Optional notes on the migration plan.

        Returns:
            Updated workflow in MIGRATION_PLANNING state.
        """
        return await self._transition(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            expected_state=DecommissionState.FLAGGED_FOR_REVIEW,
            target_state=DecommissionState.MIGRATION_PLANNING,
            notes=migration_notes,
        )

    async def start_traffic_drain(
        self,
        workflow_id: str,
        tenant_id: str,
        notes: str = "",
    ) -> DecommissionWorkflow:
        """Transition from MIGRATION_PLANNING to TRAFFIC_DRAINING.

        Records the drain start timestamp and sets the scheduled end time.

        Args:
            workflow_id: Workflow UUID string.
            tenant_id: Owning tenant.
            notes: Optional notes on why drain is starting now.

        Returns:
            Updated workflow in TRAFFIC_DRAINING state with drain schedule.
        """
        workflow = await self._load_workflow(workflow_id, tenant_id)
        self._assert_state(workflow, DecommissionState.MIGRATION_PLANNING)

        now = datetime.now(timezone.utc)
        drain_period_hours = workflow.drain_step_hours * self._drain_steps
        workflow.drain_start_at = now
        workflow.drain_end_at = now + timedelta(hours=drain_period_hours)

        return await self._apply_transition(
            workflow=workflow,
            from_state=DecommissionState.MIGRATION_PLANNING,
            to_state=DecommissionState.TRAFFIC_DRAINING,
            notes=notes,
        )

    async def advance_traffic_drain(
        self,
        workflow_id: str,
        tenant_id: str,
    ) -> DecommissionWorkflow:
        """Advance traffic drain by one step (reduces traffic by 100/drain_steps %).

        Called on a schedule (e.g., once per drain_step_hours). When traffic
        reaches 0%, automatically transitions to VALIDATION state.

        Args:
            workflow_id: Workflow UUID string.
            tenant_id: Owning tenant.

        Returns:
            Updated workflow with reduced traffic percentage.
        """
        workflow = await self._load_workflow(workflow_id, tenant_id)
        self._assert_state(workflow, DecommissionState.TRAFFIC_DRAINING)

        step_size = 100.0 / self._drain_steps
        workflow.traffic_percentage = max(0.0, workflow.traffic_percentage - step_size)
        workflow.drain_steps_completed += 1
        workflow.updated_at = datetime.now(timezone.utc)

        logger.info(
            "traffic_drain_step",
            workflow_id=workflow_id,
            model_id=workflow.model_id,
            traffic_percentage=workflow.traffic_percentage,
            steps_completed=workflow.drain_steps_completed,
            steps_total=self._drain_steps,
        )

        # Auto-transition to VALIDATION when traffic reaches 0%
        if workflow.traffic_percentage <= 0.0:
            return await self._apply_transition(
                workflow=workflow,
                from_state=DecommissionState.TRAFFIC_DRAINING,
                to_state=DecommissionState.VALIDATION,
                notes="Traffic fully drained. Beginning validation.",
            )

        await self._repo.save(workflow)
        return workflow

    async def complete_validation(
        self,
        workflow_id: str,
        tenant_id: str,
        validation_notes: str = "",
    ) -> DecommissionWorkflow:
        """Transition from VALIDATION to ARCHIVED.

        Args:
            workflow_id: Workflow UUID string.
            tenant_id: Owning tenant.
            validation_notes: Summary of what was validated.

        Returns:
            Updated workflow in ARCHIVED state.
        """
        workflow = await self._transition(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            expected_state=DecommissionState.VALIDATION,
            target_state=DecommissionState.ARCHIVED,
            notes=validation_notes,
        )
        workflow.completed_at = datetime.now(timezone.utc)
        await self._repo.save(workflow)
        return workflow

    async def delete_model(
        self,
        workflow_id: str,
        tenant_id: str,
        deletion_notes: str = "",
    ) -> DecommissionWorkflow:
        """Transition from ARCHIVED to DELETED (terminal state).

        Args:
            workflow_id: Workflow UUID string.
            tenant_id: Owning tenant.
            deletion_notes: Notes on what was deleted.

        Returns:
            Updated workflow in DELETED state.
        """
        return await self._transition(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            expected_state=DecommissionState.ARCHIVED,
            target_state=DecommissionState.DELETED,
            notes=deletion_notes,
        )

    # ------------------------------------------------------------------
    # Abort workflow
    # ------------------------------------------------------------------

    async def abort_workflow(
        self,
        workflow_id: str,
        tenant_id: str,
        abort_reason: str,
        confirmed: bool = False,
    ) -> DecommissionWorkflow:
        """Abort a decommission workflow and return the model to ACTIVE.

        Aborting from TRAFFIC_DRAINING or VALIDATION requires confirmed=True
        because it means restoring traffic, which requires coordination.

        Args:
            workflow_id: Workflow UUID string.
            tenant_id: Owning tenant.
            abort_reason: Human-readable reason for aborting.
            confirmed: Must be True when aborting from TRAFFIC_DRAINING or VALIDATION.

        Returns:
            Updated workflow in ACTIVE state.

        Raises:
            ValueError: If the workflow is in a terminal or non-abortable state,
                or if confirmed=False when required.
        """
        workflow = await self._load_workflow(workflow_id, tenant_id)

        if workflow.state == DecommissionState.DELETED:
            raise ValueError(
                f"Cannot abort workflow '{workflow_id}': "
                "model is in DELETED state (terminal)."
            )
        if workflow.state == DecommissionState.ARCHIVED:
            raise ValueError(
                f"Cannot abort workflow '{workflow_id}': "
                "model is ARCHIVED. Delete confirmation is required to proceed."
            )

        if workflow.state in _ABORT_CONFIRMATION_REQUIRED and not confirmed:
            raise ValueError(
                f"Aborting from {workflow.state.value} requires confirmed=True. "
                "This will restore traffic to the model and requires coordination "
                "with the serving infrastructure."
            )

        previous_state = workflow.state
        workflow.state = DecommissionState.ACTIVE
        workflow.traffic_percentage = 100.0
        workflow.aborted_at = datetime.now(timezone.utc)
        workflow.abort_reason = abort_reason
        workflow.record_transition(
            from_state=previous_state,
            to_state=DecommissionState.ACTIVE,
            notes=f"ABORTED: {abort_reason}",
        )

        await self._repo.save(workflow)

        logger.warning(
            "decommission_workflow_aborted",
            workflow_id=workflow_id,
            model_id=workflow.model_id,
            previous_state=previous_state.value,
            abort_reason=abort_reason,
            confirmed=confirmed,
        )
        return workflow

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_workflow_status(
        self,
        workflow_id: str,
        tenant_id: str,
    ) -> DecommissionWorkflow:
        """Load and return the current workflow status.

        Args:
            workflow_id: Workflow UUID string.
            tenant_id: Owning tenant.

        Returns:
            Current DecommissionWorkflow.

        Raises:
            ValueError: If no workflow with this ID exists for the tenant.
        """
        return await self._load_workflow(workflow_id, tenant_id)

    async def list_workflows(
        self,
        tenant_id: str,
        state: DecommissionState | None = None,
        limit: int = 50,
    ) -> list[DecommissionWorkflow]:
        """List decommission workflows for a tenant.

        Args:
            tenant_id: Owning tenant.
            state: Optional state filter.
            limit: Maximum results.

        Returns:
            List of DecommissionWorkflow objects.
        """
        return await self._repo.list_by_tenant(
            tenant_id=tenant_id,
            state=state,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _load_workflow(
        self,
        workflow_id: str,
        tenant_id: str,
    ) -> DecommissionWorkflow:
        """Load a workflow or raise ValueError if not found.

        Args:
            workflow_id: Workflow UUID string.
            tenant_id: Owning tenant.

        Returns:
            DecommissionWorkflow.

        Raises:
            ValueError: If the workflow is not found.
        """
        workflow = await self._repo.get_by_id(workflow_id, tenant_id)
        if workflow is None:
            raise ValueError(
                f"No decommission workflow found with id='{workflow_id}' "
                f"for tenant='{tenant_id}'."
            )
        return workflow

    def _assert_state(
        self,
        workflow: DecommissionWorkflow,
        expected: DecommissionState,
    ) -> None:
        """Assert a workflow is in the expected state.

        Args:
            workflow: The workflow to check.
            expected: The expected state.

        Raises:
            ValueError: If the workflow is not in the expected state.
        """
        if workflow.state != expected:
            raise ValueError(
                f"Workflow '{workflow.workflow_id}' is in state {workflow.state.value}, "
                f"expected {expected.value}."
            )

    async def _transition(
        self,
        workflow_id: str,
        tenant_id: str,
        expected_state: DecommissionState,
        target_state: DecommissionState,
        notes: str = "",
    ) -> DecommissionWorkflow:
        """Load, validate, and apply a state transition.

        Args:
            workflow_id: Workflow UUID.
            tenant_id: Owning tenant.
            expected_state: Required current state.
            target_state: State to transition to.
            notes: Optional transition notes.

        Returns:
            Updated workflow.
        """
        workflow = await self._load_workflow(workflow_id, tenant_id)
        self._assert_state(workflow, expected_state)
        return await self._apply_transition(workflow, expected_state, target_state, notes)

    async def _apply_transition(
        self,
        workflow: DecommissionWorkflow,
        from_state: DecommissionState,
        to_state: DecommissionState,
        notes: str = "",
    ) -> DecommissionWorkflow:
        """Apply a state transition and persist.

        Args:
            workflow: The workflow to update.
            from_state: Previous state (for validation).
            to_state: Target state.
            notes: Optional notes.

        Returns:
            Updated workflow.

        Raises:
            ValueError: If the transition is not allowed.
        """
        allowed = _ALLOWED_TRANSITIONS.get(from_state, [])
        if to_state not in allowed:
            raise ValueError(
                f"Transition {from_state.value} → {to_state.value} is not allowed. "
                f"Valid transitions from {from_state.value}: "
                f"{[s.value for s in allowed]}"
            )

        workflow.state = to_state
        workflow.record_transition(from_state, to_state, notes)
        await self._repo.save(workflow)

        logger.info(
            "decommission_state_transition",
            workflow_id=workflow.workflow_id,
            model_id=workflow.model_id,
            from_state=from_state.value,
            to_state=to_state.value,
        )
        return workflow


__all__ = [
    "DecommissionWorkflowManager",
    "DecommissionWorkflow",
    "DecommissionState",
    "IDecommissionWorkflowRepository",
]
