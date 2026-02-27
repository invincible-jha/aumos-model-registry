"""Pydantic schemas for the model decommission API.

All inputs and outputs are Pydantic v2 models. Never return raw dicts.

Endpoints served:
  GET  /api/v1/models/decommission/candidates
  POST /api/v1/models/{model_id}/decommission/initiate
  GET  /api/v1/models/{model_id}/decommission/workflow/{workflow_id}
  POST /api/v1/models/{model_id}/decommission/abort
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Signal and score schemas
# ---------------------------------------------------------------------------


class SignalDetail(BaseModel):
    """Contribution breakdown for a single decommission signal.

    Attributes:
        raw: Raw signal value (0.0–1.0).
        weight: Configured weight for this signal.
        weighted_contribution: raw * weight.
    """

    raw: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(..., ge=0.0, le=1.0)
    weighted_contribution: float = Field(..., ge=0.0, le=1.0)


class DecommissionCandidateResponse(BaseModel):
    """Decommission candidate summary for a model version.

    Attributes:
        model_id: UUID of the model version.
        model_name: Human-readable model name.
        tenant_id: Owning tenant.
        composite_score: Weighted composite decommission score (0.0–1.0).
        recommendation: HEALTHY | REVIEW | AUTO_DECOMMISSION.
        should_flag_for_review: True when score > review threshold.
        should_auto_decommission: True when score > auto-decommission threshold.
        primary_signal: Signal with the highest weighted contribution.
        signal_scores: Per-signal breakdown.
        evaluated_at: When the score was computed.
    """

    model_id: str
    model_name: str
    tenant_id: str
    composite_score: float
    recommendation: Literal["HEALTHY", "REVIEW", "AUTO_DECOMMISSION"]
    should_flag_for_review: bool
    should_auto_decommission: bool
    primary_signal: str
    signal_scores: dict[str, SignalDetail]
    evaluated_at: datetime


# ---------------------------------------------------------------------------
# Workflow schemas
# ---------------------------------------------------------------------------


class InitiateDecommissionRequest(BaseModel):
    """Request body to initiate a decommission workflow.

    Attributes:
        trigger_reason: Why decommissioning is being initiated.
        drain_days: Override the default traffic drain period in days.
        metadata: Optional context for auditing.
    """

    trigger_reason: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Human-readable reason for initiating decommissioning",
    )
    drain_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Number of days to drain traffic over (default 7)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional context for auditing (jira ticket, runbook URL, etc.)",
    )


class StateTransitionRecord(BaseModel):
    """A single state transition in the workflow history.

    Attributes:
        from_state: Previous state.
        to_state: State transitioned to.
        at: Timestamp of the transition.
        notes: Optional notes.
    """

    from_state: str
    to_state: str
    at: datetime
    notes: str = ""


class DecommissionWorkflowResponse(BaseModel):
    """Full decommission workflow status response.

    Attributes:
        workflow_id: Unique workflow identifier.
        model_id: Model version being decommissioned.
        model_name: Human-readable model name.
        tenant_id: Owning tenant.
        state: Current lifecycle state.
        trigger_reason: Why decommissioning was initiated.
        initiated_by: User or system that started the workflow.
        traffic_percentage: Current traffic percentage (100 = full, 0 = none).
        drain_start_at: When traffic drain began.
        drain_end_at: Scheduled end of traffic drain.
        drain_steps_completed: Number of drain steps completed.
        state_history: Chronological transition history.
        created_at: Workflow creation timestamp.
        updated_at: Last update timestamp.
        completed_at: When the workflow reached ARCHIVED/DELETED.
        aborted_at: When the workflow was aborted (if applicable).
        abort_reason: Reason provided at abort time.
    """

    workflow_id: str
    model_id: str
    model_name: str
    tenant_id: str
    state: Literal[
        "ACTIVE",
        "FLAGGED_FOR_REVIEW",
        "MIGRATION_PLANNING",
        "TRAFFIC_DRAINING",
        "VALIDATION",
        "ARCHIVED",
        "DELETED",
    ]
    trigger_reason: str
    initiated_by: str
    traffic_percentage: float
    drain_start_at: datetime | None
    drain_end_at: datetime | None
    drain_steps_completed: int
    state_history: list[dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    aborted_at: datetime | None
    abort_reason: str | None


class AbortDecommissionRequest(BaseModel):
    """Request body to abort a decommission workflow.

    Attributes:
        abort_reason: Human-readable reason for aborting.
        confirmed: Must be True when aborting from TRAFFIC_DRAINING or VALIDATION.
    """

    abort_reason: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Human-readable reason for aborting the decommission workflow",
    )
    confirmed: bool = Field(
        default=False,
        description=(
            "Must be True when aborting from TRAFFIC_DRAINING or VALIDATION states, "
            "as this requires coordination with serving infrastructure."
        ),
    )
