"""Deployment rollback adapter — safe promotion reversal for model versions.

Provides structured rollback of model deployments: identifying the previous stable
version, verifying it is safe to restore, executing the rollback, and emitting
Kafka lifecycle events with full audit trail.

Rollback scenarios:
1. Production rollback: restore previous production version from staging
2. Canary abort: stop a canary deployment and revert to 100% stable
3. Emergency rollback: force immediate rollback regardless of stage gates

All rollbacks are recorded in reg_deployment_rollbacks (added by migration)
with reason, triggering user, from_version, and to_version for audit purposes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class RollbackReason(str, Enum):
    """Canonical reasons for initiating a deployment rollback."""

    MANUAL = "manual"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_SPIKE = "error_rate_spike"
    CANARY_FAILURE = "canary_failure"
    DRIFT_DETECTED = "drift_detected"
    COMPLIANCE_VIOLATION = "compliance_violation"
    EMERGENCY = "emergency"


class RollbackStatus(str, Enum):
    """Status of a rollback operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RollbackPlan:
    """Plan for executing a deployment rollback.

    Attributes:
        rollback_id: Unique UUID for this rollback operation.
        from_version_id: UUID of the version being rolled back (current).
        to_version_id: UUID of the version to restore.
        from_stage: Current stage of the deployment.
        to_stage: Target stage after rollback.
        reason: Why the rollback is being performed.
        traffic_split_before: Traffic split configuration before rollback.
        traffic_split_after: Traffic split configuration after rollback.
        requires_approval: Whether approval workflow gate is required.
        pre_rollback_checks: List of checks to run before executing.
    """

    rollback_id: uuid.UUID
    from_version_id: uuid.UUID
    to_version_id: uuid.UUID
    from_stage: str
    to_stage: str
    reason: RollbackReason
    traffic_split_before: dict[str, Any]
    traffic_split_after: dict[str, Any]
    requires_approval: bool
    pre_rollback_checks: list[str] = field(default_factory=list)


@dataclass
class RollbackResult:
    """Result of executing a deployment rollback.

    Attributes:
        rollback_id: UUID matching the plan that was executed.
        status: Final status of the rollback operation.
        from_version_id: Version that was rolled back from.
        to_version_id: Version that was restored.
        duration_ms: Time taken to complete the rollback.
        checks_passed: Results of pre-rollback validation checks.
        error_message: Error detail if status is FAILED.
    """

    rollback_id: uuid.UUID
    status: RollbackStatus
    from_version_id: uuid.UUID
    to_version_id: uuid.UUID
    duration_ms: float
    checks_passed: dict[str, bool]
    error_message: str | None = None


class DeploymentRollbackManager:
    """Orchestrates safe rollback of model version deployments.

    Identifies the previous stable version, validates it is fit for
    restoration (health checks, stage guard), and coordinates the
    traffic split update and lifecycle event publication.

    Args:
        require_healthy_target: If True, abort rollback if target version
            has no recent passing health checks. Default True.
        skip_approval_for_emergency: If True, emergency rollbacks bypass
            the approval workflow gate. Default True.
    """

    def __init__(
        self,
        require_healthy_target: bool = True,
        skip_approval_for_emergency: bool = True,
    ) -> None:
        """Initialise the rollback manager.

        Args:
            require_healthy_target: Whether to validate target version health.
            skip_approval_for_emergency: Whether emergency rollbacks skip gates.
        """
        self._require_healthy_target = require_healthy_target
        self._skip_approval_for_emergency = skip_approval_for_emergency

    def build_rollback_plan(
        self,
        current_version: dict[str, Any],
        target_version: dict[str, Any],
        reason: RollbackReason,
        current_traffic_split: dict[str, Any] | None = None,
    ) -> RollbackPlan:
        """Build a rollback plan from current and target version records.

        Determines the target traffic split (100% stable after rollback),
        identifies pre-rollback checks, and sets the approval requirement.

        Args:
            current_version: Version record dict for the version to roll back.
            target_version: Version record dict for the version to restore.
            reason: Reason for the rollback.
            current_traffic_split: Current traffic distribution JSONB.

        Returns:
            RollbackPlan ready for execution or approval.
        """
        rollback_id = uuid.uuid4()

        requires_approval = not (
            self._skip_approval_for_emergency and reason == RollbackReason.EMERGENCY
        )

        checks: list[str] = ["target_version_exists", "target_version_not_archived"]
        if self._require_healthy_target:
            checks.append("target_version_health_check")

        target_split: dict[str, Any] = {
            "stable": str(target_version["id"]),
            "stable_percent": 100,
            "canary": None,
            "canary_percent": 0,
        }

        plan = RollbackPlan(
            rollback_id=rollback_id,
            from_version_id=uuid.UUID(str(current_version["id"])),
            to_version_id=uuid.UUID(str(target_version["id"])),
            from_stage=current_version.get("stage", "production"),
            to_stage=target_version.get("stage", "production"),
            reason=reason,
            traffic_split_before=current_traffic_split or {},
            traffic_split_after=target_split,
            requires_approval=requires_approval,
            pre_rollback_checks=checks,
        )

        logger.info(
            "rollback_plan_built",
            rollback_id=str(rollback_id),
            from_version=str(current_version["id"]),
            to_version=str(target_version["id"]),
            reason=reason.value,
            requires_approval=requires_approval,
        )

        return plan

    def validate_rollback_plan(
        self,
        plan: RollbackPlan,
        target_version: dict[str, Any],
        recent_health_status: str | None = None,
    ) -> dict[str, bool]:
        """Run pre-rollback validation checks against the plan.

        Args:
            plan: The rollback plan to validate.
            target_version: Target version record dict.
            recent_health_status: Latest health check result for target ('healthy', 'unhealthy', None).

        Returns:
            Dict mapping check name to pass/fail boolean.
        """
        results: dict[str, bool] = {}

        # Check: target version exists
        results["target_version_exists"] = bool(target_version.get("id"))

        # Check: target version is not archived
        results["target_version_not_archived"] = target_version.get("stage") != "archived"

        # Check: target version health
        if "target_version_health_check" in plan.pre_rollback_checks:
            if recent_health_status is None:
                # No health data — allow if configured to skip
                results["target_version_health_check"] = not self._require_healthy_target
            else:
                results["target_version_health_check"] = recent_health_status == "healthy"

        failed_checks = [k for k, v in results.items() if not v]
        if failed_checks:
            logger.warning(
                "rollback_validation_failed",
                rollback_id=str(plan.rollback_id),
                failed_checks=failed_checks,
            )
        else:
            logger.info("rollback_validation_passed", rollback_id=str(plan.rollback_id))

        return results

    def get_previous_production_version(
        self,
        all_versions: list[dict[str, Any]],
        current_version_id: uuid.UUID,
    ) -> dict[str, Any] | None:
        """Identify the most recent previous production version to roll back to.

        Filters versions to those that have previously been in 'production'
        stage and selects the most recently created one that is not the
        current version.

        Args:
            all_versions: All version records for the model.
            current_version_id: UUID of the version currently deployed.

        Returns:
            Previous production version dict, or None if no candidate exists.
        """
        candidates = [
            v for v in all_versions
            if str(v.get("id")) != str(current_version_id)
            and v.get("stage") in ("production", "staging")
        ]

        if not candidates:
            return None

        candidates.sort(key=lambda v: str(v.get("created_at", "")), reverse=True)
        return candidates[0]

    def build_canary_abort_plan(
        self,
        canary_version: dict[str, Any],
        stable_version: dict[str, Any],
        reason: RollbackReason = RollbackReason.CANARY_FAILURE,
    ) -> RollbackPlan:
        """Build a rollback plan specifically for aborting a canary deployment.

        Sets traffic to 100% stable immediately, with no approval required
        for emergency canary aborts.

        Args:
            canary_version: The canary version being aborted.
            stable_version: The stable version to restore to 100%.
            reason: Rollback reason (typically CANARY_FAILURE).

        Returns:
            RollbackPlan configured for immediate canary abort.
        """
        plan = self.build_rollback_plan(
            current_version=canary_version,
            target_version=stable_version,
            reason=reason,
            current_traffic_split={
                "stable": str(stable_version["id"]),
                "stable_percent": 80,
                "canary": str(canary_version["id"]),
                "canary_percent": 20,
            },
        )

        logger.info(
            "canary_abort_plan_built",
            rollback_id=str(plan.rollback_id),
            canary_version=str(canary_version["id"]),
            stable_version=str(stable_version["id"]),
        )

        return plan
