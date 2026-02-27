"""DecommissionSignalCollector — collects and scores five decommission signals.

The five weighted signals that determine whether a model should be decommissioned:

  Signal                    Weight  Description
  ────────────────────────  ──────  ──────────────────────────────────────────
  drift_severity            0.30    Model output drift vs baseline (0.0–1.0)
  cost_per_decision_trend   0.25    Rising cost trend over recent decisions
  traffic_decline_pct       0.20    Traffic drop vs 30-day rolling average
  newer_model_available     0.15    Whether a newer compatible model exists
  compliance_expiry         0.10    Days until compliance certification expires

Composite score thresholds:
  > 0.7  → flag for human review (FLAGGED_FOR_REVIEW)
  > 0.9  → auto-initiate decommission (triggers workflow)

Score range: 0.0 (fully healthy, no action needed) → 1.0 (immediate action required).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Signal weights — must sum to 1.0
_SIGNAL_WEIGHTS: dict[str, float] = {
    "drift_severity": 0.30,
    "cost_per_decision_trend": 0.25,
    "traffic_decline_pct": 0.20,
    "newer_model_available": 0.15,
    "compliance_expiry": 0.10,
}

# Score thresholds
REVIEW_THRESHOLD: float = 0.7
AUTO_DECOMMISSION_THRESHOLD: float = 0.9


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecommissionSignals:
    """Raw signal values for a model at a point in time.

    All values are normalized to 0.0–1.0 where 1.0 represents maximum urgency.

    Attributes:
        model_id: UUID of the model version being evaluated.
        model_name: Human-readable model name for logging.
        tenant_id: Owning tenant.
        drift_severity: Normalized drift score (0.0 = no drift, 1.0 = critical drift).
        cost_per_decision_trend: Normalized cost trend (0.0 = improving, 1.0 = sharply rising).
        traffic_decline_pct: Normalized traffic decline (0.0 = no decline, 1.0 = zero traffic).
        newer_model_available: 1.0 if a newer compatible model exists, else 0.0.
        compliance_expiry: Normalized compliance urgency (0.0 = far future, 1.0 = expired).
        evaluated_at: When these signals were collected.
        raw_metadata: Additional context for auditing and debugging.
    """

    model_id: str
    model_name: str
    tenant_id: str
    drift_severity: float
    cost_per_decision_trend: float
    traffic_decline_pct: float
    newer_model_available: float
    compliance_expiry: float
    evaluated_at: datetime
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecommissionScore:
    """Computed decommission score with recommendation for a model.

    Attributes:
        model_id: UUID of the evaluated model version.
        model_name: Human-readable name.
        tenant_id: Owning tenant.
        composite_score: Weighted sum of all signals (0.0–1.0).
        signal_scores: Individual signal scores and their weighted contributions.
        recommendation: HEALTHY | REVIEW | AUTO_DECOMMISSION.
        should_flag_for_review: True when composite_score > REVIEW_THRESHOLD.
        should_auto_decommission: True when composite_score > AUTO_DECOMMISSION_THRESHOLD.
        evaluated_at: When the score was computed.
        primary_signal: The signal with the highest weighted contribution.
    """

    model_id: str
    model_name: str
    tenant_id: str
    composite_score: float
    signal_scores: dict[str, dict[str, float]]
    recommendation: str
    should_flag_for_review: bool
    should_auto_decommission: bool
    evaluated_at: datetime
    primary_signal: str


# ---------------------------------------------------------------------------
# Signal source protocol
# ---------------------------------------------------------------------------


class IDecommissionSignalSource:
    """Protocol for fetching raw decommission signals for a model.

    Implementations fetch data from MLflow, Prometheus, the registry
    database, and compliance stores.
    """

    async def get_drift_severity(
        self,
        model_id: str,
        tenant_id: str,
    ) -> float:
        """Fetch the current drift severity for a model (0.0–1.0).

        Args:
            model_id: Model version UUID.
            tenant_id: Owning tenant.

        Returns:
            Normalized drift score. 0.0 = no drift, 1.0 = critical drift.
        """
        raise NotImplementedError

    async def get_cost_per_decision_trend(
        self,
        model_id: str,
        tenant_id: str,
        lookback_days: int = 30,
    ) -> float:
        """Compute cost-per-decision trend over the look-back window (0.0–1.0).

        Returns 0.0 when cost is stable or declining.
        Returns 1.0 when cost has more than doubled over the window.

        Args:
            model_id: Model version UUID.
            tenant_id: Owning tenant.
            lookback_days: Days to compute trend over.

        Returns:
            Normalized cost trend score.
        """
        raise NotImplementedError

    async def get_traffic_decline_pct(
        self,
        model_id: str,
        tenant_id: str,
        lookback_days: int = 30,
    ) -> float:
        """Compute normalized traffic decline (0.0–1.0).

        Returns 0.0 when traffic is stable or growing.
        Returns 1.0 when the model receives no traffic.

        Args:
            model_id: Model version UUID.
            tenant_id: Owning tenant.
            lookback_days: Days to compute decline over.

        Returns:
            Normalized traffic decline score.
        """
        raise NotImplementedError

    async def has_newer_model_available(
        self,
        model_id: str,
        tenant_id: str,
    ) -> bool:
        """Check whether a newer compatible model is registered and deployed.

        Args:
            model_id: Model version UUID.
            tenant_id: Owning tenant.

        Returns:
            True if a newer, superior model is available.
        """
        raise NotImplementedError

    async def get_compliance_expiry_score(
        self,
        model_id: str,
        tenant_id: str,
    ) -> float:
        """Compute compliance urgency based on certification expiry (0.0–1.0).

        Returns 0.0 when certification is more than 180 days away.
        Returns 1.0 when certification has expired.

        Args:
            model_id: Model version UUID.
            tenant_id: Owning tenant.

        Returns:
            Normalized compliance urgency score.
        """
        raise NotImplementedError


class _NoOpSignalSource(IDecommissionSignalSource):
    """No-operation signal source that returns zeroed scores.

    Used as a safe default in testing and environments without full
    signal infrastructure.
    """

    async def get_drift_severity(self, model_id: str, tenant_id: str) -> float:
        return 0.0

    async def get_cost_per_decision_trend(
        self, model_id: str, tenant_id: str, lookback_days: int = 30
    ) -> float:
        return 0.0

    async def get_traffic_decline_pct(
        self, model_id: str, tenant_id: str, lookback_days: int = 30
    ) -> float:
        return 0.0

    async def has_newer_model_available(self, model_id: str, tenant_id: str) -> bool:
        return False

    async def get_compliance_expiry_score(self, model_id: str, tenant_id: str) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# DecommissionSignalCollector
# ---------------------------------------------------------------------------


class DecommissionSignalCollector:
    """Collects and scores five decommission signals for AI models.

    Orchestrates signal collection from the configured source, applies
    weights, computes a composite score, and returns a recommendation.

    Signal weights:
        drift_severity          0.30
        cost_per_decision_trend 0.25
        traffic_decline_pct     0.20
        newer_model_available   0.15
        compliance_expiry       0.10

    Thresholds:
        composite > 0.7 → flag for review
        composite > 0.9 → auto-initiate decommission

    Args:
        signal_source: Data source for signal values.
        signal_weights: Override default signal weights (must sum to 1.0).
        review_threshold: Score above which models are flagged for review.
        auto_decommission_threshold: Score above which auto-decommission triggers.
    """

    def __init__(
        self,
        signal_source: IDecommissionSignalSource | None = None,
        signal_weights: dict[str, float] | None = None,
        review_threshold: float = REVIEW_THRESHOLD,
        auto_decommission_threshold: float = AUTO_DECOMMISSION_THRESHOLD,
    ) -> None:
        """Initialise the collector with signal source and thresholds."""
        self._source = signal_source or _NoOpSignalSource()
        self._weights = signal_weights or _SIGNAL_WEIGHTS
        self._review_threshold = review_threshold
        self._auto_threshold = auto_decommission_threshold

        # Validate weights sum to 1.0 (allow small floating-point tolerance)
        weight_sum = sum(self._weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Signal weights must sum to 1.0, got {weight_sum:.6f}. "
                f"Weights: {self._weights}"
            )

    async def score_model(
        self,
        model_id: str,
        model_name: str,
        tenant_id: str,
        lookback_days: int = 30,
    ) -> DecommissionScore:
        """Collect all five signals and compute a composite decommission score.

        Args:
            model_id: UUID of the model version to evaluate.
            model_name: Human-readable model name for logging.
            tenant_id: Owning tenant.
            lookback_days: Days of history for trend-based signals.

        Returns:
            DecommissionScore with composite score and recommendation.
        """
        now = datetime.now(timezone.utc)

        # Collect all five raw signals
        drift = await self._source.get_drift_severity(model_id, tenant_id)
        cost_trend = await self._source.get_cost_per_decision_trend(
            model_id, tenant_id, lookback_days
        )
        traffic_decline = await self._source.get_traffic_decline_pct(
            model_id, tenant_id, lookback_days
        )
        newer_available = await self._source.has_newer_model_available(model_id, tenant_id)
        compliance_urgency = await self._source.get_compliance_expiry_score(model_id, tenant_id)

        raw_signals: dict[str, float] = {
            "drift_severity": _clamp(drift),
            "cost_per_decision_trend": _clamp(cost_trend),
            "traffic_decline_pct": _clamp(traffic_decline),
            "newer_model_available": 1.0 if newer_available else 0.0,
            "compliance_expiry": _clamp(compliance_urgency),
        }

        # Compute weighted scores per signal
        signal_scores: dict[str, dict[str, float]] = {}
        composite_score = 0.0
        for signal_name, raw_value in raw_signals.items():
            weight = self._weights.get(signal_name, 0.0)
            weighted = raw_value * weight
            composite_score += weighted
            signal_scores[signal_name] = {
                "raw": raw_value,
                "weight": weight,
                "weighted_contribution": weighted,
            }

        composite_score = _clamp(composite_score)

        # Determine recommendation
        should_auto = composite_score > self._auto_threshold
        should_review = composite_score > self._review_threshold

        if should_auto:
            recommendation = "AUTO_DECOMMISSION"
        elif should_review:
            recommendation = "REVIEW"
        else:
            recommendation = "HEALTHY"

        # Find primary signal (highest weighted contribution)
        primary_signal = max(
            signal_scores.keys(),
            key=lambda s: signal_scores[s]["weighted_contribution"],
        )

        score = DecommissionScore(
            model_id=model_id,
            model_name=model_name,
            tenant_id=tenant_id,
            composite_score=composite_score,
            signal_scores=signal_scores,
            recommendation=recommendation,
            should_flag_for_review=should_review,
            should_auto_decommission=should_auto,
            evaluated_at=now,
            primary_signal=primary_signal,
        )

        logger.info(
            "decommission_score_computed",
            model_id=model_id,
            model_name=model_name,
            tenant_id=tenant_id,
            composite_score=composite_score,
            recommendation=recommendation,
            primary_signal=primary_signal,
        )

        return score

    async def score_models_batch(
        self,
        models: list[tuple[str, str, str]],
        lookback_days: int = 30,
    ) -> list[DecommissionScore]:
        """Score a batch of models and return all with score > 0.

        Args:
            models: List of (model_id, model_name, tenant_id) tuples.
            lookback_days: Days of history for trend signals.

        Returns:
            List of DecommissionScore objects sorted by composite_score descending.
        """
        scores: list[DecommissionScore] = []
        for model_id, model_name, tenant_id in models:
            score = await self.score_model(
                model_id=model_id,
                model_name=model_name,
                tenant_id=tenant_id,
                lookback_days=lookback_days,
            )
            scores.append(score)

        scores.sort(key=lambda s: s.composite_score, reverse=True)

        logger.info(
            "batch_decommission_scoring_complete",
            total_models=len(scores),
            review_candidates=sum(1 for s in scores if s.should_flag_for_review),
            auto_decommission_candidates=sum(1 for s in scores if s.should_auto_decommission),
        )
        return scores

    @property
    def weights(self) -> dict[str, float]:
        """Current signal weights (read-only)."""
        return dict(self._weights)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clamp(value: float) -> float:
    """Clamp a float to [0.0, 1.0].

    Args:
        value: Input float.

    Returns:
        Value clamped to [0.0, 1.0].
    """
    return max(0.0, min(1.0, value))


__all__ = [
    "DecommissionSignalCollector",
    "DecommissionSignals",
    "DecommissionScore",
    "IDecommissionSignalSource",
    "REVIEW_THRESHOLD",
    "AUTO_DECOMMISSION_THRESHOLD",
]
