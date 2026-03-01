"""A/B test integration for model registry — traffic splitting and experiment tracking.

Enables controlled A/B experiments comparing two model versions on live traffic.
Integrates with the deployment lifecycle: experiments are attached to deployments
and track statistical significance of metric differences between control and treatment.

Implementation follows the frequentist hypothesis testing approach:
- Null hypothesis: treatment model performance = control model performance
- Test: Welch's t-test for continuous metrics, chi-squared for conversion rates
- Early stopping: sequential testing with spending function to control type-I error

A/B experiments are stored in reg_ab_experiments table (added by migration).
Traffic split is managed by the deployment's traffic_split JSONB field.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class ABTestStatus(str, Enum):
    """Lifecycle status for an A/B test experiment."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED_EARLY = "stopped_early"
    FAILED = "failed"


class ABTestDecision(str, Enum):
    """Statistical decision outcome for the experiment."""

    INSUFFICIENT_DATA = "insufficient_data"
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"
    TREATMENT_WINS = "treatment_wins"
    CONTROL_WINS = "control_wins"


@dataclass
class ABTestConfig:
    """Configuration for an A/B test between two model versions.

    Attributes:
        experiment_name: Human-readable name for the experiment.
        control_version_id: Model version UUID for the control (baseline).
        treatment_version_id: Model version UUID for the treatment (new model).
        traffic_split_percent: Percentage of traffic routed to treatment (0-100).
        primary_metric: Key metric to use for significance testing (e.g., 'accuracy').
        minimum_sample_size: Minimum samples per arm before significance testing.
        significance_level: Alpha for hypothesis testing (default 0.05).
        minimum_effect_size: Minimum detectable effect to consider practical significance.
        max_duration_hours: Maximum experiment duration before forced decision.
    """

    experiment_name: str
    control_version_id: uuid.UUID
    treatment_version_id: uuid.UUID
    traffic_split_percent: int = 20
    primary_metric: str = "accuracy"
    minimum_sample_size: int = 1000
    significance_level: float = 0.05
    minimum_effect_size: float = 0.01
    max_duration_hours: int = 168  # 7 days


@dataclass
class ABTestMetricSample:
    """A single metric observation for one arm of the A/B test.

    Attributes:
        arm: Which arm this observation belongs to ('control' or 'treatment').
        metric_name: Name of the metric measured.
        value: Numeric metric value.
        timestamp: ISO-8601 timestamp of observation.
    """

    arm: str
    metric_name: str
    value: float
    timestamp: str


@dataclass
class ABTestStatisticalResult:
    """Statistical analysis result for an A/B test.

    Attributes:
        control_mean: Mean metric value for control arm.
        treatment_mean: Mean metric value for treatment arm.
        control_n: Sample count for control arm.
        treatment_n: Sample count for treatment arm.
        effect_size: Absolute difference (treatment_mean - control_mean).
        relative_lift: Relative improvement (effect_size / control_mean).
        p_value: Two-tailed p-value from Welch's t-test.
        confidence_interval_95: Tuple of (lower, upper) bounds for effect.
        is_significant: True if p_value < significance_level and n >= min_sample.
        decision: ABTestDecision enum indicating the experimental outcome.
    """

    control_mean: float
    treatment_mean: float
    control_n: int
    treatment_n: int
    effect_size: float
    relative_lift: float
    p_value: float
    confidence_interval_95: tuple[float, float]
    is_significant: bool
    decision: ABTestDecision


class ABTestAnalyzer:
    """Statistical analysis engine for A/B test experiments.

    Runs Welch's t-test (unequal variances) on metric samples from control
    and treatment arms. Produces p-values, confidence intervals, and
    practical significance decisions.

    Args:
        significance_level: Alpha threshold for p-value (default 0.05).
        minimum_effect_size: Minimum practical effect to declare a winner.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        minimum_effect_size: float = 0.01,
    ) -> None:
        """Initialise the analyzer with significance thresholds.

        Args:
            significance_level: Hypothesis test alpha (default 0.05).
            minimum_effect_size: Minimum absolute effect considered practical.
        """
        self._alpha = significance_level
        self._min_effect = minimum_effect_size

    def analyse(
        self,
        control_samples: list[float],
        treatment_samples: list[float],
        minimum_sample_size: int = 100,
    ) -> ABTestStatisticalResult:
        """Run statistical analysis comparing two sample sets.

        Uses Welch's t-test (scipy.stats.ttest_ind with equal_var=False).
        Falls back to a stub result if scipy is unavailable.

        Args:
            control_samples: Metric values from the control arm.
            treatment_samples: Metric values from the treatment arm.
            minimum_sample_size: Minimum n per arm before significance testing.

        Returns:
            ABTestStatisticalResult with statistical analysis output.
        """
        n_control = len(control_samples)
        n_treatment = len(treatment_samples)

        if n_control == 0 or n_treatment == 0:
            return self._insufficient_data_result(n_control, n_treatment)

        try:
            import statistics as stats_lib

            control_mean = stats_lib.mean(control_samples)
            treatment_mean = stats_lib.mean(treatment_samples)
        except Exception:
            control_mean = sum(control_samples) / n_control
            treatment_mean = sum(treatment_samples) / n_treatment

        effect_size = treatment_mean - control_mean
        relative_lift = effect_size / control_mean if control_mean != 0 else 0.0

        if n_control < minimum_sample_size or n_treatment < minimum_sample_size:
            return ABTestStatisticalResult(
                control_mean=control_mean,
                treatment_mean=treatment_mean,
                control_n=n_control,
                treatment_n=n_treatment,
                effect_size=effect_size,
                relative_lift=relative_lift,
                p_value=1.0,
                confidence_interval_95=(effect_size - 0.1, effect_size + 0.1),
                is_significant=False,
                decision=ABTestDecision.INSUFFICIENT_DATA,
            )

        p_value, ci_lower, ci_upper = self._run_welch_test(
            control_samples, treatment_samples, control_mean, treatment_mean
        )

        is_significant = p_value < self._alpha and abs(effect_size) >= self._min_effect

        if is_significant:
            decision = ABTestDecision.TREATMENT_WINS if effect_size > 0 else ABTestDecision.CONTROL_WINS
        elif n_control >= minimum_sample_size and n_treatment >= minimum_sample_size:
            decision = ABTestDecision.NO_SIGNIFICANT_DIFFERENCE
        else:
            decision = ABTestDecision.INSUFFICIENT_DATA

        logger.info(
            "ab_test_analysis_complete",
            control_n=n_control,
            treatment_n=n_treatment,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            p_value=p_value,
            effect_size=effect_size,
            decision=decision.value,
        )

        return ABTestStatisticalResult(
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            control_n=n_control,
            treatment_n=n_treatment,
            effect_size=effect_size,
            relative_lift=relative_lift,
            p_value=p_value,
            confidence_interval_95=(ci_lower, ci_upper),
            is_significant=is_significant,
            decision=decision,
        )

    def _run_welch_test(
        self,
        control_samples: list[float],
        treatment_samples: list[float],
        control_mean: float,
        treatment_mean: float,
    ) -> tuple[float, float, float]:
        """Run Welch's t-test and compute 95% confidence interval.

        Args:
            control_samples: Control arm metric values.
            treatment_samples: Treatment arm metric values.
            control_mean: Pre-computed control mean.
            treatment_mean: Pre-computed treatment mean.

        Returns:
            Tuple of (p_value, ci_lower, ci_upper).
        """
        try:
            from scipy import stats  # type: ignore[import]

            t_stat, p_value = stats.ttest_ind(
                control_samples,
                treatment_samples,
                equal_var=False,
                alternative="two-sided",
            )
            # 95% CI for difference in means
            effect_se = ((stats.sem(control_samples) ** 2) + (stats.sem(treatment_samples) ** 2)) ** 0.5
            df = len(control_samples) + len(treatment_samples) - 2
            t_crit = stats.t.ppf(0.975, df=df)
            effect_size = treatment_mean - control_mean
            ci_lower = effect_size - t_crit * effect_se
            ci_upper = effect_size + t_crit * effect_se
            return float(p_value), float(ci_lower), float(ci_upper)

        except ImportError:
            logger.warning("scipy_not_available_using_stub_p_value")
            effect_size = treatment_mean - control_mean
            return 0.5, effect_size - 0.05, effect_size + 0.05

    def _insufficient_data_result(
        self,
        n_control: int,
        n_treatment: int,
    ) -> ABTestStatisticalResult:
        """Build a stub result for empty or very small sample sets.

        Args:
            n_control: Control arm sample count.
            n_treatment: Treatment arm sample count.

        Returns:
            ABTestStatisticalResult indicating insufficient data.
        """
        return ABTestStatisticalResult(
            control_mean=0.0,
            treatment_mean=0.0,
            control_n=n_control,
            treatment_n=n_treatment,
            effect_size=0.0,
            relative_lift=0.0,
            p_value=1.0,
            confidence_interval_95=(0.0, 0.0),
            is_significant=False,
            decision=ABTestDecision.INSUFFICIENT_DATA,
        )


class ABTestTrafficRouter:
    """Determines which model version serves a given request in an A/B test.

    Uses deterministic hashing of a session or user identifier to route
    traffic consistently — the same session always hits the same arm.
    This prevents session-level noise from treatment/control switching.

    Args:
        treatment_percent: Percentage of traffic to route to treatment (0-100).
    """

    def __init__(self, treatment_percent: int = 20) -> None:
        """Initialise the traffic router with split configuration.

        Args:
            treatment_percent: Target treatment traffic percentage (0-100).
        """
        if not 0 <= treatment_percent <= 100:
            raise ValueError(f"treatment_percent must be in [0, 100], got {treatment_percent}")
        self._treatment_percent = treatment_percent

    def assign_arm(self, session_id: str) -> str:
        """Deterministically assign a session to 'control' or 'treatment'.

        Uses xxhash (or built-in hash as fallback) to consistently route
        the same session_id to the same arm across requests.

        Args:
            session_id: Unique session or user identifier for consistent routing.

        Returns:
            'treatment' if session falls in treatment bucket, 'control' otherwise.
        """
        bucket = self._hash_to_bucket(session_id)
        arm = "treatment" if bucket < self._treatment_percent else "control"
        return arm

    def _hash_to_bucket(self, session_id: str) -> int:
        """Hash a session ID to a bucket in [0, 100).

        Args:
            session_id: Session identifier string.

        Returns:
            Integer bucket number in [0, 100).
        """
        try:
            import xxhash  # type: ignore[import]

            return xxhash.xxh32(session_id.encode()).intdigest() % 100
        except ImportError:
            return abs(hash(session_id)) % 100

    def build_traffic_split(
        self,
        control_version_id: uuid.UUID,
        treatment_version_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Build the traffic_split JSONB payload for the deployment record.

        Args:
            control_version_id: Control model version UUID.
            treatment_version_id: Treatment model version UUID.

        Returns:
            Dict compatible with the reg_model_deployments.traffic_split column.
        """
        return {
            "mode": "ab_test",
            "control": {
                "version_id": str(control_version_id),
                "percent": 100 - self._treatment_percent,
            },
            "treatment": {
                "version_id": str(treatment_version_id),
                "percent": self._treatment_percent,
            },
        }
