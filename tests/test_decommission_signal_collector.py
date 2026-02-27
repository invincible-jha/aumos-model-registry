"""Tests for DecommissionSignalCollector.

Covers:
  - Weight validation
  - score_model() with all signals at various values
  - HEALTHY / REVIEW / AUTO_DECOMMISSION recommendation thresholds
  - primary_signal selection
  - score_models_batch() sort order
  - Custom signal weights
  - Clamp behaviour for out-of-range inputs
  - NoOp signal source produces zero scores
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aumos_model_registry.decommission.signal_collector import (
    AUTO_DECOMMISSION_THRESHOLD,
    REVIEW_THRESHOLD,
    DecommissionScore,
    DecommissionSignalCollector,
    IDecommissionSignalSource,
    _NoOpSignalSource,
    _clamp,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def noop_collector() -> DecommissionSignalCollector:
    """Return a collector backed by the no-op signal source."""
    return DecommissionSignalCollector()


@pytest.fixture
def mock_source() -> AsyncMock:
    """Return an AsyncMock implementing IDecommissionSignalSource."""
    source = AsyncMock(spec=IDecommissionSignalSource)
    source.get_drift_severity.return_value = 0.0
    source.get_cost_per_decision_trend.return_value = 0.0
    source.get_traffic_decline_pct.return_value = 0.0
    source.has_newer_model_available.return_value = False
    source.get_compliance_expiry_score.return_value = 0.0
    return source


@pytest.fixture
def collector_with_mock(mock_source: AsyncMock) -> DecommissionSignalCollector:
    """Return a collector backed by the mock source."""
    return DecommissionSignalCollector(signal_source=mock_source)


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------


def test_default_weights_sum_to_one() -> None:
    """Default signal weights must sum to exactly 1.0."""
    collector = DecommissionSignalCollector()
    assert abs(sum(collector.weights.values()) - 1.0) < 1e-6


def test_invalid_weights_raise_value_error() -> None:
    """Weights that do not sum to 1.0 raise ValueError on construction."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        DecommissionSignalCollector(
            signal_weights={
                "drift_severity": 0.50,
                "cost_per_decision_trend": 0.50,
                "traffic_decline_pct": 0.50,  # sum = 1.50, invalid
                "newer_model_available": 0.00,
                "compliance_expiry": 0.00,
            }
        )


def test_custom_weights_accepted() -> None:
    """Custom weights summing to 1.0 are accepted without error."""
    collector = DecommissionSignalCollector(
        signal_weights={
            "drift_severity": 0.40,
            "cost_per_decision_trend": 0.20,
            "traffic_decline_pct": 0.20,
            "newer_model_available": 0.10,
            "compliance_expiry": 0.10,
        }
    )
    assert abs(sum(collector.weights.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# score_model() — HEALTHY recommendation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_model_all_zeros_returns_healthy(
    collector_with_mock: DecommissionSignalCollector,
) -> None:
    """All-zero signals → composite = 0.0 → HEALTHY recommendation."""
    score = await collector_with_mock.score_model(
        model_id="model-001",
        model_name="Test Model",
        tenant_id="tenant-a",
    )
    assert score.composite_score == 0.0
    assert score.recommendation == "HEALTHY"
    assert score.should_flag_for_review is False
    assert score.should_auto_decommission is False


@pytest.mark.asyncio
async def test_score_model_low_signals_healthy(
    mock_source: AsyncMock,
) -> None:
    """Low but non-zero signals below review threshold produce HEALTHY."""
    mock_source.get_drift_severity.return_value = 0.2
    mock_source.get_cost_per_decision_trend.return_value = 0.1
    # composite ≈ 0.2*0.30 + 0.1*0.25 = 0.060 + 0.025 = 0.085
    collector = DecommissionSignalCollector(signal_source=mock_source)
    score = await collector.score_model("m1", "name", "t1")
    assert score.composite_score < REVIEW_THRESHOLD
    assert score.recommendation == "HEALTHY"


# ---------------------------------------------------------------------------
# score_model() — REVIEW recommendation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_model_above_review_threshold(
    mock_source: AsyncMock,
) -> None:
    """Composite score just above 0.7 produces REVIEW recommendation."""
    # drift=1.0 (w=0.30), cost=1.0 (w=0.25), traffic=1.0 (w=0.20) = 0.75 > 0.7
    mock_source.get_drift_severity.return_value = 1.0
    mock_source.get_cost_per_decision_trend.return_value = 1.0
    mock_source.get_traffic_decline_pct.return_value = 1.0
    mock_source.has_newer_model_available.return_value = False
    mock_source.get_compliance_expiry_score.return_value = 0.0
    collector = DecommissionSignalCollector(signal_source=mock_source)
    score = await collector.score_model("m1", "n", "t1")
    assert score.composite_score > REVIEW_THRESHOLD
    assert score.composite_score <= AUTO_DECOMMISSION_THRESHOLD
    assert score.recommendation == "REVIEW"
    assert score.should_flag_for_review is True
    assert score.should_auto_decommission is False


# ---------------------------------------------------------------------------
# score_model() — AUTO_DECOMMISSION recommendation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_model_all_ones_auto_decommission(
    mock_source: AsyncMock,
) -> None:
    """All signals at 1.0 → composite = 1.0 → AUTO_DECOMMISSION."""
    mock_source.get_drift_severity.return_value = 1.0
    mock_source.get_cost_per_decision_trend.return_value = 1.0
    mock_source.get_traffic_decline_pct.return_value = 1.0
    mock_source.has_newer_model_available.return_value = True
    mock_source.get_compliance_expiry_score.return_value = 1.0
    collector = DecommissionSignalCollector(signal_source=mock_source)
    score = await collector.score_model("m-full", "Full Signal Model", "tenant-x")
    assert score.composite_score == pytest.approx(1.0)
    assert score.recommendation == "AUTO_DECOMMISSION"
    assert score.should_auto_decommission is True
    assert score.should_flag_for_review is True


# ---------------------------------------------------------------------------
# Signal score structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_model_signal_scores_keys(
    collector_with_mock: DecommissionSignalCollector,
) -> None:
    """signal_scores dict must contain all five signal keys."""
    score = await collector_with_mock.score_model("m", "n", "t")
    expected_keys = {
        "drift_severity",
        "cost_per_decision_trend",
        "traffic_decline_pct",
        "newer_model_available",
        "compliance_expiry",
    }
    assert set(score.signal_scores.keys()) == expected_keys


@pytest.mark.asyncio
async def test_score_model_weighted_contribution_correct(
    mock_source: AsyncMock,
) -> None:
    """Weighted contribution equals raw * weight for each signal."""
    mock_source.get_drift_severity.return_value = 0.5
    collector = DecommissionSignalCollector(signal_source=mock_source)
    score = await collector.score_model("m", "n", "t")
    drift_entry = score.signal_scores["drift_severity"]
    assert drift_entry["raw"] == pytest.approx(0.5)
    assert drift_entry["weight"] == pytest.approx(0.30)
    assert drift_entry["weighted_contribution"] == pytest.approx(0.5 * 0.30)


# ---------------------------------------------------------------------------
# primary_signal selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_primary_signal_is_highest_weighted_contribution(
    mock_source: AsyncMock,
) -> None:
    """primary_signal is the signal with the highest weighted contribution."""
    # drift = 1.0 * 0.30 = 0.30 (highest)
    # traffic = 1.0 * 0.20 = 0.20
    # newer = True → 1.0 * 0.15 = 0.15
    mock_source.get_drift_severity.return_value = 1.0
    mock_source.get_traffic_decline_pct.return_value = 1.0
    mock_source.has_newer_model_available.return_value = True
    collector = DecommissionSignalCollector(signal_source=mock_source)
    score = await collector.score_model("m", "n", "t")
    assert score.primary_signal == "drift_severity"


@pytest.mark.asyncio
async def test_primary_signal_when_cost_trend_dominates(
    mock_source: AsyncMock,
) -> None:
    """primary_signal = cost_per_decision_trend when it has the highest contribution."""
    mock_source.get_cost_per_decision_trend.return_value = 1.0  # 0.25 contribution
    mock_source.get_drift_severity.return_value = 0.0
    collector = DecommissionSignalCollector(signal_source=mock_source)
    score = await collector.score_model("m", "n", "t")
    assert score.primary_signal == "cost_per_decision_trend"


# ---------------------------------------------------------------------------
# Clamp helper
# ---------------------------------------------------------------------------


def test_clamp_within_range() -> None:
    """Values within [0, 1] are returned unchanged."""
    assert _clamp(0.5) == pytest.approx(0.5)
    assert _clamp(0.0) == pytest.approx(0.0)
    assert _clamp(1.0) == pytest.approx(1.0)


def test_clamp_below_zero() -> None:
    """Negative values are clamped to 0.0."""
    assert _clamp(-0.5) == pytest.approx(0.0)


def test_clamp_above_one() -> None:
    """Values above 1.0 are clamped to 1.0."""
    assert _clamp(1.5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# NoOp source
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_noop_source_all_zeros() -> None:
    """NoOp source returns 0.0 for all numeric signals."""
    source = _NoOpSignalSource()
    assert await source.get_drift_severity("m", "t") == pytest.approx(0.0)
    assert await source.get_cost_per_decision_trend("m", "t") == pytest.approx(0.0)
    assert await source.get_traffic_decline_pct("m", "t") == pytest.approx(0.0)
    assert await source.has_newer_model_available("m", "t") is False
    assert await source.get_compliance_expiry_score("m", "t") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_models_batch()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_models_batch_sorted_descending(
    mock_source: AsyncMock,
) -> None:
    """Batch results are sorted by composite_score descending."""
    call_count = 0

    async def drift_side_effect(model_id: str, tenant_id: str) -> float:
        nonlocal call_count
        call_count += 1
        return 0.1 * call_count  # m1=0.1, m2=0.2, m3=0.3

    mock_source.get_drift_severity.side_effect = drift_side_effect
    collector = DecommissionSignalCollector(signal_source=mock_source)
    models = [
        ("model-a", "Model A", "tenant-1"),
        ("model-b", "Model B", "tenant-1"),
        ("model-c", "Model C", "tenant-1"),
    ]
    scores = await collector.score_models_batch(models)
    assert len(scores) == 3
    # Scores should be descending
    assert scores[0].composite_score >= scores[1].composite_score
    assert scores[1].composite_score >= scores[2].composite_score


@pytest.mark.asyncio
async def test_score_models_batch_empty_list(
    noop_collector: DecommissionSignalCollector,
) -> None:
    """Empty model list returns empty list without error."""
    scores = await noop_collector.score_models_batch([])
    assert scores == []


@pytest.mark.asyncio
async def test_score_models_batch_returns_all_models(
    noop_collector: DecommissionSignalCollector,
) -> None:
    """Batch includes all models even when composite_score == 0.0."""
    models = [
        ("m1", "Model 1", "tenant"),
        ("m2", "Model 2", "tenant"),
    ]
    scores = await noop_collector.score_models_batch(models)
    assert len(scores) == 2


# ---------------------------------------------------------------------------
# weights property
# ---------------------------------------------------------------------------


def test_weights_property_is_copy() -> None:
    """Modifying the returned weights dict does not affect the collector."""
    collector = DecommissionSignalCollector()
    weights = collector.weights
    weights["drift_severity"] = 0.99
    assert collector.weights["drift_severity"] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# DecommissionScore identity fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_model_fields_populated(
    mock_source: AsyncMock,
) -> None:
    """Score has correct model_id, model_name, tenant_id, and evaluated_at."""
    collector = DecommissionSignalCollector(signal_source=mock_source)
    score = await collector.score_model("model-xyz", "My Model", "acme")
    assert score.model_id == "model-xyz"
    assert score.model_name == "My Model"
    assert score.tenant_id == "acme"
    assert score.evaluated_at is not None
