"""Per-model cost attribution adapter for the AumOS Model Registry.

Tracks training, inference, and storage costs per model version. Provides
cost breakdown by resource type (GPU, CPU, storage), trend analysis across
versions, budget alerting, and structured cost report generation.

All monetary values use Decimal (never float) per AumOS coding standards.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from aumos_common.observability import get_logger

from aumos_model_registry.core.interfaces import IDeploymentRepository, IVersionRepository
from aumos_model_registry.core.models import ModelDeployment, ModelVersion

logger = get_logger(__name__)

# Default pricing rates (USD)
_DEFAULT_GPU_HOURLY_RATE_USD = Decimal("2.50")
_DEFAULT_CPU_HOURLY_RATE_USD = Decimal("0.096")
_DEFAULT_STORAGE_GB_MONTHLY_RATE_USD = Decimal("0.023")
_DEFAULT_INFERENCE_PER_MILLION_USD = Decimal("0.002")

# Budget alert threshold percentage (80% of budget)
_BUDGET_ALERT_THRESHOLD_PCT = Decimal("0.80")

# Cost rounding precision
_COST_PRECISION = Decimal("0.0001")


class ModelCostAttribution:
    """Per-model cost tracking and reporting across training, inference, and storage.

    Aggregates costs from three sources:
      1. Training cost — recorded at version creation time (GPU/CPU wall-clock hours).
      2. Inference cost — accumulated incrementally on each deployment's inference_count.
      3. Storage cost — derived from artifact size_bytes and monthly storage rate.

    Provides trend analysis, budget alerting, and structured cost reports.

    Usage::

        tracker = ModelCostAttribution(
            version_repo=version_repo,
            deployment_repo=deployment_repo,
        )
        report = await tracker.generate_cost_report(
            model_id=model_id,
            tenant_id=tenant_id,
        )
    """

    def __init__(
        self,
        version_repo: IVersionRepository,
        deployment_repo: IDeploymentRepository,
        gpu_hourly_rate_usd: Decimal = _DEFAULT_GPU_HOURLY_RATE_USD,
        cpu_hourly_rate_usd: Decimal = _DEFAULT_CPU_HOURLY_RATE_USD,
        storage_gb_monthly_rate_usd: Decimal = _DEFAULT_STORAGE_GB_MONTHLY_RATE_USD,
        inference_per_million_usd: Decimal = _DEFAULT_INFERENCE_PER_MILLION_USD,
    ) -> None:
        """Initialise the cost attribution adapter.

        Args:
            version_repo: Model version repository.
            deployment_repo: Deployment repository with inference cost tracking.
            gpu_hourly_rate_usd: USD cost per GPU-hour for training attribution.
            cpu_hourly_rate_usd: USD cost per CPU-hour for training attribution.
            storage_gb_monthly_rate_usd: USD cost per GB per month for storage.
            inference_per_million_usd: USD cost per million inference requests.
        """
        self._versions = version_repo
        self._deployments = deployment_repo
        self._gpu_rate = gpu_hourly_rate_usd
        self._cpu_rate = cpu_hourly_rate_usd
        self._storage_rate = storage_gb_monthly_rate_usd
        self._inference_rate = inference_per_million_usd

    async def get_version_cost_breakdown(
        self,
        version_id: uuid.UUID,
        storage_months: int = 1,
    ) -> dict[str, Any]:
        """Return a full cost breakdown for a single model version.

        Args:
            version_id: Target version UUID.
            storage_months: Number of months to amortise storage cost over.

        Returns:
            Dict with training_cost, inference_cost, storage_cost, total_cost,
            and resource_breakdown keys.
        """
        version = await self._versions.get_by_id(version_id)
        if version is None:
            raise ValueError(f"Model version {version_id} not found")

        deployments = await self._deployments.list_by_version(version_id)

        training_cost = self._compute_training_cost(version)
        inference_cost = self._compute_inference_cost(deployments)
        storage_cost = self._compute_storage_cost(version, months=storage_months)

        resource_breakdown = self._build_resource_breakdown(
            version=version,
            deployments=deployments,
            training_cost=training_cost,
            inference_cost=inference_cost,
            storage_cost=storage_cost,
        )

        total_cost = (training_cost + inference_cost + storage_cost).quantize(
            _COST_PRECISION, rounding=ROUND_HALF_UP
        )

        breakdown = {
            "version_id": str(version_id),
            "version_number": version.version,
            "stage": version.stage,
            "training_cost_usd": str(training_cost),
            "inference_cost_usd": str(inference_cost),
            "storage_cost_usd": str(storage_cost),
            "total_cost_usd": str(total_cost),
            "resource_breakdown": resource_breakdown,
            "deployment_count": len(deployments),
            "total_inference_count": sum(d.inference_count for d in deployments),
            "computed_at": datetime.now(UTC).isoformat(),
        }

        logger.info(
            "Version cost breakdown computed",
            version_id=str(version_id),
            total_cost=str(total_cost),
        )
        return breakdown

    async def get_model_cost_summary(
        self,
        model_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Aggregate costs across all versions of a model.

        Args:
            model_id: Parent model UUID.
            tenant_id: Owning tenant.

        Returns:
            Dict with per-version costs and model-level totals.
        """
        versions = await self._versions.list_by_model(model_id)

        version_costs: list[dict[str, Any]] = []
        total_training = Decimal("0.00")
        total_inference = Decimal("0.00")
        total_storage = Decimal("0.00")

        for version in versions:
            training = self._compute_training_cost(version)
            deployments = await self._deployments.list_by_version(version.id)
            inference = self._compute_inference_cost(deployments)
            storage = self._compute_storage_cost(version)

            total_training += training
            total_inference += inference
            total_storage += storage

            version_costs.append(
                {
                    "version_id": str(version.id),
                    "version_number": version.version,
                    "stage": version.stage,
                    "training_cost_usd": str(training),
                    "inference_cost_usd": str(inference),
                    "storage_cost_usd": str(storage),
                    "version_total_usd": str(
                        (training + inference + storage).quantize(_COST_PRECISION)
                    ),
                }
            )

        model_total = (total_training + total_inference + total_storage).quantize(
            _COST_PRECISION, rounding=ROUND_HALF_UP
        )

        return {
            "model_id": str(model_id),
            "tenant_id": str(tenant_id),
            "version_count": len(versions),
            "total_training_cost_usd": str(total_training.quantize(_COST_PRECISION)),
            "total_inference_cost_usd": str(total_inference.quantize(_COST_PRECISION)),
            "total_storage_cost_usd": str(total_storage.quantize(_COST_PRECISION)),
            "model_total_cost_usd": str(model_total),
            "version_costs": version_costs,
            "computed_at": datetime.now(UTC).isoformat(),
        }

    async def analyze_cost_trends(
        self,
        model_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Compute cost trend analysis across all versions of a model.

        Calculates cost delta between consecutive versions and identifies
        whether costs are increasing or decreasing over time.

        Args:
            model_id: Parent model UUID.

        Returns:
            Dict with trend data including direction, deltas, and projected next cost.
        """
        versions = await self._versions.list_by_model(model_id)

        if len(versions) < 2:
            return {
                "model_id": str(model_id),
                "trend": "insufficient_data",
                "version_count": len(versions),
                "cost_deltas": [],
                "projected_next_cost_usd": None,
            }

        # Sort by version number ascending for trend calculation
        sorted_versions = sorted(versions, key=lambda v: v.version)

        version_totals: list[tuple[int, Decimal]] = []
        for version in sorted_versions:
            deployments = await self._deployments.list_by_version(version.id)
            total = (
                self._compute_training_cost(version)
                + self._compute_inference_cost(deployments)
                + self._compute_storage_cost(version)
            )
            version_totals.append((version.version, total))

        # Compute deltas between consecutive versions
        deltas: list[dict[str, Any]] = []
        for i in range(1, len(version_totals)):
            prev_version, prev_cost = version_totals[i - 1]
            curr_version, curr_cost = version_totals[i]
            delta = curr_cost - prev_cost
            pct_change = (delta / prev_cost * 100) if prev_cost != 0 else Decimal("0")
            deltas.append(
                {
                    "from_version": prev_version,
                    "to_version": curr_version,
                    "cost_delta_usd": str(delta.quantize(_COST_PRECISION)),
                    "pct_change": str(pct_change.quantize(Decimal("0.01"))),
                    "direction": "increase" if delta > 0 else "decrease" if delta < 0 else "flat",
                }
            )

        # Simple linear projection for next version cost
        if len(version_totals) >= 2:
            last_cost = version_totals[-1][1]
            avg_delta = sum(d - c for (_, c), (_, d) in zip(version_totals, version_totals[1:])) / (
                len(version_totals) - 1
            )
            projected = (last_cost + avg_delta).quantize(_COST_PRECISION)
        else:
            projected = version_totals[-1][1]

        total_delta = version_totals[-1][1] - version_totals[0][1]
        overall_trend = "increasing" if total_delta > 0 else "decreasing" if total_delta < 0 else "flat"

        return {
            "model_id": str(model_id),
            "trend": overall_trend,
            "version_count": len(versions),
            "cost_deltas": deltas,
            "projected_next_cost_usd": str(projected),
            "analysis_period_versions": [v for v, _ in version_totals],
            "computed_at": datetime.now(UTC).isoformat(),
        }

    async def check_budget_alert(
        self,
        model_id: uuid.UUID,
        tenant_id: uuid.UUID,
        budget_usd: Decimal,
    ) -> dict[str, Any]:
        """Check whether a model's total cost has exceeded the alert threshold.

        Args:
            model_id: Parent model UUID.
            tenant_id: Owning tenant.
            budget_usd: Total budget allocation in USD.

        Returns:
            Dict with alert_triggered flag, current_spend, and threshold_usd.
        """
        summary = await self.get_model_cost_summary(model_id=model_id, tenant_id=tenant_id)
        current_spend = Decimal(summary["model_total_cost_usd"])
        alert_threshold = (budget_usd * _BUDGET_ALERT_THRESHOLD_PCT).quantize(_COST_PRECISION)
        alert_triggered = current_spend >= alert_threshold
        budget_utilisation_pct = (current_spend / budget_usd * 100).quantize(
            Decimal("0.01")
        ) if budget_usd > 0 else Decimal("0")

        if alert_triggered:
            logger.warning(
                "Model budget alert triggered",
                model_id=str(model_id),
                tenant_id=str(tenant_id),
                current_spend=str(current_spend),
                budget=str(budget_usd),
                utilisation_pct=str(budget_utilisation_pct),
            )

        return {
            "model_id": str(model_id),
            "tenant_id": str(tenant_id),
            "budget_usd": str(budget_usd),
            "current_spend_usd": str(current_spend),
            "alert_threshold_usd": str(alert_threshold),
            "budget_utilisation_pct": str(budget_utilisation_pct),
            "alert_triggered": alert_triggered,
            "checked_at": datetime.now(UTC).isoformat(),
        }

    async def generate_cost_report(
        self,
        model_id: uuid.UUID,
        tenant_id: uuid.UUID,
        include_trends: bool = True,
    ) -> dict[str, Any]:
        """Generate a comprehensive cost report for a model.

        Args:
            model_id: Parent model UUID.
            tenant_id: Owning tenant.
            include_trends: Whether to include trend analysis section.

        Returns:
            Full cost report dict suitable for JSON serialisation.
        """
        summary = await self.get_model_cost_summary(model_id=model_id, tenant_id=tenant_id)

        report: dict[str, Any] = {
            "report_type": "model_cost_report",
            "model_id": str(model_id),
            "tenant_id": str(tenant_id),
            "generated_at": datetime.now(UTC).isoformat(),
            "summary": summary,
        }

        if include_trends:
            report["trends"] = await self.analyze_cost_trends(model_id=model_id)

        logger.info(
            "Cost report generated",
            model_id=str(model_id),
            tenant_id=str(tenant_id),
        )
        return report

    def _compute_training_cost(self, version: ModelVersion) -> Decimal:
        """Compute the training cost for a model version.

        Uses the recorded training_cost if present. Falls back to estimating
        from size_bytes as a proxy when no explicit cost is recorded.

        Args:
            version: ModelVersion ORM instance.

        Returns:
            Training cost as Decimal.
        """
        if version.training_cost is not None:
            return version.training_cost

        # Proxy estimate: $0.10 per GB of model artifact size
        if version.size_bytes is not None:
            gb = Decimal(str(version.size_bytes)) / Decimal("1073741824")
            return (gb * Decimal("0.10")).quantize(_COST_PRECISION, rounding=ROUND_HALF_UP)

        return Decimal("0.00")

    def _compute_inference_cost(self, deployments: list[ModelDeployment]) -> Decimal:
        """Sum inference costs across all deployments.

        Args:
            deployments: List of ModelDeployment ORM instances.

        Returns:
            Total inference cost as Decimal.
        """
        total = Decimal("0.00")
        for deployment in deployments:
            if deployment.inference_cost:
                total += deployment.inference_cost
        return total.quantize(_COST_PRECISION, rounding=ROUND_HALF_UP)

    def _compute_storage_cost(
        self, version: ModelVersion, months: int = 1
    ) -> Decimal:
        """Compute monthly storage cost based on artifact size.

        Args:
            version: ModelVersion ORM instance.
            months: Number of months to amortise storage cost over.

        Returns:
            Storage cost as Decimal.
        """
        if version.size_bytes is None:
            return Decimal("0.00")

        gb = Decimal(str(version.size_bytes)) / Decimal("1073741824")
        monthly = gb * self._storage_rate * Decimal(str(months))
        return monthly.quantize(_COST_PRECISION, rounding=ROUND_HALF_UP)

    def _build_resource_breakdown(
        self,
        version: ModelVersion,
        deployments: list[ModelDeployment],
        training_cost: Decimal,
        inference_cost: Decimal,
        storage_cost: Decimal,
    ) -> dict[str, Any]:
        """Build a resource-type breakdown for cost reporting.

        Args:
            version: ModelVersion ORM instance.
            deployments: Deployment instances for the version.
            training_cost: Pre-computed training cost.
            inference_cost: Pre-computed inference cost.
            storage_cost: Pre-computed storage cost.

        Returns:
            Dict with gpu, cpu, storage, and inference resource costs.
        """
        # Attribute training cost to GPU (majority) vs CPU (small portion)
        gpu_training = (training_cost * Decimal("0.80")).quantize(_COST_PRECISION)
        cpu_training = (training_cost * Decimal("0.20")).quantize(_COST_PRECISION)

        size_gb = (
            Decimal(str(version.size_bytes)) / Decimal("1073741824")
            if version.size_bytes
            else Decimal("0")
        )

        return {
            "gpu_cost_usd": str(gpu_training),
            "cpu_cost_usd": str(cpu_training),
            "storage_cost_usd": str(storage_cost),
            "inference_cost_usd": str(inference_cost),
            "storage_gb": str(size_gb.quantize(Decimal("0.001"))),
            "active_deployments": sum(1 for d in deployments if d.status == "active"),
            "total_inference_requests": sum(d.inference_count for d in deployments),
        }
