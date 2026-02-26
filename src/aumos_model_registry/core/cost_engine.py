"""Per-model cost attribution engine.

Calculates and aggregates costs across three dimensions:
  1. Training cost   — GPU/CPU hours during model training
  2. Inference cost  — accumulated cost from deployment tracking
  3. Storage cost    — monthly cost derived from artifact size

All monetary values are returned as Decimal with 2 decimal places
to avoid floating-point rounding errors in financial calculations.
"""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from aumos_model_registry.core.models import ModelDeployment, ModelVersion


_TWO_PLACES = Decimal("0.01")


@dataclass(frozen=True)
class ModelCostBreakdown:
    """Aggregated cost attribution for a model version.

    Attributes:
        model_version_id: UUID string of the model version.
        training_cost_usd: One-time training compute cost.
        inference_cost_usd: Running total from all deployments.
        storage_cost_monthly_usd: Estimated monthly artifact storage cost.
        total_cost_usd: Sum of all cost dimensions.
        currency: ISO 4217 currency code (always USD for now).
    """

    model_version_id: str
    training_cost_usd: Decimal
    inference_cost_usd: Decimal
    storage_cost_monthly_usd: Decimal
    total_cost_usd: Decimal
    currency: str = "USD"


@dataclass(frozen=True)
class TrainingCostInput:
    """Inputs required to estimate training cost.

    Attributes:
        gpu_hours: Number of GPU-hours consumed during training.
        cpu_hours: Number of CPU-hours consumed.
        gpu_hourly_rate_usd: Cost per GPU-hour in USD.
        cpu_hourly_rate_usd: Cost per CPU-hour in USD.
        additional_costs_usd: Any extra costs (storage I/O, data transfer, etc.).
    """

    gpu_hours: float
    cpu_hours: float
    gpu_hourly_rate_usd: float = 3.50
    cpu_hourly_rate_usd: float = 0.10
    additional_costs_usd: float = 0.0


def estimate_training_cost(inputs: TrainingCostInput) -> Decimal:
    """Calculate estimated training cost from compute usage.

    Args:
        inputs: Training cost input parameters.

    Returns:
        Rounded total training cost in USD.
    """
    gpu_cost = Decimal(str(inputs.gpu_hours)) * Decimal(str(inputs.gpu_hourly_rate_usd))
    cpu_cost = Decimal(str(inputs.cpu_hours)) * Decimal(str(inputs.cpu_hourly_rate_usd))
    additional = Decimal(str(inputs.additional_costs_usd))
    total = gpu_cost + cpu_cost + additional
    return total.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)


def estimate_storage_cost_monthly(
    size_bytes: int,
    gb_monthly_rate_usd: float = 0.023,
) -> Decimal:
    """Calculate estimated monthly artifact storage cost.

    Uses AWS S3-standard pricing as a baseline. MinIO on-prem costs
    can be substituted via the gb_monthly_rate_usd parameter.

    Args:
        size_bytes: Artifact size in bytes.
        gb_monthly_rate_usd: Storage cost per GB per month in USD.

    Returns:
        Rounded monthly storage cost in USD.
    """
    size_gb = Decimal(str(size_bytes)) / Decimal("1073741824")  # 1024^3
    cost = size_gb * Decimal(str(gb_monthly_rate_usd))
    return cost.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)


def aggregate_deployment_inference_cost(
    deployments: list[ModelDeployment],
) -> Decimal:
    """Sum inference costs across all deployments of a model version.

    Args:
        deployments: List of ModelDeployment ORM instances.

    Returns:
        Total accumulated inference cost in USD.
    """
    total = sum(
        (deployment.inference_cost for deployment in deployments),
        start=Decimal("0.00"),
    )
    return total.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)


def calculate_model_version_cost(
    version: ModelVersion,
    deployments: list[ModelDeployment],
    storage_gb_monthly_rate_usd: float = 0.023,
) -> ModelCostBreakdown:
    """Produce a full cost breakdown for a model version.

    Args:
        version: SQLAlchemy ModelVersion ORM instance.
        deployments: All deployments for this version.
        storage_gb_monthly_rate_usd: Per-GB monthly storage rate.

    Returns:
        ModelCostBreakdown with all three cost dimensions populated.
    """
    training_cost = (
        version.training_cost
        if version.training_cost is not None
        else Decimal("0.00")
    )
    inference_cost = aggregate_deployment_inference_cost(deployments)
    storage_cost = (
        estimate_storage_cost_monthly(version.size_bytes, storage_gb_monthly_rate_usd)
        if version.size_bytes is not None
        else Decimal("0.00")
    )

    total = (training_cost + inference_cost + storage_cost).quantize(
        _TWO_PLACES, rounding=ROUND_HALF_UP
    )

    return ModelCostBreakdown(
        model_version_id=str(version.id),
        training_cost_usd=training_cost.quantize(_TWO_PLACES),
        inference_cost_usd=inference_cost,
        storage_cost_monthly_usd=storage_cost,
        total_cost_usd=total,
        currency="USD",
    )
