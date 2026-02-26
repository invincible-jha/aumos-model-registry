"""Model Registry service settings extending AumOS base configuration."""

from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Configuration for the AumOS Model Registry service.

    Extends base AumOS settings with model-registry-specific configuration
    for MLflow integration, artifact storage, and cost tracking.
    """

    service_name: str = "aumos-model-registry"

    # MLflow backend store
    mlflow_tracking_uri: str = "postgresql+psycopg2://aumos:aumos_dev@localhost:5432/aumos"
    mlflow_artifact_root: str = "s3://aumos-model-artifacts"
    mlflow_server_port: int = 5000

    # MinIO / S3 artifact storage
    artifact_bucket_prefix: str = "reg-models"

    # Cost tracking
    default_gpu_hourly_cost_usd: float = 3.50
    default_cpu_hourly_cost_usd: float = 0.10
    default_storage_gb_monthly_cost_usd: float = 0.023

    # pgvector semantic search
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Stage transition approval (requires human sign-off when True)
    require_approval_for_production: bool = True

    model_config = SettingsConfigDict(env_prefix="AUMOS_REGISTRY_")
