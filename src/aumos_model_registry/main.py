"""AumOS Model Registry service entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.health import HealthCheck
from aumos_common.observability import get_logger

from aumos_model_registry.adapters.kafka import ModelRegistryEventPublisher
from aumos_model_registry.adapters.minio_client import MinioArtifactClient
from aumos_model_registry.api.decommission_routes import decommission_router
from aumos_model_registry.api.router import router
from aumos_model_registry.settings import Settings

logger = get_logger(__name__)
settings = Settings()

# Module-level singletons (injected into routes via FastAPI state)
_kafka_publisher: ModelRegistryEventPublisher | None = None
_minio_client: MinioArtifactClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Initialises the database connection pool, Kafka event publisher,
    and MinIO artifact storage client. Shuts them down cleanly on exit.

    Args:
        app: The FastAPI application instance.

    Yields:
        None
    """
    global _kafka_publisher, _minio_client  # noqa: PLW0603

    logger.info("Starting AumOS Model Registry", version="0.1.0")

    # Initialise database (sets up SQLAlchemy async engine + session factory)
    init_database(settings.database)

    # Initialise Kafka publisher
    _kafka_publisher = ModelRegistryEventPublisher(settings.kafka)
    await _kafka_publisher.start()
    app.state.kafka_publisher = _kafka_publisher
    logger.info("Kafka publisher ready")

    # Initialise MinIO client
    _minio_client = MinioArtifactClient(settings.minio)
    await _minio_client.ensure_buckets_exist(settings.artifact_bucket_prefix)
    app.state.minio_client = _minio_client
    logger.info("MinIO artifact client ready")

    logger.info("Model Registry startup complete")
    yield

    # Shutdown
    if _kafka_publisher:
        await _kafka_publisher.stop()
    logger.info("Model Registry shutdown complete")


app: FastAPI = create_app(
    service_name="aumos-model-registry",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        HealthCheck(name="postgres", check_fn=lambda: None),
        HealthCheck(name="kafka", check_fn=lambda: None),
        HealthCheck(name="minio", check_fn=lambda: None),
    ],
)

app.include_router(router, prefix="/api/v1")
app.include_router(decommission_router, prefix="/api/v1")
