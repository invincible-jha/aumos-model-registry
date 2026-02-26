"""Kafka event publisher for the AumOS Model Registry.

Wraps aumos-common EventPublisher with registry-specific typed event methods
for model lifecycle events: registered, deployed, retired.
"""

import uuid

from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class ModelRegistryEventPublisher:
    """Typed Kafka event publisher for model lifecycle events.

    Delegates to aumos-common EventPublisher for transport; this class
    provides registry-specific named methods with typed parameters.
    """

    def __init__(self, kafka_settings: object) -> None:
        """Initialise the underlying EventPublisher.

        Args:
            kafka_settings: Kafka settings object from AumOSSettings.kafka.
        """
        self._publisher = EventPublisher(kafka_settings)

    async def start(self) -> None:
        """Start the underlying Kafka producer.

        Must be called during application startup before publishing events.
        """
        await self._publisher.start()
        logger.info("Model Registry Kafka publisher started")

    async def stop(self) -> None:
        """Flush and close the underlying Kafka producer.

        Must be called during application shutdown to avoid message loss.
        """
        await self._publisher.stop()
        logger.info("Model Registry Kafka publisher stopped")

    async def publish_model_registered(
        self,
        tenant_id: uuid.UUID,
        model_id: uuid.UUID,
        model_name: str,
    ) -> None:
        """Publish a model.registered event to the MODEL_LIFECYCLE topic.

        Args:
            tenant_id: Owning tenant UUID.
            model_id: Newly registered model UUID.
            model_name: Human-readable model name.
        """
        await self._publisher.publish(
            Topics.MODEL_LIFECYCLE,
            {
                "event_type": "model.registered",
                "tenant_id": str(tenant_id),
                "model_id": str(model_id),
                "name": model_name,
            },
        )
        logger.info(
            "Published model.registered",
            model_id=str(model_id),
            tenant_id=str(tenant_id),
        )

    async def publish_model_deployed(
        self,
        tenant_id: uuid.UUID,
        model_id: uuid.UUID,
        version_id: uuid.UUID,
        environment: str,
    ) -> None:
        """Publish a model.deployed event to the MODEL_LIFECYCLE topic.

        Args:
            tenant_id: Owning tenant UUID.
            model_id: Parent model UUID.
            version_id: Deployed version UUID.
            environment: Target deployment environment (dev, staging, production).
        """
        await self._publisher.publish(
            Topics.MODEL_LIFECYCLE,
            {
                "event_type": "model.deployed",
                "tenant_id": str(tenant_id),
                "model_id": str(model_id),
                "version_id": str(version_id),
                "environment": environment,
            },
        )
        logger.info(
            "Published model.deployed",
            model_id=str(model_id),
            version_id=str(version_id),
            environment=environment,
        )

    async def publish_model_retired(
        self,
        tenant_id: uuid.UUID,
        model_id: uuid.UUID,
    ) -> None:
        """Publish a model.retired event to the MODEL_LIFECYCLE topic.

        Args:
            tenant_id: Owning tenant UUID.
            model_id: Retired model UUID.
        """
        await self._publisher.publish(
            Topics.MODEL_LIFECYCLE,
            {
                "event_type": "model.retired",
                "tenant_id": str(tenant_id),
                "model_id": str(model_id),
            },
        )
        logger.info(
            "Published model.retired",
            model_id=str(model_id),
            tenant_id=str(tenant_id),
        )

    async def publish(self, topic: str, payload: dict) -> None:
        """Generic passthrough for arbitrary event payloads.

        Args:
            topic: Kafka topic name.
            payload: Event payload dict (must include tenant_id).
        """
        await self._publisher.publish(topic, payload)
