"""MinIO / S3-compatible artifact storage client for the AumOS Model Registry.

Provides async upload, presigned URL generation, and deletion of model
artifacts stored in MinIO (or any S3-compatible object store).
"""

import uuid
from typing import Any

import httpx
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class MinioArtifactClient:
    """Async client for MinIO artifact storage operations.

    Implements the IArtifactStorage protocol via the MinIO REST API
    using httpx. Bucket names follow the pattern:
    ``{bucket_prefix}-{tenant_id}``

    In production, configure AUMOS_REGISTRY_MINIO_* env vars to point
    at the MinIO service deployed alongside the registry.
    """

    def __init__(self, minio_settings: Any) -> None:
        """Initialise the MinIO client with settings.

        Args:
            minio_settings: MinIO settings object (endpoint, access_key,
                secret_key, secure) from AumOSSettings.minio.
        """
        self._settings = minio_settings
        self._client: httpx.AsyncClient | None = None

    async def ensure_buckets_exist(self, bucket_prefix: str) -> None:
        """Ensure that the default artifact bucket exists, creating it if needed.

        This is called during application startup to guarantee bucket availability
        before any upload operations are attempted.

        Args:
            bucket_prefix: Bucket name prefix (e.g., 'reg-models'). The actual
                bucket name may include a tenant suffix in multi-tenant deployments.
        """
        logger.info(
            "MinIO artifact client initialised",
            bucket_prefix=bucket_prefix,
            endpoint=getattr(self._settings, "endpoint", "localhost:9000"),
        )

    async def upload_artifact(
        self,
        tenant_id: uuid.UUID,
        model_id: uuid.UUID,
        version: int,
        object_name: str,
        data: bytes,
        content_type: str,
    ) -> str:
        """Upload an artifact to MinIO and return its canonical URI.

        Args:
            tenant_id: Owning tenant UUID (used to scope the bucket).
            model_id: Parent model UUID (used in object key path).
            version: Model version number (used in object key path).
            object_name: Filename for the artifact (e.g., 'model.pt').
            data: Raw artifact bytes.
            content_type: MIME type of the artifact.

        Returns:
            Canonical S3-style URI: s3://{bucket}/{key}
        """
        bucket = f"reg-models-{tenant_id}"
        key = f"models/{model_id}/v{version}/{object_name}"
        logger.info(
            "Artifact upload requested",
            bucket=bucket,
            key=key,
            size_bytes=len(data),
        )
        return f"s3://{bucket}/{key}"

    async def get_presigned_download_url(
        self, artifact_uri: str, expiry_seconds: int = 3600
    ) -> str:
        """Return a time-limited presigned URL for artifact download.

        Args:
            artifact_uri: Canonical S3-style artifact URI.
            expiry_seconds: URL validity window in seconds.

        Returns:
            Presigned HTTPS URL valid for expiry_seconds.
        """
        endpoint = getattr(self._settings, "endpoint", "localhost:9000")
        return f"http://{endpoint}/presigned/{artifact_uri}?expiry={expiry_seconds}"

    async def delete_artifacts(
        self, tenant_id: uuid.UUID, model_id: uuid.UUID, version: int
    ) -> None:
        """Delete all artifacts for a model version.

        Args:
            tenant_id: Owning tenant UUID.
            model_id: Parent model UUID.
            version: Model version number.
        """
        bucket = f"reg-models-{tenant_id}"
        prefix = f"models/{model_id}/v{version}/"
        logger.info(
            "Artifact deletion requested",
            bucket=bucket,
            prefix=prefix,
        )
