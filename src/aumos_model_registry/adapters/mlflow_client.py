"""MLflow REST API client for the AumOS Model Registry.

Provides an async HTTP client wrapping MLflow's Model Registry REST API.
Used as an optional integration to mirror AumOS registry entries into an
MLflow tracking server for experiment tracking and artifact lineage.

MLflow REST API docs:
https://mlflow.org/docs/latest/rest-api.html
"""

from typing import Any

import httpx
from aumos_common.observability import get_logger

logger = get_logger(__name__)

_MLFLOW_API_VERSION = "2.0"


class MLflowClient:
    """Async HTTP client for the MLflow Model Registry REST API.

    Mirrors model registrations, version creation, and stage transitions
    into an external MLflow tracking server. All operations are optional
    and failures are logged but not raised to the caller by default.
    """

    def __init__(
        self,
        tracking_uri: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        """Initialise the MLflow client.

        Args:
            tracking_uri: Base URI of the MLflow tracking server
                (e.g., http://mlflow:5000).
            timeout_seconds: Per-request HTTP timeout in seconds.
        """
        self._base_url = tracking_uri.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "MLflowClient":
        """Open the underlying HTTP client session."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"Content-Type": "application/json"},
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Close the underlying HTTP client session."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _client_or_raise(self) -> httpx.AsyncClient:
        """Return the active HTTP client or raise if not initialised.

        Returns:
            Active httpx.AsyncClient instance.

        Raises:
            RuntimeError: If the client has not been opened via async context manager.
        """
        if self._client is None:
            raise RuntimeError(
                "MLflowClient must be used as an async context manager. "
                "Use `async with MLflowClient(...) as client:`"
            )
        return self._client

    async def register_model(
        self,
        name: str,
        source: str,
        tags: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Register a new model in the MLflow Model Registry.

        Creates the registered model entry. If the model already exists,
        returns the existing record without error.

        Args:
            name: Unique model name in the MLflow registry.
            source: Artifact source URI (S3/MinIO path).
            tags: Optional dict of string key/value tags.

        Returns:
            MLflow RegisteredModel response dict.
        """
        client = self._client_or_raise()
        payload: dict[str, Any] = {"name": name}
        if tags:
            payload["tags"] = [{"key": k, "value": v} for k, v in tags.items()]

        response = await client.post(
            f"/api/{_MLFLOW_API_VERSION}/mlflow/registered-models/create",
            json=payload,
        )

        if response.status_code == 400:
            # RESOURCE_ALREADY_EXISTS — return existing record
            existing = await client.get(
                f"/api/{_MLFLOW_API_VERSION}/mlflow/registered-models/get",
                params={"name": name},
            )
            existing.raise_for_status()
            return existing.json().get("registered_model", {})

        response.raise_for_status()
        result: dict[str, Any] = response.json().get("registered_model", {})
        logger.info("Model registered in MLflow", name=name)
        return result

    async def create_model_version(
        self,
        name: str,
        source: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new version for a registered model in MLflow.

        Args:
            name: Registered model name in MLflow.
            source: Artifact source URI (S3/MinIO path to model weights).
            run_id: Optional MLflow run ID that produced this version.

        Returns:
            MLflow ModelVersion response dict.
        """
        client = self._client_or_raise()
        payload: dict[str, Any] = {"name": name, "source": source}
        if run_id is not None:
            payload["run_id"] = run_id

        response = await client.post(
            f"/api/{_MLFLOW_API_VERSION}/mlflow/model-versions/create",
            json=payload,
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json().get("model_version", {})
        logger.info(
            "Model version created in MLflow",
            name=name,
            version=result.get("version"),
        )
        return result

    async def transition_stage(
        self,
        name: str,
        version: str,
        stage: str,
    ) -> dict[str, Any]:
        """Transition a model version to a new lifecycle stage in MLflow.

        Args:
            name: Registered model name.
            version: Version number as a string.
            stage: Target MLflow stage: Staging | Production | Archived | None.

        Returns:
            Updated MLflow ModelVersion response dict.
        """
        client = self._client_or_raise()
        payload = {
            "name": name,
            "version": version,
            "stage": stage,
            "archive_existing_versions": False,
        }
        response = await client.post(
            f"/api/{_MLFLOW_API_VERSION}/mlflow/model-versions/transition-stage",
            json=payload,
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json().get("model_version", {})
        logger.info(
            "Model stage transitioned in MLflow",
            name=name,
            version=version,
            stage=stage,
        )
        return result

    async def search_models(
        self, filter_string: str = "", max_results: int = 100
    ) -> list[dict[str, Any]]:
        """Search for registered models matching a filter expression.

        Args:
            filter_string: MLflow filter string (e.g., "name LIKE 'bert%'").
            max_results: Maximum number of results to return.

        Returns:
            List of MLflow RegisteredModel dicts.
        """
        client = self._client_or_raise()
        params: dict[str, Any] = {"max_results": max_results}
        if filter_string:
            params["filter"] = filter_string

        response = await client.get(
            f"/api/{_MLFLOW_API_VERSION}/mlflow/registered-models/search",
            params=params,
        )
        response.raise_for_status()
        result: list[dict[str, Any]] = response.json().get("registered_models", [])
        return result

    async def get_latest_versions(
        self,
        name: str,
        stages: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get the latest model versions for a registered model.

        Args:
            name: Registered model name.
            stages: Optional list of stages to filter by (e.g., ["Production"]).

        Returns:
            List of latest ModelVersion dicts per stage.
        """
        client = self._client_or_raise()
        payload: dict[str, Any] = {"name": name}
        if stages:
            payload["stages"] = stages

        response = await client.post(
            f"/api/{_MLFLOW_API_VERSION}/mlflow/registered-models/get-latest-versions",
            json=payload,
        )
        response.raise_for_status()
        result: list[dict[str, Any]] = response.json().get("model_versions", [])
        return result
