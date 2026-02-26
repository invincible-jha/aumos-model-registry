"""CycloneDX ML Bill of Materials (ML-BOM) generator.

Produces a CycloneDX 1.5 compliant JSON document describing a model version's
training data, dependencies, hyperparameters, and performance characteristics.

Spec reference: https://cyclonedx.org/specification/overview/
ML extensions: https://cyclonedx.org/capabilities/mlbom/
"""

import uuid
from datetime import datetime, timezone

from aumos_model_registry.core.models import ModelVersion


def _make_serial_number() -> str:
    """Generate a CycloneDX-style URN serial number."""
    return f"urn:uuid:{uuid.uuid4()}"


def _training_data_components(training_data: dict) -> list[dict]:
    """Convert training data provenance dict into CycloneDX data components."""
    components = []
    datasets = training_data.get("datasets", [])
    if not isinstance(datasets, list):
        datasets = [training_data]

    for ds in datasets:
        component: dict = {
            "type": "data",
            "name": ds.get("name", "unknown-dataset"),
            "version": ds.get("version", "1.0.0"),
        }
        if ds.get("source"):
            component["externalReferences"] = [
                {"type": "distribution", "url": ds["source"]}
            ]
        if ds.get("records"):
            component["data"] = {"records": ds["records"]}
        if ds.get("license"):
            component["licenses"] = [{"license": {"name": ds["license"]}}]
        components.append(component)

    return components


def _hyperparameter_properties(hyperparameters: dict) -> list[dict]:
    """Convert hyperparameters dict to CycloneDX property list."""
    return [
        {"name": f"aumos:hyperparameter:{key}", "value": str(value)}
        for key, value in hyperparameters.items()
    ]


def _metrics_properties(metrics: dict) -> list[dict]:
    """Convert evaluation metrics dict to CycloneDX property list."""
    return [
        {"name": f"aumos:metric:{key}", "value": str(value)}
        for key, value in metrics.items()
    ]


def generate_ml_bom(
    model_version: ModelVersion,
    model_name: str,
    tenant_id: uuid.UUID,
) -> dict:
    """Generate a CycloneDX 1.5 ML-BOM for a model version.

    The BOM captures:
    - Model identity (name, version, stage)
    - Training data provenance as data components
    - Hyperparameters and evaluation metrics as properties
    - Artifact location as external reference
    - Lineage pointer to parent model if fine-tuned

    Args:
        model_version: The SQLAlchemy ModelVersion ORM instance.
        model_name: Human-readable name of the parent model.
        tenant_id: Owning tenant UUID (included in metadata).

    Returns:
        A dict conforming to CycloneDX 1.5 JSON schema for ML models.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # Primary model component
    primary_component: dict = {
        "type": "machine-learning-model",
        "name": model_name,
        "version": str(model_version.version),
        "bom-ref": f"model:{model_version.id}",
        "description": f"Stage: {model_version.stage}",
        "properties": [],
    }

    if model_version.hyperparameters:
        primary_component["properties"].extend(
            _hyperparameter_properties(model_version.hyperparameters)
        )

    if model_version.metrics:
        primary_component["properties"].extend(
            _metrics_properties(model_version.metrics)
        )

    if model_version.training_cost is not None:
        primary_component["properties"].append(
            {
                "name": "aumos:cost:training_usd",
                "value": str(model_version.training_cost),
            }
        )

    if model_version.size_bytes is not None:
        primary_component["properties"].append(
            {"name": "aumos:artifact:size_bytes", "value": str(model_version.size_bytes)}
        )

    # Artifact external reference
    external_references: list[dict] = []
    if model_version.artifact_uri:
        external_references.append(
            {
                "type": "distribution",
                "url": model_version.artifact_uri,
                "comment": "Model artifact (weights, config, tokenizer)",
            }
        )

    if external_references:
        primary_component["externalReferences"] = external_references

    # Training data sub-components
    data_components: list[dict] = []
    if model_version.training_data:
        data_components = _training_data_components(model_version.training_data)

    # Lineage / parent model dependency
    dependencies: list[dict] = []
    if model_version.parent_model_id:
        parent_ref = f"model:{model_version.parent_model_id}"
        dependencies.append(
            {
                "ref": f"model:{model_version.id}",
                "dependsOn": [parent_ref],
            }
        )

    bom: dict = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": _make_serial_number(),
        "version": 1,
        "metadata": {
            "timestamp": now_iso,
            "tools": [
                {
                    "vendor": "AumOS Enterprise",
                    "name": "aumos-model-registry",
                    "version": "0.1.0",
                }
            ],
            "component": primary_component,
            "properties": [
                {"name": "aumos:tenant_id", "value": str(tenant_id)},
                {"name": "aumos:model_version_id", "value": str(model_version.id)},
            ],
        },
        "components": data_components,
        "dependencies": dependencies,
    }

    return bom
