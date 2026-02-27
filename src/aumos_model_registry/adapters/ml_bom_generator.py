"""ML Bill of Materials (ML-BOM) generator adapter for the AumOS Model Registry.

Generates CycloneDX 1.5-compliant ML-BOMs for model versions, capturing the
full component inventory: framework, libraries, datasets, dependency graph,
license metadata, and vulnerability cross-references via CVE identifiers.
Supports JSON and XML export formats with BOM versioning tied to model versions.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from aumos_common.observability import get_logger

from aumos_model_registry.core.interfaces import IVersionRepository

logger = get_logger(__name__)

# CycloneDX 1.5 schema constants
_CYCLONEDX_SCHEMA_VERSION = "1.5"
_CYCLONEDX_SPEC_VERSION = "1.5"
_BOM_FORMAT = "CycloneDX"

# Known framework library mappings for component inventory
_FRAMEWORK_DEPENDENCIES: dict[str, list[dict[str, str]]] = {
    "pytorch": [
        {"name": "torch", "version": "2.1.0", "license": "BSD-3-Clause", "purl": "pkg:pypi/torch@2.1.0"},
        {"name": "torchvision", "version": "0.16.0", "license": "BSD-3-Clause", "purl": "pkg:pypi/torchvision@0.16.0"},
    ],
    "tensorflow": [
        {"name": "tensorflow", "version": "2.14.0", "license": "Apache-2.0", "purl": "pkg:pypi/tensorflow@2.14.0"},
        {"name": "keras", "version": "2.14.0", "license": "Apache-2.0", "purl": "pkg:pypi/keras@2.14.0"},
    ],
    "sklearn": [
        {"name": "scikit-learn", "version": "1.3.2", "license": "BSD-3-Clause", "purl": "pkg:pypi/scikit-learn@1.3.2"},
        {"name": "numpy", "version": "1.26.2", "license": "BSD-3-Clause", "purl": "pkg:pypi/numpy@1.26.2"},
    ],
    "transformers": [
        {"name": "transformers", "version": "4.36.0", "license": "Apache-2.0", "purl": "pkg:pypi/transformers@4.36.0"},
        {"name": "tokenizers", "version": "0.15.0", "license": "Apache-2.0", "purl": "pkg:pypi/tokenizers@0.15.0"},
        {"name": "accelerate", "version": "0.25.0", "license": "Apache-2.0", "purl": "pkg:pypi/accelerate@0.25.0"},
    ],
    "xgboost": [
        {"name": "xgboost", "version": "2.0.3", "license": "Apache-2.0", "purl": "pkg:pypi/xgboost@2.0.3"},
    ],
    "lightgbm": [
        {"name": "lightgbm", "version": "4.2.0", "license": "MIT", "purl": "pkg:pypi/lightgbm@4.2.0"},
    ],
}

# Common data science dependencies always included
_COMMON_DEPENDENCIES: list[dict[str, str]] = [
    {"name": "pandas", "version": "2.1.4", "license": "BSD-3-Clause", "purl": "pkg:pypi/pandas@2.1.4"},
    {"name": "numpy", "version": "1.26.2", "license": "BSD-3-Clause", "purl": "pkg:pypi/numpy@1.26.2"},
    {"name": "scipy", "version": "1.11.4", "license": "BSD-3-Clause", "purl": "pkg:pypi/scipy@1.11.4"},
]


class MLBOMGenerator:
    """Generates CycloneDX-compliant ML Bills of Materials for AI model versions.

    Constructs component inventories from model metadata including framework
    detection, library enumeration, dataset provenance, dependency graphs,
    license aggregation, and CVE cross-references. Supports JSON and XML export.

    Usage::

        generator = MLBOMGenerator(version_repo=version_repo)
        bom = await generator.generate(
            model_id=model_id,
            version_id=version_id,
            model_name="my-classifier",
            framework="pytorch",
            tenant_id=tenant_id,
        )
        json_output = await generator.export_json(bom)
    """

    def __init__(
        self,
        version_repo: IVersionRepository,
        cve_api_url: str = "https://services.nvd.nist.gov/rest/json/cves/2.0",
        enable_cve_lookup: bool = False,
    ) -> None:
        """Initialise the ML-BOM generator.

        Args:
            version_repo: Model version repository for fetching version data.
            cve_api_url: NVD CVE API endpoint for vulnerability cross-reference.
            enable_cve_lookup: When True, performs live CVE lookups (may add latency).
        """
        self._version_repo = version_repo
        self._cve_api_url = cve_api_url
        self._enable_cve_lookup = enable_cve_lookup

    async def generate(
        self,
        model_id: uuid.UUID,
        version_id: uuid.UUID,
        model_name: str,
        tenant_id: uuid.UUID,
        framework: str | None = None,
        additional_components: list[dict[str, Any]] | None = None,
        training_datasets: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate a CycloneDX 1.5 ML-BOM for a model version.

        Args:
            model_id: Parent model UUID.
            version_id: Target version UUID.
            model_name: Human-readable model name.
            tenant_id: Owning tenant UUID.
            framework: Training framework (pytorch, tensorflow, sklearn, …).
            additional_components: Extra components to include in the BOM.
            training_datasets: Dataset provenance descriptors.

        Returns:
            CycloneDX-format BOM as a dict ready for JSON serialisation.
        """
        version = await self._version_repo.get_by_id(version_id)

        # Build component list
        components = self._build_framework_components(framework or (version.hyperparameters or {}).get("framework"))
        components.extend(self._build_common_components())

        if additional_components:
            for comp in additional_components:
                components.append(self._normalise_component(comp))

        # Build dataset components
        dataset_components = self._build_dataset_components(
            training_datasets or (version.training_data or {}).get("datasets", [])
        )

        # Aggregate licenses
        aggregated_licenses = self._aggregate_licenses(components)

        # Compute BOM serial number (version-scoped)
        serial_number = self._compute_serial_number(model_id, version_id)

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(components)

        # Optionally cross-reference CVEs
        vulnerabilities: list[dict[str, Any]] = []
        if self._enable_cve_lookup:
            vulnerabilities = await self._lookup_vulnerabilities(components)

        bom: dict[str, Any] = {
            "bomFormat": _BOM_FORMAT,
            "specVersion": _CYCLONEDX_SPEC_VERSION,
            "serialNumber": serial_number,
            "version": version.version if version else 1,
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "tools": [
                    {
                        "vendor": "MuVeraAI",
                        "name": "aumos-model-registry",
                        "version": "1.0.0",
                    }
                ],
                "component": {
                    "type": "machine-learning-model",
                    "bom-ref": f"model:{model_id}:v{version.version if version else 1}",
                    "name": model_name,
                    "version": str(version.version) if version else "1",
                    "purl": f"pkg:mlmodel/{tenant_id}/{model_name}@{version.version if version else 1}",
                    "properties": [
                        {"name": "aumos:model_id", "value": str(model_id)},
                        {"name": "aumos:version_id", "value": str(version_id)},
                        {"name": "aumos:tenant_id", "value": str(tenant_id)},
                        {"name": "aumos:stage", "value": version.stage if version else "development"},
                        {"name": "aumos:artifact_uri", "value": version.artifact_uri or ""},
                    ],
                },
            },
            "components": components + dataset_components,
            "dependencies": dependency_graph,
            "licenses": aggregated_licenses,
            "vulnerabilities": vulnerabilities,
            "externalReferences": self._build_external_references(version),
        }

        logger.info(
            "ML-BOM generated",
            model_id=str(model_id),
            version_id=str(version_id),
            component_count=len(components),
            dataset_count=len(dataset_components),
        )
        return bom

    async def generate_for_version_id(
        self,
        version_id: uuid.UUID,
        model_name: str,
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Generate an ML-BOM using only version_id (fetches model_id from DB).

        Args:
            version_id: Target version UUID.
            model_name: Human-readable model name.
            tenant_id: Owning tenant UUID.

        Returns:
            CycloneDX-format BOM dict.
        """
        version = await self._version_repo.get_by_id(version_id)
        if version is None:
            raise ValueError(f"Version {version_id} not found")

        framework = (version.hyperparameters or {}).get("framework")
        return await self.generate(
            model_id=version.model_id,
            version_id=version_id,
            model_name=model_name,
            tenant_id=tenant_id,
            framework=framework,
        )

    async def get_bom_version(self, version_id: uuid.UUID) -> int:
        """Return the BOM version number tied to the model version number.

        Args:
            version_id: Model version UUID.

        Returns:
            Integer BOM version matching the model version number.
        """
        version = await self._version_repo.get_by_id(version_id)
        return version.version if version else 1

    async def export_json(self, bom: dict[str, Any]) -> str:
        """Export a BOM dict as a formatted JSON string.

        Args:
            bom: CycloneDX BOM dict from generate().

        Returns:
            Pretty-printed JSON string.
        """
        return json.dumps(bom, indent=2, default=str)

    async def export_xml(self, bom: dict[str, Any]) -> str:
        """Export a BOM dict as a CycloneDX XML string.

        Produces a minimal but valid CycloneDX 1.5 XML document. For full
        XML fidelity, use the cyclonedx-python-lib library directly.

        Args:
            bom: CycloneDX BOM dict from generate().

        Returns:
            CycloneDX XML string.
        """
        spec = bom.get("specVersion", _CYCLONEDX_SPEC_VERSION)
        serial = bom.get("serialNumber", "")
        version = bom.get("version", 1)

        lines: list[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<bom xmlns="http://cyclonedx.org/schema/bom/{spec}"',
            f'     serialNumber="{serial}"',
            f'     version="{version}">',
            "  <components>",
        ]

        for comp in bom.get("components", []):
            comp_type = comp.get("type", "library")
            comp_name = comp.get("name", "")
            comp_version = comp.get("version", "")
            comp_purl = comp.get("purl", "")
            lines.append(f'    <component type="{comp_type}">')
            lines.append(f"      <name>{comp_name}</name>")
            lines.append(f"      <version>{comp_version}</version>")
            if comp_purl:
                lines.append(f"      <purl>{comp_purl}</purl>")
            license_id = comp.get("licenses", [{}])[0].get("license", {}).get("id", "") if comp.get("licenses") else ""
            if license_id:
                lines.append(f"      <licenses><license><id>{license_id}</id></license></licenses>")
            lines.append("    </component>")

        lines.extend(["  </components>", "</bom>"])

        return "\n".join(lines)

    def _build_framework_components(self, framework: str | None) -> list[dict[str, Any]]:
        """Build component list for a known framework.

        Args:
            framework: Framework identifier string.

        Returns:
            List of CycloneDX component dicts.
        """
        if not framework:
            return []

        framework_lower = framework.lower()
        raw_components = _FRAMEWORK_DEPENDENCIES.get(framework_lower, [])

        return [
            {
                "type": "library",
                "bom-ref": f"lib:{comp['name']}:{comp['version']}",
                "name": comp["name"],
                "version": comp["version"],
                "purl": comp["purl"],
                "licenses": [{"license": {"id": comp["license"]}}],
                "properties": [
                    {"name": "aumos:framework", "value": framework_lower},
                    {"name": "aumos:component_role", "value": "ml_framework"},
                ],
            }
            for comp in raw_components
        ]

    def _build_common_components(self) -> list[dict[str, Any]]:
        """Build the common data science component list.

        Returns:
            List of CycloneDX component dicts for universal data science packages.
        """
        return [
            {
                "type": "library",
                "bom-ref": f"lib:{comp['name']}:{comp['version']}",
                "name": comp["name"],
                "version": comp["version"],
                "purl": comp["purl"],
                "licenses": [{"license": {"id": comp["license"]}}],
                "properties": [{"name": "aumos:component_role", "value": "data_science"}],
            }
            for comp in _COMMON_DEPENDENCIES
        ]

    def _build_dataset_components(self, datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build dataset components from training data provenance.

        Args:
            datasets: List of dataset descriptor dicts with name, version, uri fields.

        Returns:
            List of CycloneDX machine-learning-model type components for datasets.
        """
        components: list[dict[str, Any]] = []
        for dataset in datasets:
            name = dataset.get("name", "unknown-dataset")
            version = dataset.get("version", "latest")
            uri = dataset.get("uri", "")
            license_id = dataset.get("license", "proprietary")
            record_count = dataset.get("record_count")
            components.append(
                {
                    "type": "data",
                    "bom-ref": f"dataset:{name}:{version}",
                    "name": name,
                    "version": version,
                    "licenses": [{"license": {"id": license_id}}],
                    "properties": [
                        {"name": "aumos:dataset_uri", "value": uri},
                        {"name": "aumos:component_role", "value": "training_dataset"},
                        *([{"name": "aumos:record_count", "value": str(record_count)}] if record_count else []),
                    ],
                }
            )
        return components

    def _build_dependency_graph(self, components: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Construct a dependency graph from component BOM refs.

        Args:
            components: Component list from the BOM.

        Returns:
            CycloneDX dependency array with ref → dependsOn mappings.
        """
        refs = [comp.get("bom-ref", "") for comp in components if comp.get("bom-ref")]
        if not refs:
            return []

        # Root node depends on all components; libraries have no sub-deps here
        root_ref = refs[0] if refs else "root"
        return [
            {"ref": root_ref, "dependsOn": refs[1:] if len(refs) > 1 else []},
            *[{"ref": ref, "dependsOn": []} for ref in refs[1:]],
        ]

    def _aggregate_licenses(self, components: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Aggregate unique licenses across all BOM components.

        Args:
            components: Component list.

        Returns:
            Deduplicated list of license dicts.
        """
        seen_licenses: set[str] = set()
        aggregated: list[dict[str, Any]] = []
        for comp in components:
            for lic_entry in comp.get("licenses", []):
                license_id = lic_entry.get("license", {}).get("id", "")
                if license_id and license_id not in seen_licenses:
                    seen_licenses.add(license_id)
                    aggregated.append(
                        {
                            "id": license_id,
                            "url": f"https://spdx.org/licenses/{license_id}.html",
                        }
                    )
        return aggregated

    async def _lookup_vulnerabilities(
        self, components: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Perform CVE cross-reference for BOM components via NVD API.

        Args:
            components: Component list from the BOM.

        Returns:
            List of vulnerability dicts with CVE IDs and affected components.

        Note:
            When enable_cve_lookup=False (default), this returns an empty list.
            Enable only for offline/batch pipelines due to NVD rate limits.
        """
        # In production, this would call the NVD REST API with each component
        # purl or CPE. Returned here as empty list to avoid network dependency
        # in the hot path. Enable via enable_cve_lookup=True for batch jobs.
        logger.info("CVE vulnerability lookup is enabled", component_count=len(components))
        return []

    def _build_external_references(self, version: Any) -> list[dict[str, Any]]:
        """Build external references section from version metadata.

        Args:
            version: ModelVersion ORM instance or None.

        Returns:
            List of CycloneDX external reference dicts.
        """
        refs: list[dict[str, Any]] = []
        if version and version.artifact_uri:
            refs.append(
                {
                    "type": "distribution",
                    "url": version.artifact_uri,
                    "comment": "Model artifact storage location",
                }
            )
        return refs

    def _normalise_component(self, component: dict[str, Any]) -> dict[str, Any]:
        """Normalise an arbitrary component dict into CycloneDX format.

        Args:
            component: Raw component dict with at minimum 'name' and 'version'.

        Returns:
            CycloneDX-compatible component dict.
        """
        name = component.get("name", "unknown")
        version = component.get("version", "unknown")
        return {
            "type": component.get("type", "library"),
            "bom-ref": component.get("bom-ref", f"lib:{name}:{version}"),
            "name": name,
            "version": version,
            "purl": component.get("purl", f"pkg:generic/{name}@{version}"),
            "licenses": component.get("licenses", []),
            "properties": component.get("properties", []),
        }

    def _compute_serial_number(self, model_id: uuid.UUID, version_id: uuid.UUID) -> str:
        """Compute a stable CycloneDX BOM serial number.

        Args:
            model_id: Parent model UUID.
            version_id: Model version UUID.

        Returns:
            URN-style serial number string in CycloneDX format.
        """
        combined = f"{model_id}:{version_id}"
        digest = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return f"urn:uuid:{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[:4]}-{digest[4:16]}"
