"""Model lineage graph adapter — tracks training provenance and derivation chains.

Builds directed acyclic graphs (DAGs) representing how models were derived from
one another: base model → fine-tuned variants → deployed versions. Integrates
with the reg_model_versions table to extract lineage from the parent_version_id
and training_data_uri fields.

Lineage graph nodes represent model versions. Edges represent:
- TRAINED_FROM: version derived by fine-tuning from another version
- DERIVED_FROM: version derived by distillation, pruning, or quantisation
- EVALUATED_ON: version evaluated using a specific dataset

The graph is serialised as adjacency list JSON compatible with D3.js force graphs
and Cytoscape.js for frontend visualisation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class LineageEdgeType:
    """Canonical edge type labels for the lineage graph."""

    TRAINED_FROM = "trained_from"
    DERIVED_FROM = "derived_from"
    EVALUATED_ON = "evaluated_on"
    DEPLOYED_AS = "deployed_as"


@dataclass
class LineageNode:
    """A single node in the model lineage graph.

    Attributes:
        node_id: Unique node identifier (model version UUID as string).
        model_id: Parent model UUID.
        version: Version tag string (e.g., 'v1', 'v2.3').
        stage: Lifecycle stage (development, staging, production, archived).
        framework: ML framework used (pytorch, tensorflow, sklearn, etc.).
        created_at: ISO-8601 timestamp of version creation.
        training_data_uri: URI of the training dataset used.
        metrics: Dict of evaluation metrics (accuracy, f1, etc.).
        tags: Arbitrary metadata tags.
    """

    node_id: str
    model_id: str
    version: str
    stage: str
    framework: str | None
    created_at: str
    training_data_uri: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class LineageEdge:
    """A directed edge in the model lineage graph.

    Attributes:
        source_id: Source node ID (parent version UUID).
        target_id: Target node ID (derived version UUID).
        edge_type: Type of relationship (trained_from, derived_from, etc.).
        metadata: Optional edge metadata (fine-tuning config, dataset hash, etc.).
    """

    source_id: str
    target_id: str
    edge_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageGraph:
    """Complete lineage graph for a model family.

    Attributes:
        root_model_id: UUID of the root model in the lineage.
        nodes: All version nodes in the graph.
        edges: All directed edges between nodes.
        depth: Maximum depth of the derivation chain.
        total_versions: Total number of versions in the graph.
    """

    root_model_id: str
    nodes: list[LineageNode]
    edges: list[LineageEdge]
    depth: int
    total_versions: int

    def to_d3_format(self) -> dict[str, Any]:
        """Serialise the graph in D3.js force-directed graph format.

        Returns:
            Dict with 'nodes' and 'links' arrays suitable for D3.js.
        """
        return {
            "nodes": [
                {
                    "id": node.node_id,
                    "model_id": node.model_id,
                    "version": node.version,
                    "stage": node.stage,
                    "framework": node.framework,
                    "created_at": node.created_at,
                    "training_data_uri": node.training_data_uri,
                    "metrics": node.metrics,
                    "tags": node.tags,
                    "group": _stage_to_group(node.stage),
                }
                for node in self.nodes
            ],
            "links": [
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.edge_type,
                    "metadata": edge.metadata,
                }
                for edge in self.edges
            ],
            "metadata": {
                "root_model_id": self.root_model_id,
                "depth": self.depth,
                "total_versions": self.total_versions,
            },
        }

    def to_cytoscape_format(self) -> dict[str, Any]:
        """Serialise the graph in Cytoscape.js format.

        Returns:
            Dict with 'elements' containing 'nodes' and 'edges' arrays.
        """
        return {
            "elements": {
                "nodes": [
                    {
                        "data": {
                            "id": node.node_id,
                            "label": f"{node.version} ({node.stage})",
                            "stage": node.stage,
                            "framework": node.framework or "unknown",
                            "metrics": node.metrics,
                        }
                    }
                    for node in self.nodes
                ],
                "edges": [
                    {
                        "data": {
                            "id": f"{edge.source_id}->{edge.target_id}",
                            "source": edge.source_id,
                            "target": edge.target_id,
                            "type": edge.edge_type,
                        }
                    }
                    for edge in self.edges
                ],
            }
        }


def _stage_to_group(stage: str) -> int:
    """Map lifecycle stage to numeric D3 group for colour coding.

    Args:
        stage: Stage string (development, staging, production, archived).

    Returns:
        Integer group number for D3 colour scale.
    """
    return {"development": 1, "staging": 2, "production": 3, "archived": 4}.get(stage, 0)


class ModelLineageGraphBuilder:
    """Builds and queries model lineage graphs from version records.

    The builder traverses parent_version_id chains to construct the full
    derivation DAG for a model family. It also supports injecting external
    lineage edges (e.g., from MLflow or aumos-data-layer dataset records).

    Args:
        max_depth: Maximum traversal depth to prevent infinite loops in
            misconfigured lineage chains (default 20).
    """

    def __init__(self, max_depth: int = 20) -> None:
        """Initialise the lineage graph builder.

        Args:
            max_depth: Maximum edge hops to traverse before stopping.
        """
        self._max_depth = max_depth

    def build_from_versions(
        self,
        root_model_id: uuid.UUID,
        versions: list[dict[str, Any]],
    ) -> LineageGraph:
        """Build a lineage graph from a list of version records.

        Traverses parent_version_id links to construct directed edges.
        Versions without a parent_version_id are treated as root nodes.

        Args:
            root_model_id: UUID of the model owning these versions.
            versions: List of version dicts with at least: id, version,
                stage, parent_version_id, framework, created_at,
                training_data_uri, performance_metrics, tags.

        Returns:
            LineageGraph with all nodes and parent-child edges.
        """
        nodes: list[LineageNode] = []
        edges: list[LineageEdge] = []
        version_index: dict[str, dict[str, Any]] = {}

        for version in versions:
            version_id = str(version["id"])
            version_index[version_id] = version
            nodes.append(
                LineageNode(
                    node_id=version_id,
                    model_id=str(root_model_id),
                    version=version.get("version", "unknown"),
                    stage=version.get("stage", "development"),
                    framework=version.get("framework"),
                    created_at=str(version.get("created_at", "")),
                    training_data_uri=version.get("training_data_uri"),
                    metrics=version.get("performance_metrics") or {},
                    tags=version.get("tags") or {},
                )
            )

        for version in versions:
            parent_id = version.get("parent_version_id")
            if parent_id and str(parent_id) in version_index:
                edges.append(
                    LineageEdge(
                        source_id=str(parent_id),
                        target_id=str(version["id"]),
                        edge_type=LineageEdgeType.TRAINED_FROM,
                        metadata={"version_tag": version.get("version", "")},
                    )
                )

        depth = self._compute_depth(nodes, edges)

        logger.info(
            "lineage_graph_built",
            root_model_id=str(root_model_id),
            node_count=len(nodes),
            edge_count=len(edges),
            depth=depth,
        )

        return LineageGraph(
            root_model_id=str(root_model_id),
            nodes=nodes,
            edges=edges,
            depth=depth,
            total_versions=len(nodes),
        )

    def add_cross_model_edge(
        self,
        graph: LineageGraph,
        source_version_id: str,
        target_version_id: str,
        edge_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> LineageGraph:
        """Add a cross-model derivation edge to an existing graph.

        Used when a model was derived from a version in a different model
        family (e.g., distillation from a large foundation model to a
        small task-specific model).

        Args:
            graph: Existing LineageGraph to augment.
            source_version_id: UUID of the source (parent) version.
            target_version_id: UUID of the target (derived) version.
            edge_type: Relationship type (typically DERIVED_FROM).
            metadata: Optional metadata dict.

        Returns:
            New LineageGraph with the additional edge appended.
        """
        new_edge = LineageEdge(
            source_id=source_version_id,
            target_id=target_version_id,
            edge_type=edge_type,
            metadata=metadata or {},
        )
        return LineageGraph(
            root_model_id=graph.root_model_id,
            nodes=graph.nodes,
            edges=[*graph.edges, new_edge],
            depth=graph.depth,
            total_versions=graph.total_versions,
        )

    def find_ancestors(
        self,
        graph: LineageGraph,
        version_id: str,
    ) -> list[str]:
        """Find all ancestor version IDs for a given version node.

        Traverses TRAINED_FROM and DERIVED_FROM edges in reverse to find
        all ancestors up to the root.

        Args:
            graph: LineageGraph to traverse.
            version_id: Starting node UUID string.

        Returns:
            List of ancestor node IDs in topological order (root first).
        """
        parent_map: dict[str, str] = {}
        for edge in graph.edges:
            if edge.edge_type in (LineageEdgeType.TRAINED_FROM, LineageEdgeType.DERIVED_FROM):
                parent_map[edge.target_id] = edge.source_id

        ancestors: list[str] = []
        current = version_id
        visited: set[str] = set()

        while current in parent_map and current not in visited:
            visited.add(current)
            parent = parent_map[current]
            ancestors.append(parent)
            current = parent

        ancestors.reverse()
        return ancestors

    def find_descendants(
        self,
        graph: LineageGraph,
        version_id: str,
    ) -> list[str]:
        """Find all descendant version IDs for a given version node.

        Traverses TRAINED_FROM and DERIVED_FROM edges forward to find
        all versions derived from the given node.

        Args:
            graph: LineageGraph to traverse.
            version_id: Starting node UUID string.

        Returns:
            List of descendant node IDs in breadth-first order.
        """
        children_map: dict[str, list[str]] = {}
        for edge in graph.edges:
            if edge.edge_type in (LineageEdgeType.TRAINED_FROM, LineageEdgeType.DERIVED_FROM):
                children_map.setdefault(edge.source_id, []).append(edge.target_id)

        descendants: list[str] = []
        queue = [version_id]
        visited: set[str] = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for child in children_map.get(current, []):
                descendants.append(child)
                queue.append(child)

        return descendants

    def _compute_depth(
        self,
        nodes: list[LineageNode],
        edges: list[LineageEdge],
    ) -> int:
        """Compute the maximum depth of the derivation chain.

        Args:
            nodes: All graph nodes.
            edges: All graph edges.

        Returns:
            Integer depth (longest path from any root to any leaf).
        """
        children_map: dict[str, list[str]] = {}
        has_parent: set[str] = set()

        for edge in edges:
            if edge.edge_type in (LineageEdgeType.TRAINED_FROM, LineageEdgeType.DERIVED_FROM):
                children_map.setdefault(edge.source_id, []).append(edge.target_id)
                has_parent.add(edge.target_id)

        node_ids = {node.node_id for node in nodes}
        roots = [node_id for node_id in node_ids if node_id not in has_parent]

        if not roots:
            return 0

        def _dfs(node_id: str, current_depth: int, visited: set[str]) -> int:
            if node_id in visited or current_depth >= self._max_depth:
                return current_depth
            visited = visited | {node_id}
            children = children_map.get(node_id, [])
            if not children:
                return current_depth
            return max(_dfs(child, current_depth + 1, visited) for child in children)

        return max(_dfs(root, 0, set()) for root in roots)
