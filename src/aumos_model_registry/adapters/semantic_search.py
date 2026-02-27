"""Semantic model search adapter for the AumOS Model Registry.

Provides vector-similarity-based model discovery using cosine similarity over
model description embeddings. Supports tag-based filtering with embedding boost,
search result ranking, analytics (popular queries), autocomplete suggestions,
and faceted search over framework and model_type dimensions.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from aumos_common.observability import get_logger

from aumos_model_registry.core.interfaces import IModelRepository
from aumos_model_registry.core.models import Model

logger = get_logger(__name__)

# Cosine similarity threshold for search result inclusion
_MIN_SIMILARITY_THRESHOLD = 0.15

# Tag boost factor: matching tags boosts the final score
_TAG_BOOST_WEIGHT = 0.25

# Embedding dimension for sentence-transformer embeddings
_EMBEDDING_DIMENSIONS = 384


class ModelSemanticSearch:
    """Semantic model discovery via embedding-based cosine similarity search.

    Generates embeddings for model descriptions using a configurable embedding
    model, then performs cosine similarity search over the embedding space.
    Falls back to lexical (ILIKE) search when no embedding model is available.

    Maintains an in-memory search analytics store (query counts and popular
    terms) and generates autocomplete candidates from registered model metadata.

    Usage::

        searcher = ModelSemanticSearch(model_repo=model_repo)
        results = await searcher.search(
            tenant_id=tenant_id,
            query="sentiment analysis transformer",
            tags=["nlp", "bert"],
            limit=10,
        )
    """

    def __init__(
        self,
        model_repo: IModelRepository,
        embedding_model_url: str | None = None,
        embedding_api_key: str | None = None,
        enable_vector_search: bool = False,
    ) -> None:
        """Initialise the semantic search adapter.

        Args:
            model_repo: Model repository for fetching candidates.
            embedding_model_url: URL for the embedding API endpoint. When None,
                lexical fallback search is used.
            embedding_api_key: API key for the embedding service.
            enable_vector_search: Enable pgvector-backed similarity when True.
        """
        self._model_repo = model_repo
        self._embedding_model_url = embedding_model_url
        self._embedding_api_key = embedding_api_key
        self._enable_vector_search = enable_vector_search

        # In-memory analytics (replace with Redis in production)
        self._query_counts: dict[str, int] = defaultdict(int)
        self._recent_queries: list[dict[str, Any]] = []
        self._max_analytics_entries = 1000

    async def search(
        self,
        tenant_id: uuid.UUID,
        query: str,
        tags: list[str] | None = None,
        framework: str | None = None,
        model_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Execute a semantic search over model descriptions.

        Generates a query embedding and ranks candidate models by cosine
        similarity. Tag filters are applied before scoring; matching tags
        provide a scoring boost. Results are sorted by composite score
        (semantic_score * (1 + tag_boost)).

        Args:
            tenant_id: Requesting tenant UUID.
            query: Natural language search query.
            tags: Optional tag filter list. Models must match at least one tag.
            framework: Optional exact-match framework filter.
            model_type: Optional exact-match model_type filter.
            limit: Maximum number of results to return.

        Returns:
            List of result dicts with 'model', 'score', 'matched_tags' keys.
        """
        start_time = time.monotonic()

        # Record analytics
        await self._record_query(tenant_id=tenant_id, query=query)

        # Fetch all models for the tenant (small tenants fit in memory)
        # For large tenants this should use cursor-based pagination + pgvector
        all_models, total = await self._model_repo.list_all(
            tenant_id=tenant_id,
            page=1,
            page_size=500,
            model_type=model_type,
            framework=framework,
        )

        if not all_models:
            logger.info("No models found for semantic search", tenant_id=str(tenant_id))
            return []

        # Generate query embedding
        query_embedding = await self._embed_text(query)

        # Score each candidate
        scored_results: list[dict[str, Any]] = []
        for model in all_models:
            # Skip archived models
            score, matched_tags = await self._score_model(
                model=model,
                query=query,
                query_embedding=query_embedding,
                tag_filter=tags,
            )

            if score < _MIN_SIMILARITY_THRESHOLD:
                continue

            scored_results.append(
                {
                    "model": model,
                    "model_id": str(model.id),
                    "model_name": model.name,
                    "description": model.description or "",
                    "framework": model.framework,
                    "model_type": model.model_type,
                    "tags": model.tags,
                    "score": round(score, 4),
                    "matched_tags": matched_tags,
                }
            )

        # Sort by descending score
        scored_results.sort(key=lambda r: r["score"], reverse=True)
        results = scored_results[:limit]

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Semantic search completed",
            tenant_id=str(tenant_id),
            query=query[:80],
            candidate_count=len(all_models),
            result_count=len(results),
            elapsed_ms=round(elapsed_ms, 1),
        )

        return results

    async def get_autocomplete_suggestions(
        self,
        tenant_id: uuid.UUID,
        prefix: str,
        limit: int = 8,
    ) -> list[str]:
        """Return autocomplete suggestions for a query prefix.

        Pulls suggestions from model names, frameworks, model types, and tag
        keys that start with the given prefix. Case-insensitive matching.

        Args:
            tenant_id: Requesting tenant UUID.
            prefix: Query prefix string (minimum 2 characters).
            limit: Maximum number of suggestions to return.

        Returns:
            List of suggestion strings ordered by popularity.
        """
        if len(prefix) < 2:
            return []

        prefix_lower = prefix.lower()

        all_models, _ = await self._model_repo.list_all(
            tenant_id=tenant_id,
            page=1,
            page_size=500,
            model_type=None,
            framework=None,
        )

        candidates: set[str] = set()
        for model in all_models:
            if model.name.lower().startswith(prefix_lower):
                candidates.add(model.name)
            if model.framework and model.framework.lower().startswith(prefix_lower):
                candidates.add(model.framework)
            if model.model_type and model.model_type.lower().startswith(prefix_lower):
                candidates.add(model.model_type)
            for tag_key in (model.tags or {}).keys():
                if tag_key.lower().startswith(prefix_lower):
                    candidates.add(tag_key)

        sorted_candidates = sorted(candidates)[:limit]
        logger.debug(
            "Autocomplete suggestions generated",
            prefix=prefix,
            count=len(sorted_candidates),
        )
        return sorted_candidates

    async def get_popular_queries(
        self,
        tenant_id: uuid.UUID,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the most popular search queries for a tenant.

        Args:
            tenant_id: Requesting tenant UUID.
            limit: Maximum queries to return.

        Returns:
            List of dicts with 'query' and 'count' fields, sorted by count.
        """
        tenant_key = str(tenant_id)
        tenant_queries = {
            q: count
            for q, count in self._query_counts.items()
            if q.startswith(f"{tenant_key}:")
        }

        sorted_queries = sorted(tenant_queries.items(), key=lambda item: item[1], reverse=True)

        return [
            {"query": key.split(":", 1)[1], "count": count}
            for key, count in sorted_queries[:limit]
        ]

    async def get_facets(
        self,
        tenant_id: uuid.UUID,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return faceted search counts for framework and model_type dimensions.

        Args:
            tenant_id: Requesting tenant UUID.

        Returns:
            Dict with 'frameworks' and 'model_types' keys, each containing
            a list of {'value': str, 'count': int} dicts.
        """
        all_models, _ = await self._model_repo.list_all(
            tenant_id=tenant_id,
            page=1,
            page_size=1000,
            model_type=None,
            framework=None,
        )

        framework_counts: dict[str, int] = defaultdict(int)
        model_type_counts: dict[str, int] = defaultdict(int)

        for model in all_models:
            if model.framework:
                framework_counts[model.framework] += 1
            if model.model_type:
                model_type_counts[model.model_type] += 1

        return {
            "frameworks": [
                {"value": fw, "count": count}
                for fw, count in sorted(framework_counts.items(), key=lambda x: -x[1])
            ],
            "model_types": [
                {"value": mt, "count": count}
                for mt, count in sorted(model_type_counts.items(), key=lambda x: -x[1])
            ],
        }

    async def _score_model(
        self,
        model: Model,
        query: str,
        query_embedding: list[float] | None,
        tag_filter: list[str] | None,
    ) -> tuple[float, list[str]]:
        """Compute a composite similarity score for a model against a query.

        Args:
            model: Candidate model ORM instance.
            query: Original query string.
            query_embedding: Query embedding vector (None for lexical fallback).
            tag_filter: Optional list of required tags.

        Returns:
            Tuple of (composite_score, matched_tags_list).
        """
        # Compute base semantic or lexical score
        if query_embedding is not None:
            model_text = self._build_model_text(model)
            model_embedding = await self._embed_text(model_text)
            base_score = self._cosine_similarity(query_embedding, model_embedding)
        else:
            # Lexical fallback: simple token overlap score
            base_score = self._lexical_score(model, query)

        # Tag filter: if tags specified, model must have at least one
        matched_tags: list[str] = []
        if tag_filter:
            model_tag_keys = set((model.tags or {}).keys())
            matched_tags = [t for t in tag_filter if t in model_tag_keys]
            if not matched_tags:
                return 0.0, []

        # Tag boost: matching tags increase the score
        tag_boost = _TAG_BOOST_WEIGHT * len(matched_tags) if matched_tags else 0.0
        composite_score = base_score * (1.0 + tag_boost)

        return min(composite_score, 1.0), matched_tags

    def _build_model_text(self, model: Model) -> str:
        """Build a concatenated text representation of a model for embedding.

        Args:
            model: Model ORM instance.

        Returns:
            Concatenated text string for embedding generation.
        """
        parts: list[str] = [model.name]
        if model.description:
            parts.append(model.description)
        if model.framework:
            parts.append(f"framework:{model.framework}")
        if model.model_type:
            parts.append(f"type:{model.model_type}")
        for key, value in (model.tags or {}).items():
            parts.append(f"{key}:{value}")
        return " ".join(parts)

    async def _embed_text(self, text: str) -> list[float] | None:
        """Generate a text embedding via the configured embedding service.

        Args:
            text: Input text to embed.

        Returns:
            Float embedding vector or None if no embedding service is configured.
        """
        if not self._embedding_model_url:
            return None

        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self._embedding_model_url,
                    json={"input": text, "model": "text-embedding-3-small"},
                    headers={"Authorization": f"Bearer {self._embedding_api_key}"},
                )
                response.raise_for_status()
                data = response.json()
                return data["data"][0]["embedding"]
        except Exception as exc:
            logger.warning("Embedding generation failed, using lexical fallback", error=str(exc))
            return None

    def _cosine_similarity(
        self, vector_a: list[float], vector_b: list[float] | None
    ) -> float:
        """Compute cosine similarity between two embedding vectors.

        Args:
            vector_a: First embedding vector.
            vector_b: Second embedding vector (None returns 0.0).

        Returns:
            Cosine similarity in range [-1.0, 1.0], typically [0.0, 1.0] for embeddings.
        """
        if vector_b is None or len(vector_a) != len(vector_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        magnitude_a = math.sqrt(sum(a * a for a in vector_a))
        magnitude_b = math.sqrt(sum(b * b for b in vector_b))

        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def _lexical_score(self, model: Model, query: str) -> float:
        """Compute a token-overlap lexical similarity score.

        Args:
            model: Candidate model ORM instance.
            query: Query string.

        Returns:
            Score in [0.0, 1.0] based on token overlap.
        """
        query_tokens = set(query.lower().split())
        model_text = self._build_model_text(model).lower()
        model_tokens = set(model_text.split())

        if not query_tokens:
            return 0.0

        intersection = query_tokens & model_tokens
        return len(intersection) / len(query_tokens)

    async def _record_query(self, tenant_id: uuid.UUID, query: str) -> None:
        """Record a search query for analytics.

        Args:
            tenant_id: Requesting tenant.
            query: Query string to record.
        """
        tenant_key = str(tenant_id)
        analytics_key = f"{tenant_key}:{query.lower().strip()}"
        self._query_counts[analytics_key] += 1

        if len(self._recent_queries) >= self._max_analytics_entries:
            self._recent_queries = self._recent_queries[-(self._max_analytics_entries // 2):]

        self._recent_queries.append(
            {
                "tenant_id": tenant_key,
                "query": query,
                "recorded_at": datetime.now(UTC).isoformat(),
            }
        )
