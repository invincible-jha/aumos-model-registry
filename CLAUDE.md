# CLAUDE.md — AumOS Model Registry

## Project Overview

AumOS Enterprise is a composable enterprise AI platform with 9 products + 2 services
across 62 repositories. This repo (`aumos-model-registry`) is part of **Foundation Infrastructure**:
the shared services layer that all synthesis and analysis engines depend on for model
lifecycle management, artifact storage, cost tracking, and compliance.

**Release Tier:** A (Fully Open)
**Product Mapping:** Product 0 — Foundation Infrastructure
**Phase:** 1A (Months 1-4)

## Repo Purpose

The Model Registry is the canonical source of truth for all AI/ML model assets within
the AumOS platform. It tracks model registrations, version histories, training provenance,
deployment lifecycle (dev → staging → production → archived), ML Bill of Materials
(CycloneDX ML-BOM), cost attribution, and fine-tuning lineage graphs. Downstream synthesis
engines register their trained models here and query the registry for approved production
versions before serving inference traffic.

## Architecture Position

```
aumos-common ──► aumos-model-registry ──► aumos-tabular-engine
aumos-proto  ──►                       ──► aumos-text-engine
aumos-data-layer ──►                   ──► all synthesis engines
                    ↘ aumos-event-bus (MODEL_LIFECYCLE events)
                    ↘ MLflow tracking server (mirror via adapters/mlflow_client.py)
                    ↘ MinIO / S3 (artifact storage via adapters/minio_client.py)
```

**Upstream dependencies (this repo IMPORTS from):**
- `aumos-common` — auth, database, events, errors, config, health, pagination
- `aumos-proto` — Protobuf message definitions for Kafka events
- `aumos-data-layer` — PostgreSQL with pgvector and RLS policies

**Downstream dependents (other repos IMPORT from this):**
- `aumos-tabular-engine` — registers trained CTGAN/TVAESynth models
- `aumos-text-engine` — registers fine-tuned LLMs
- All synthesis engines — query for production-approved model versions

## Tech Stack (DO NOT DEVIATE)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API framework |
| SQLAlchemy | 2.0+ (async) | Database ORM |
| asyncpg | 0.29+ | PostgreSQL async driver |
| Pydantic | 2.6+ | Data validation, settings, API schemas |
| confluent-kafka | 2.3+ | Kafka producer/consumer |
| structlog | 24.1+ | Structured JSON logging |
| OpenTelemetry | 1.23+ | Distributed tracing |
| mlflow | 2.10+ | MLflow REST API integration (mirror only) |
| cyclonedx-python-lib | 6.0+ | CycloneDX ML-BOM generation |
| httpx | 0.27+ | Async HTTP client (MLflow, MinIO) |
| pytest | 8.0+ | Testing framework |
| ruff | 0.3+ | Linting and formatting |
| mypy | 1.8+ | Type checking |

## Coding Standards

### ABSOLUTE RULES (violations will break integration with other repos)

1. **Import aumos-common, never reimplement.** If aumos-common provides it, use it.
   ```python
   # CORRECT
   from aumos_common.auth import get_current_tenant, get_current_user
   from aumos_common.database import get_db_session, Base, AumOSModel, BaseRepository
   from aumos_common.events import EventPublisher, Topics
   from aumos_common.errors import NotFoundError, ErrorCode
   from aumos_common.config import AumOSSettings
   from aumos_common.health import create_health_router
   from aumos_common.app import create_app
   ```

2. **Type hints on EVERY function.** No exceptions.

3. **Pydantic models for ALL API inputs/outputs.** Never return raw dicts from endpoints.

4. **RLS tenant isolation via aumos-common.** Never write raw SQL that bypasses RLS.

5. **Structured logging via structlog.** Never use print() or logging.getLogger().

6. **Publish domain events to Kafka after state changes.**

7. **Async by default.** All I/O operations must be async.

8. **Google-style docstrings** on all public classes and functions.

### Style Rules

- Max line length: **120 characters**
- Import order: stdlib → third-party → aumos-common → local
- Linter: `ruff` (select E, W, F, I, N, UP, ANN, B, A, COM, C4, PT, RUF)
- Type checker: `mypy` strict mode
- Formatter: `ruff format`

### File Structure Convention

```
src/aumos_model_registry/
├── __init__.py
├── main.py                   # FastAPI app entry point
├── settings.py               # Extends AumOSSettings
├── api/
│   ├── __init__.py
│   ├── router.py             # All API endpoints
│   └── schemas.py            # Pydantic request/response models
├── core/
│   ├── __init__.py
│   ├── models.py             # SQLAlchemy ORM models (reg_ prefix)
│   ├── services.py           # ModelService, ExperimentService
│   ├── cost_engine.py        # Cost attribution calculations
│   ├── ml_bom.py             # CycloneDX ML-BOM generator
│   └── interfaces.py         # Protocol interfaces
└── adapters/
    ├── __init__.py
    ├── repositories.py       # SQLAlchemy repositories
    ├── kafka.py              # ModelRegistryEventPublisher
    ├── mlflow_client.py      # MLflow REST API client
    └── minio_client.py       # MinIO artifact storage client
```

## API Conventions

- All endpoints under `/api/v1/` prefix
- Auth: Bearer JWT token (validated by aumos-common)
- Tenant: `X-Tenant-ID` header (set by auth middleware)
- Request ID: `X-Request-ID` header (auto-generated if missing)
- Pagination: `?page=1&page_size=20`
- Errors: Standard `ErrorResponse` from aumos-common
- Content-Type: `application/json` (always)

## Database Conventions

- Table prefix: `reg_` (e.g., `reg_models`, `reg_model_versions`)
- ALL tenant-scoped tables: extend `AumOSModel` (gets id, tenant_id, created_at, updated_at)
- RLS policy on every tenant table (created in migration)
- Migration naming: `{timestamp}_reg_{description}.py`
- Foreign keys to other repos' tables: use UUID type, no FK constraints (cross-service)

## Kafka Conventions

- Publish to `Topics.MODEL_LIFECYCLE` for all model lifecycle events
- Event types: `model.registered`, `model.version_created`, `model.deployed`,
  `model.stage_staging`, `model.stage_production`, `model.deprecated`, `model.retired`
- Always include `tenant_id`, `model_id` in every event payload

## Repo-Specific Context

### MLflow Integration
- MLflow is used as an **optional mirror**, not the source of truth. AumOS registry is canonical.
- `adapters/mlflow_client.py` provides async HTTP calls to a sidecar MLflow tracking server
- Set `AUMOS_REGISTRY_MLFLOW_TRACKING_URI` to enable; if unset, MLflow sync is disabled
- MLflow stage names map: development=None, staging=Staging, production=Production, archived=Archived

### CycloneDX ML-BOM Generation
- BOM is generated in `core/ml_bom.py` using CycloneDX 1.5 JSON schema
- BOMs are generated on version creation (if `generate_bom=True`) and stored in `reg_model_versions.ml_bom`
- If no BOM exists at query time, one is generated on-the-fly and persisted
- BOM captures: model identity, training data provenance, hyperparameters, metrics, artifact URI, lineage

### Cost Attribution
- Three cost dimensions: training (recorded at version creation), inference (accumulated from deployments), storage (derived from `size_bytes`)
- All monetary values use `Decimal` with 2 decimal places — never float
- Default rates configurable via `AUMOS_REGISTRY_DEFAULT_GPU_HOURLY_COST_USD` etc.

### Semantic Search
- Current implementation: PostgreSQL ILIKE-based full-text search on name, description, tags
- Planned: pgvector cosine similarity on `embedding_dimensions=1536` vector column
- Embeddings will be generated using `embedding_model=text-embedding-3-small` via AumOS AI gateway

### Stage Lifecycle
- Valid transitions: development → staging → production → archived
- Reverse transitions are blocked except staging → development (for rollback)
- Production transitions require `require_approval_for_production=True` guard (enforced in service layer)

### Artifact Storage
- Artifacts stored in MinIO buckets: `reg-models-{tenant_id}`
- Object key format: `models/{model_id}/v{version}/{filename}`
- Set `AUMOS_REGISTRY_ARTIFACT_STORE` to override MinIO endpoint

## What Claude Code Should NOT Do

1. **Do NOT reimplement anything in aumos-common.**
2. **Do NOT use print().** Use `get_logger(__name__)`.
3. **Do NOT return raw dicts from API endpoints.** Use Pydantic models.
4. **Do NOT write raw SQL.** Use SQLAlchemy ORM with BaseRepository.
5. **Do NOT hardcode configuration.** Use Pydantic Settings with env vars.
6. **Do NOT skip type hints.** Every function signature must be typed.
7. **Do NOT import AGPL/GPL licensed packages** without explicit legal approval.
8. **Do NOT put business logic in API routes.** Routes call services; services contain logic.
9. **Do NOT use float for monetary values.** Always use `Decimal`.
10. **Do NOT bypass RLS.** Cross-tenant queries require `get_db_session_no_tenant`.

---

# IDENTIFIER REFERENCE

# REPO_DISPLAY_NAME:          Model Registry
# repo-name:                  model-registry
# repo_name_underscored:      model_registry
# TIER_NAME:                  Foundation Infrastructure
# Release Tier:               A (Fully Open)
# REPO_PREFIX:                REGISTRY
# repo_3letter_prefix:        reg
