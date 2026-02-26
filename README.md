# aumos-model-registry

[![CI](https://github.com/aumos-enterprise/aumos-model-registry/actions/workflows/ci.yml/badge.svg)](https://github.com/aumos-enterprise/aumos-model-registry/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/aumos-enterprise/aumos-model-registry/branch/main/graph/badge.svg)](https://codecov.io/gh/aumos-enterprise/aumos-model-registry)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> MLflow-backed AI/ML model lifecycle management with lineage tracking, cost attribution, and CycloneDX ML-BOM generation.

## Overview

The AumOS Model Registry is the canonical source of truth for all AI/ML model assets within
the AumOS enterprise platform. It provides a unified lifecycle management system that tracks
model registrations from training through production deployment and retirement.

The registry supports the full model lifecycle: register a new model, create versioned
artifacts with full training provenance, transition versions through dev → staging →
production → archived stages, deploy to target environments, track cost attribution across
training and inference, generate CycloneDX 1.5 ML Bills of Materials (ML-BOM) for compliance,
and query fine-tuning lineage graphs for model provenance auditing.

Downstream synthesis engines (tabular, text, vision) register their trained models here and
query for production-approved versions before serving inference traffic. The registry
optionally mirrors registrations into an external MLflow tracking server while remaining the
authoritative data source.

**Product:** Foundation Infrastructure (Product 0)
**Tier:** Foundation Infrastructure
**Phase:** 1A

## Architecture

```
aumos-common ──► aumos-model-registry ──► aumos-tabular-engine
aumos-proto  ──►                       ──► aumos-text-engine
aumos-data-layer ──►                   ──► all synthesis engines
                    ↘ aumos-event-bus (MODEL_LIFECYCLE topics)
                    ↘ MLflow tracking server (optional mirror)
                    ↘ MinIO / S3 (artifact storage)
```

This service follows AumOS hexagonal architecture:

- `api/` — FastAPI routes (thin, delegates to services)
- `core/` — Business logic with no framework dependencies
- `adapters/` — External integrations (PostgreSQL, Kafka, MLflow, MinIO)

### Key Capabilities

| Feature | Implementation |
|---------|---------------|
| Model CRUD + versioning | `core/services.py` → `ModelService` |
| Stage lifecycle transitions | `ModelService.transition_stage()` with guard rules |
| CycloneDX ML-BOM | `core/ml_bom.py` (CycloneDX 1.5 JSON) |
| Cost attribution | `core/cost_engine.py` (training + inference + storage) |
| Model lineage graph | `ModelService.get_lineage()` (parent → child chains) |
| Semantic search | `ModelRepository.search()` (ILIKE, pgvector planned) |
| MLflow mirror | `adapters/mlflow_client.py` (async httpx REST client) |
| Artifact storage | `adapters/minio_client.py` (MinIO/S3-compatible) |
| Kafka events | `adapters/kafka.py` → `Topics.MODEL_LIFECYCLE` |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Access to AumOS internal PyPI for `aumos-common` and `aumos-proto`

### Local Development

```bash
# Clone the repo
git clone https://github.com/aumos-enterprise/aumos-model-registry.git
cd aumos-model-registry

# Set up environment
cp .env.example .env
# Edit .env with your local values

# Install dependencies
make install

# Start infrastructure (PostgreSQL, MLflow, MinIO, Kafka)
make docker-run

# Run the service
uvicorn aumos_model_registry.main:app --reload
```

The service will be available at `http://localhost:8000`.

Health check: `http://localhost:8000/live`
API docs: `http://localhost:8000/docs`

## API Reference

### Authentication

All endpoints require a Bearer JWT token:

```
Authorization: Bearer <token>
X-Tenant-ID: <tenant-uuid>
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/live` | Liveness probe |
| GET | `/ready` | Readiness probe |
| POST | `/api/v1/models` | Register new model |
| GET | `/api/v1/models` | List models (paginated) |
| GET | `/api/v1/models/{model_id}` | Get model details |
| PUT | `/api/v1/models/{model_id}` | Update model metadata |
| DELETE | `/api/v1/models/{model_id}` | Delete model |
| POST | `/api/v1/models/{model_id}/versions` | Create model version |
| GET | `/api/v1/models/{model_id}/versions` | List model versions |
| PATCH | `/api/v1/models/{model_id}/versions/{version_id}/stage` | Transition stage |
| GET | `/api/v1/models/{model_id}/lineage` | Get lineage graph |
| GET | `/api/v1/models/{model_id}/versions/{version_id}/bom` | Get CycloneDX ML-BOM |
| GET | `/api/v1/models/{model_id}/versions/{version_id}/cost` | Get cost breakdown |
| POST | `/api/v1/models/search` | Semantic model search |
| POST | `/api/v1/models/{model_id}/versions/{version_id}/deployments` | Deploy version |
| POST | `/api/v1/deployments/{deployment_id}/rollback` | Roll back deployment |
| POST | `/api/v1/experiments` | Create experiment |
| GET | `/api/v1/experiments` | List experiments |
| POST | `/api/v1/experiments/{experiment_id}/runs` | Start run |
| PATCH | `/api/v1/runs/{run_id}` | Log metrics/artifacts |
| GET | `/api/v1/experiments/{experiment_id}/runs` | List runs |

Full OpenAPI spec available at `/docs` when running locally.

## Configuration

All configuration is via environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `AUMOS_SERVICE_NAME` | `aumos-model-registry` | Service identifier |
| `AUMOS_ENVIRONMENT` | `development` | Runtime environment |
| `AUMOS_DATABASE__URL` | — | PostgreSQL connection string |
| `AUMOS_KAFKA__BROKERS` | `localhost:9092` | Kafka broker list |
| `AUMOS_REGISTRY_MLFLOW_TRACKING_URI` | `postgresql+psycopg2://...` | MLflow backend store |
| `AUMOS_REGISTRY_MLFLOW_ARTIFACT_ROOT` | `s3://aumos-model-artifacts` | MLflow artifact root |
| `AUMOS_REGISTRY_ARTIFACT_BUCKET_PREFIX` | `reg-models` | MinIO bucket prefix |
| `AUMOS_REGISTRY_DEFAULT_GPU_HOURLY_COST_USD` | `3.50` | GPU cost rate for training estimates |
| `AUMOS_REGISTRY_REQUIRE_APPROVAL_FOR_PRODUCTION` | `true` | Production stage gate |
| `AUMOS_REGISTRY_EMBEDDING_MODEL` | `text-embedding-3-small` | Model for semantic search embeddings |

## Database Schema

All tables use the `reg_` prefix:

| Table | Description |
|-------|-------------|
| `reg_models` | Registered model definitions |
| `reg_model_versions` | Versioned model artifacts with provenance |
| `reg_model_deployments` | Active/historical deployment records |
| `reg_experiments` | MLflow-compatible experiment groups |
| `reg_experiment_runs` | Individual training run records |

## Development

### Running Tests

```bash
# Full test suite with coverage
make test

# Fast run (stop on first failure)
make test-quick
```

### Linting and Formatting

```bash
# Check for issues
make lint

# Auto-fix formatting
make format

# Type checking
make typecheck
```

### Adding Dependencies

```bash
# Add a runtime dependency
# Edit pyproject.toml → [project] dependencies
# IMPORTANT: Verify the license is MIT, BSD, Apache, or ISC — never GPL/AGPL

# Add a dev dependency
# Edit pyproject.toml → [project.optional-dependencies] dev
```

## Deployment

### Docker

```bash
# Build image
make docker-build

# Run with docker-compose (includes postgres, mlflow, minio, kafka)
make docker-run
```

### Production

This service is deployed via the AumOS GitOps pipeline. Deployments are triggered
automatically on merge to `main` after CI passes.

**Resource requirements:**
- CPU: 2 cores
- Memory: 2GB
- Storage: 10GB (ephemeral; artifacts in MinIO/S3)

## Related Repos

| Repo | Relationship | Description |
|------|-------------|-------------|
| [aumos-common](https://github.com/aumos-enterprise/aumos-common) | Dependency | Shared utilities, auth, database, events |
| [aumos-proto](https://github.com/aumos-enterprise/aumos-proto) | Dependency | Protobuf event schemas |
| [aumos-data-layer](https://github.com/aumos-enterprise/aumos-data-layer) | Upstream | PostgreSQL + pgvector + RLS |
| [aumos-event-bus](https://github.com/aumos-enterprise/aumos-event-bus) | Upstream | Kafka topics and schema registry |
| [aumos-tabular-engine](https://github.com/aumos-enterprise/aumos-tabular-engine) | Downstream | Registers CTGAN/TVAESynth models here |
| [aumos-text-engine](https://github.com/aumos-enterprise/aumos-text-engine) | Downstream | Registers fine-tuned LLMs here |

## License

Copyright 2026 AumOS Enterprise. Licensed under the [Apache License 2.0](LICENSE).

This software must not incorporate AGPL or GPL licensed components.
See [CONTRIBUTING.md](CONTRIBUTING.md) for license compliance requirements.
