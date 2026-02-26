# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added

- Initial scaffolding for `aumos-model-registry`
- `core/models.py` — SQLAlchemy ORM models with `reg_` table prefix:
  - `Model` (reg_models) — tenant-scoped model definitions
  - `ModelVersion` (reg_model_versions) — versioned artifacts with training provenance
  - `ModelDeployment` (reg_model_deployments) — deployment lifecycle tracking
  - `Experiment` (reg_experiments) — MLflow-compatible experiment grouping
  - `ExperimentRun` (reg_experiment_runs) — individual training run records
- `core/interfaces.py` — Protocol interfaces for all repositories and storage adapters
- `core/services.py` — `ModelService` and `ExperimentService` with full lifecycle logic
- `core/cost_engine.py` — Training, inference, and storage cost attribution
- `core/ml_bom.py` — CycloneDX 1.5 ML Bill of Materials generator
- `api/schemas.py` — Full set of Pydantic v2 request/response schemas
- `api/router.py` — FastAPI APIRouter with 20+ endpoints:
  - Model CRUD (POST/GET/PUT/DELETE `/models`)
  - Model versioning with auto-increment
  - Stage lifecycle transitions with guard rules
  - Model lineage graph
  - CycloneDX ML-BOM generation and retrieval
  - Cost attribution breakdown
  - Semantic model search (ILIKE-based, pgvector planned)
  - Deployment management with rollback
  - Experiment and run management (MLflow-compatible)
- `adapters/repositories.py` — SQLAlchemy async repositories for all entities
- `adapters/kafka.py` — `ModelRegistryEventPublisher` wrapping aumos-common EventPublisher
- `adapters/mlflow_client.py` — Async httpx REST client for MLflow Model Registry API
- `adapters/minio_client.py` — MinIO/S3-compatible artifact storage client
- `main.py` — FastAPI app with lifespan lifecycle (Kafka + MinIO initialization)
- `settings.py` — `Settings` extending `AumOSSettings` with AUMOS_REGISTRY_ prefix
- All standard AumOS deliverables: CLAUDE.md, README.md, pyproject.toml, Dockerfile,
  .env.example, Makefile, .gitignore, .dockerignore, LICENSE (Apache 2.0),
  CONTRIBUTING.md, SECURITY.md, CHANGELOG.md, .github/workflows/ci.yml,
  docker-compose.dev.yml (includes postgres, mlflow, minio, kafka services)
