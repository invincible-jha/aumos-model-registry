# Contributing to aumos-model-registry

Thank you for contributing to AumOS Enterprise. This guide covers everything you need
to get started and ensure your contributions meet our standards.

## Getting Started

1. Fork the repository (external contributors) or clone directly (AumOS team members)
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```
3. Make your changes following the standards below
4. Submit a pull request targeting `main`

## Development Setup

### Prerequisites

- Python 3.11 or 3.12
- Docker and Docker Compose
- Access to AumOS internal PyPI (for `aumos-common` and `aumos-proto`)

### Install

```bash
# Install all dependencies including dev tools
make install

# Copy and configure environment
cp .env.example .env
# Edit .env with your local settings

# Start local infrastructure (postgres, mlflow, minio, kafka)
make docker-run
```

### Verify Setup

```bash
make lint       # Should pass with no errors
make typecheck  # Should pass with no errors
make test       # Should pass with coverage >= 80%
```

## Code Standards

All code in this repository must follow the standards defined in [CLAUDE.md](CLAUDE.md).
Key requirements:

- **Type hints on every function** — no exceptions
- **Pydantic models for all API inputs/outputs** — never return raw dicts
- **Structured logging** — use `get_logger(__name__)`, never `print()`
- **Async by default** — all I/O must be async
- **Import from aumos-common** — never reimplement shared utilities
- **Google-style docstrings** on all public classes and methods
- **Max line length: 120 characters**
- **Decimal for all monetary values** — never float

Run `make lint` and `make typecheck` before every commit.

## PR Process

1. Ensure all CI checks pass (lint, typecheck, test, docker build, license check)
2. Fill out the PR template completely
3. Request review from at least one member of `@aumos/platform-team`
4. Squash merge only — keep history clean
5. Delete your branch after merge

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add pgvector semantic search for model discovery
fix: resolve version number race condition on concurrent creates
refactor: extract cost calculation into separate helper functions
docs: update API reference for ML-BOM endpoint
test: add integration tests for MLflow mirror adapter
chore: bump mlflow to 2.11.0
```

Commit messages explain **WHY**, not just what changed.

## License Compliance — CRITICAL

**This is the most important section. Read it carefully.**

AumOS Enterprise is licensed under Apache 2.0. Our enterprise customers have strict
requirements that prohibit AGPL and GPL licensed code in our platform.

### What You MUST NOT Do

- **NEVER add a dependency with a GPL or AGPL license**, even indirectly
- **NEVER copy GPL/AGPL code** into this repository
- **NEVER wrap a GPL/AGPL tool** without explicit written approval from legal

### Approved Licenses

The following licenses are approved for dependencies:

- MIT
- BSD (2-clause or 3-clause)
- Apache Software License 2.0
- ISC
- Python Software Foundation (PSF)
- Mozilla Public License 2.0 (MPL 2.0) — with restrictions, check with team

### Checking License Before Adding a Dependency

```bash
# Before adding any new package, check its license:
pip install pip-licenses
pip install <new-package>
pip-licenses --packages <new-package>

# The CI license-check job enforces this automatically
```

Note: MLflow is Apache 2.0 licensed. cyclonedx-python-lib is Apache 2.0 licensed.

## Testing Requirements

- All new features must include tests
- Coverage must remain >= 80% for `core/` modules
- Coverage must remain >= 60% for `adapters/`
- Use `testcontainers` for integration tests requiring real PostgreSQL/Kafka
- Mock MLflow and MinIO in unit tests

```bash
# Run the full test suite
make test

# Run a specific test file
pytest tests/test_services.py -v

# Run with coverage report
pytest tests/ --cov --cov-report=html
```

## Code of Conduct

We are committed to providing a welcoming and respectful environment for all contributors.
All participants are expected to:

- Be respectful and constructive in all interactions
- Focus on what is best for the project and platform
- Accept feedback graciously and provide it thoughtfully
- Report unacceptable behavior to the platform team

Violations may result in removal from the project.
