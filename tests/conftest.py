"""Test configuration and shared fixtures for aumos-model-registry tests."""

import sys
from pathlib import Path

import pytest

# Ensure the src/ layout is importable without installing the package.
# Required when running tests outside of a virtual environment where
# `pip install -e .` has not been executed.
_SRC_PATH = Path(__file__).parent.parent / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

# pytest-asyncio configuration is set to auto mode in pyproject.toml
# (asyncio_mode = "auto"), so async test functions are discovered automatically.
