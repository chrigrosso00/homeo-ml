"""Pytest configuration and session-level fixtures.

Automatically ensures that minimal synthetic test fixtures and baseline
model artifacts are present before executing any unit or integration tests.
"""

import pytest

from scripts.generate_test_fixtures import (
    ensure_fraud_raw_fixtures,
    ensure_model_artifacts,
    ensure_taxi_raw_fixtures,
)


@pytest.fixture(scope="session", autouse=True)
def ensure_test_environment() -> None:
    """Ensure minimal test fixtures and baseline artifacts exist."""
    ensure_fraud_raw_fixtures()
    ensure_taxi_raw_fixtures()
    ensure_model_artifacts()
