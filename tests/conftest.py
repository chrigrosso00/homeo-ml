"""Pytest configuration and session-level fixtures.

Automatically ensures that minimal synthetic test data fixtures are present
before executing tests, preventing FileNotFoundError in fresh environments.
"""

import pytest

from scripts.generate_test_fixtures import (
    ensure_fraud_raw_fixtures,
    ensure_taxi_raw_fixtures,
)


@pytest.fixture(scope="session", autouse=True)
def ensure_test_fixtures() -> None:
    """Ensure minimal raw test fixtures exist."""
    ensure_fraud_raw_fixtures()
    ensure_taxi_raw_fixtures()
