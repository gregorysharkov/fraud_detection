import pytest


@pytest.fixture(scope="session")
def wait_time_seconds() -> float:
    return 0.15
