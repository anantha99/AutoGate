import pytest

from autogate_bench import Connectivity, Context, SpeedBucket, Workload


@pytest.fixture
def parked() -> Context:
    return Context(SpeedBucket.PARKED, Connectivity.GOOD, Workload.LOW)


@pytest.fixture
def cruising() -> Context:
    """Highway, low workload: MEDIUM demand."""
    return Context(SpeedBucket.HIGH, Connectivity.GOOD, Workload.LOW)


@pytest.fixture
def busy() -> Context:
    """High speed, high workload: HIGH demand."""
    return Context(SpeedBucket.HIGH, Connectivity.GOOD, Workload.HIGH)


@pytest.fixture
def offline() -> Context:
    return Context(SpeedBucket.LOW, Connectivity.NONE, Workload.LOW)
