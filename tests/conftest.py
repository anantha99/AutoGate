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


# --------------------------------------------------------------------------- #
# Pilot rows as dicts, and written to a temporary rows.jsonl
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def pilot_rows() -> list[dict]:
    from autogate_bench.dataset import build_rows

    return [r.to_dict() for r in build_rows(rng_seed=0)]


@pytest.fixture(scope="session")
def pilot_jsonl(tmp_path_factory, pilot_rows):
    import json

    path = tmp_path_factory.mktemp("pilot") / "rows.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in pilot_rows:
            f.write(json.dumps(r) + "\n")
    return path
