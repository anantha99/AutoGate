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
# Router fixtures: pilot rows and a tiny random backbone (nothing downloaded)
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


@pytest.fixture(scope="session")
def tiny_backbone(tmp_path_factory, pilot_rows):
    """A tiny random Qwen3 (hidden 32, 2 layers, 2 heads) and a BPE tokenizer trained here."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from autogate_router.tiny import build_tiny_backbone

    return build_tiny_backbone(tmp_path_factory.mktemp("tiny") / "backbone", pilot_rows)


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_backbone):
    from autogate_router.model import load_tokenizer

    return load_tokenizer(tiny_backbone)
