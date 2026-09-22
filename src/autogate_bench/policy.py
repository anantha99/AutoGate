"""The labeling policy as three small tables plus a handful of switches.

Everything an OEM might want to change lives here as data. The precedence
tree in ``rulebook.py`` reads these tables and contains no policy of its own.

Table 1  demand_table      (speed, workload)          -> DrivingDemand
Table 2  distraction_gate  (distraction, demand)      -> allowed?
Table 3  actuation_gate    (actuation, demand)        -> allowed?

Switches: how much a passenger relaxes the distraction gate, which
connectivity levels count as "available" for a cloud round-trip, and where a
sensitive request goes when privacy mode is strict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autogate_bench.schema import (
    ActuationClass,
    Connectivity,
    DistractionLevel,
    DrivingDemand,
    Route,
    SpeedBucket,
    Workload,
)

D = DrivingDemand

# Table 1: driving demand from speed x workload. Rows: speed. Columns: workload.
DEFAULT_DEMAND_TABLE: dict[tuple[SpeedBucket, Workload], DrivingDemand] = {
    (SpeedBucket.PARKED, Workload.LOW): D.PARKED,
    (SpeedBucket.PARKED, Workload.MEDIUM): D.PARKED,
    (SpeedBucket.PARKED, Workload.HIGH): D.PARKED,
    (SpeedBucket.LOW, Workload.LOW): D.LOW,
    (SpeedBucket.LOW, Workload.MEDIUM): D.MEDIUM,
    (SpeedBucket.LOW, Workload.HIGH): D.HIGH,
    (SpeedBucket.MEDIUM, Workload.LOW): D.MEDIUM,
    (SpeedBucket.MEDIUM, Workload.MEDIUM): D.MEDIUM,
    (SpeedBucket.MEDIUM, Workload.HIGH): D.HIGH,
    (SpeedBucket.HIGH, Workload.LOW): D.MEDIUM,  # highway cruising
    (SpeedBucket.HIGH, Workload.MEDIUM): D.HIGH,
    (SpeedBucket.HIGH, Workload.HIGH): D.HIGH,
}

# Table 2: is a response of this distraction level allowed at this demand?
DEFAULT_DISTRACTION_GATE: dict[tuple[DistractionLevel, DrivingDemand], bool] = {
    (DistractionLevel.LOW, D.PARKED): True,
    (DistractionLevel.LOW, D.LOW): True,
    (DistractionLevel.LOW, D.MEDIUM): True,
    (DistractionLevel.LOW, D.HIGH): True,
    (DistractionLevel.MEDIUM, D.PARKED): True,
    (DistractionLevel.MEDIUM, D.LOW): True,
    (DistractionLevel.MEDIUM, D.MEDIUM): True,
    (DistractionLevel.MEDIUM, D.HIGH): False,
    (DistractionLevel.HIGH, D.PARKED): True,
    (DistractionLevel.HIGH, D.LOW): False,
    (DistractionLevel.HIGH, D.MEDIUM): False,
    (DistractionLevel.HIGH, D.HIGH): False,
}

# Table 3: may the car perform this class of actuation at this demand?
# SAFETY_CRITICAL is always allowed and short-circuits the tree; it is listed
# here so the table is complete and an OEM cannot accidentally leave it out.
DEFAULT_ACTUATION_GATE: dict[tuple[ActuationClass, DrivingDemand], bool] = {
    **{(ActuationClass.SAFETY_CRITICAL, d): True for d in D},
    **{(ActuationClass.COMFORT, d): True for d in D},
    (ActuationClass.RESTRICTED, D.PARKED): True,
    (ActuationClass.RESTRICTED, D.LOW): False,
    (ActuationClass.RESTRICTED, D.MEDIUM): False,
    (ActuationClass.RESTRICTED, D.HIGH): False,
}


@dataclass(frozen=True)
class Policy:
    """A complete, serializable labeling policy."""

    demand_table: dict[tuple[SpeedBucket, Workload], DrivingDemand] = field(
        default_factory=lambda: dict(DEFAULT_DEMAND_TABLE)
    )
    distraction_gate: dict[tuple[DistractionLevel, DrivingDemand], bool] = field(
        default_factory=lambda: dict(DEFAULT_DISTRACTION_GATE)
    )
    actuation_gate: dict[tuple[ActuationClass, DrivingDemand], bool] = field(
        default_factory=lambda: dict(DEFAULT_ACTUATION_GATE)
    )
    # A passenger relaxes the distraction gate by this many demand steps
    # (never below LOW, never for actuation). 0 makes the field a no-op.
    passenger_relax_steps: int = 1
    # Connectivity levels good enough for a cloud round-trip.
    cloud_connectivity: frozenset[Connectivity] = frozenset({Connectivity.WEAK, Connectivity.GOOD})
    # Where a request with sensitive spans goes when privacy mode is strict
    # (masking is not trusted, so it never leaves the car).
    strict_privacy_route: Route = Route.LOCAL

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #

    def demand(self, speed: SpeedBucket, workload: Workload) -> DrivingDemand:
        return self.demand_table[(speed, workload)]

    def distraction_allowed(self, level: DistractionLevel, demand: DrivingDemand) -> bool:
        return self.distraction_gate[(level, demand)]

    def actuation_allowed(self, cls: ActuationClass, demand: DrivingDemand) -> bool:
        return self.actuation_gate[(cls, demand)]

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Every cell present, tables monotone, safety-critical never blocked."""
        missing = [(s, w) for s in SpeedBucket for w in Workload if (s, w) not in self.demand_table]
        if missing:
            raise ValueError(f"demand_table missing cells: {missing}")
        missing = [
            (lv, d) for lv in DistractionLevel for d in D if (lv, d) not in self.distraction_gate
        ]
        if missing:
            raise ValueError(f"distraction_gate missing cells: {missing}")
        gated = [c for c in ActuationClass if c is not ActuationClass.NONE]
        missing = [(c, d) for c in gated for d in D if (c, d) not in self.actuation_gate]
        if missing:
            raise ValueError(f"actuation_gate missing cells: {missing}")

        for d in D:
            if not self.actuation_gate[(ActuationClass.SAFETY_CRITICAL, d)]:
                raise ValueError("safety_critical actuation may never be blocked")
        for w in Workload:
            if self.demand_table[(SpeedBucket.PARKED, w)] is not D.PARKED:
                raise ValueError("a parked car must have PARKED demand")

        # Monotone: more demand never allows more; a higher distraction level
        # is never allowed where a lower one is refused.
        for level in DistractionLevel:
            allowed = [self.distraction_gate[(level, d)] for d in D]
            if any(
                later and not earlier for earlier, later in zip(allowed, allowed[1:], strict=False)
            ):
                raise ValueError(f"distraction_gate not monotone in demand for {level}")
        levels = list(DistractionLevel)
        for d in D:
            for lo, hi in zip(levels, levels[1:], strict=False):
                if self.distraction_gate[(hi, d)] and not self.distraction_gate[(lo, d)]:
                    raise ValueError(f"distraction_gate not monotone in level at {d}")
        for cls in gated:
            allowed = [self.actuation_gate[(cls, d)] for d in D]
            if any(
                later and not earlier for earlier, later in zip(allowed, allowed[1:], strict=False)
            ):
                raise ValueError(f"actuation_gate not monotone in demand for {cls}")
        if self.passenger_relax_steps < 0:
            raise ValueError("passenger_relax_steps must be >= 0")
        if self.strict_privacy_route in (Route.CLOUD, Route.CLOUD_MASKED):
            raise ValueError("strict privacy must not send sensitive spans to the cloud")

    # ------------------------------------------------------------------ #
    # Serialization (so an OEM can ship a policy as a JSON file)
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "demand_table": {f"{s}|{w}": str(d) for (s, w), d in self.demand_table.items()},
            "distraction_gate": {f"{lv}|{d}": a for (lv, d), a in self.distraction_gate.items()},
            "actuation_gate": {f"{c}|{d}": a for (c, d), a in self.actuation_gate.items()},
            "passenger_relax_steps": self.passenger_relax_steps,
            "cloud_connectivity": sorted(str(c) for c in self.cloud_connectivity),
            "strict_privacy_route": str(self.strict_privacy_route),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        def split(key: str) -> tuple[str, str]:
            a, b = key.split("|")
            return a, b

        demand = {
            (SpeedBucket(a), Workload(b)): DrivingDemand(v)
            for k, v in data["demand_table"].items()
            for a, b in [split(k)]
        }
        distraction = {
            (DistractionLevel(a), DrivingDemand(b)): bool(v)
            for k, v in data["distraction_gate"].items()
            for a, b in [split(k)]
        }
        actuation = {
            (ActuationClass(a), DrivingDemand(b)): bool(v)
            for k, v in data["actuation_gate"].items()
            for a, b in [split(k)]
        }
        return cls(
            demand_table=demand,
            distraction_gate=distraction,
            actuation_gate=actuation,
            passenger_relax_steps=int(data.get("passenger_relax_steps", 1)),
            cloud_connectivity=frozenset(
                Connectivity(c) for c in data.get("cloud_connectivity", ["weak", "good"])
            ),
            strict_privacy_route=Route(data.get("strict_privacy_route", "LOCAL")),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        return cls.from_dict(json.loads(Path(path).read_text()))


DEFAULT_POLICY = Policy()
