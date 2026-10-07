"""The labeling policy as four small tables plus a handful of switches.

Everything an OEM might want to change lives here as data. The precedence
tree in ``rulebook.py`` reads these tables and contains no policy of its own.

Table 1  demand_table      (speed, workload)          -> DrivingDemand
Table 2  distraction_gate  (distraction, demand)      -> allowed?
Table 3  actuation_gate    (actuation, demand)        -> allowed?
Table 4  capability        intent                     -> Capability

Switches: how much a passenger relaxes the distraction gate, which
connectivity levels count as "available" for a cloud round-trip, and where a
sensitive request goes when privacy mode is strict.

Tables 1-3 carry the safety, distraction and privacy decisions. Table 4 is
about the onboard model: the default is conservative (only what a tiny,
<= 2B head-unit model reliably handles is ``local_ok``) and an OEM with a
stronger model widens it.

Locked: ``validate`` rejects any policy that breaks these, so no OEM file can
configure them away.

* Safety-critical actuation is never blocked. The rulebook also checks it
  before any table, so those commands always run on the head unit.
* Every vehicle command is ``local_ok`` in Table 4. Capability can never send
  a command to the cloud or make it wait; only Table 3 decides whether a
  restricted command is refused while moving.
* Strict privacy never sends sensitive spans to the cloud, masked or not.
* A parked car has PARKED demand, and Tables 2 and 3 are monotone: more
  demand never allows more.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autogate_bench.intents import INTENTS, INTENTS_BY_NAME
from autogate_bench.schema import (
    ActuationClass,
    Capability,
    Connectivity,
    DistractionLevel,
    DrivingDemand,
    Intent,
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

# Table 4: what it takes to answer each intent well, given the onboard model.
# local_ok          any head-unit model, including a tiny (<= 2B) one
# needs_small_local LOCAL on a small (<= 8B) head-unit model, cloud on a tiny one
# needs_cloud       live data or a frontier model; offline -> DEFER
# cloud_preferred   better in the cloud, onboard data can answer; offline -> LOCAL
# The context's local_model_tier says which model this car has; this table says
# what the OEM trusts each tier with. A trailing comment marks a judgment call.
_CAP = Capability
DEFAULT_CAPABILITY: dict[str, Capability] = {
    # vehicle_control: locked to local_ok (see the module docstring)
    **{i.name: _CAP.LOCAL_OK for i in INTENTS if i.actuation is not ActuationClass.NONE},
    # navigation
    "next_turn": _CAP.LOCAL_OK,
    "cancel_route": _CAP.LOCAL_OK,
    "navigate_home": _CAP.LOCAL_OK,  # saved destination plus onboard maps; no search or live data
    "eta_to_destination": _CAP.LOCAL_OK,  # the active nav engine already holds the ETA
    "navigate_to_contact_address": _CAP.CLOUD_PREFERRED,  # geocoding a free-text address
    "find_nearby_place": _CAP.CLOUD_PREFERRED,
    "traffic_on_route": _CAP.NEEDS_CLOUD,
    "add_stop_along_route": _CAP.CLOUD_PREFERRED,
    # communication
    "call_contact": _CAP.LOCAL_OK,
    "lookup_contact_info": _CAP.LOCAL_OK,
    "send_text_message": _CAP.NEEDS_SMALL_LOCAL,
    "reply_to_message": _CAP.NEEDS_SMALL_LOCAL,
    "read_messages": _CAP.NEEDS_SMALL_LOCAL,
    "summarize_group_chat": _CAP.NEEDS_SMALL_LOCAL,
    "play_voicemail": _CAP.LOCAL_OK,  # playback is trivial
    "start_video_call": _CAP.LOCAL_OK,
    "compose_long_email": _CAP.NEEDS_CLOUD,
    # media
    "play_music": _CAP.LOCAL_OK,
    "pause_playback": _CAP.LOCAL_OK,
    "skip_track": _CAP.LOCAL_OK,
    "tune_radio_station": _CAP.LOCAL_OK,
    "play_podcast": _CAP.LOCAL_OK,  # the media app resolves the catalog search
    "play_video": _CAP.LOCAL_OK,
    "show_photos": _CAP.LOCAL_OK,
    # information
    "weather_now": _CAP.NEEDS_CLOUD,
    "current_time": _CAP.LOCAL_OK,
    "sports_score": _CAP.NEEDS_CLOUD,
    "general_knowledge_question": _CAP.NEEDS_SMALL_LOCAL,
    "explain_topic_in_depth": _CAP.NEEDS_CLOUD,
    "vehicle_manual_question": _CAP.NEEDS_SMALL_LOCAL,  # onboard-manual retrieval: 8B, not 2B
    "fuel_range": _CAP.LOCAL_OK,
    # planning
    "plan_multi_day_trip": _CAP.NEEDS_CLOUD,
    "plan_day_itinerary": _CAP.NEEDS_CLOUD,
    "book_restaurant": _CAP.NEEDS_CLOUD,
    "plan_charging_stops": _CAP.CLOUD_PREFERRED,
    # productivity
    "calendar_today": _CAP.LOCAL_OK,  # a templated read-out of the synced calendar
    "add_reminder": _CAP.LOCAL_OK,  # time plus free-text slot filling
    "add_calendar_event": _CAP.NEEDS_SMALL_LOCAL,  # dates, durations and attendees together
    "take_note": _CAP.LOCAL_OK,  # saved verbatim, unlike a message that is rephrased
    "edit_note": _CAP.NEEDS_SMALL_LOCAL,
    "read_document": _CAP.NEEDS_CLOUD,  # documents live in cloud storage, exceed local context
    "pair_bluetooth_device": _CAP.LOCAL_OK,
    "browse_web": _CAP.NEEDS_CLOUD,
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
    # Table 4, keyed by intent name.
    capability: dict[str, Capability] = field(default_factory=lambda: dict(DEFAULT_CAPABILITY))

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

    def capability_of(self, intent: Intent | str) -> Capability:
        return self.capability[intent if isinstance(intent, str) else intent.name]

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

        missing = sorted(set(INTENTS_BY_NAME) - set(self.capability))
        if missing:
            raise ValueError(f"capability missing intents: {missing}")
        unknown = sorted(set(self.capability) - set(INTENTS_BY_NAME))
        if unknown:
            raise ValueError(f"capability names unknown intents: {unknown}")
        cloud_commands = sorted(
            i.name
            for i in INTENTS
            if i.actuation is not ActuationClass.NONE
            and self.capability[i.name] is not Capability.LOCAL_OK
        )
        if cloud_commands:
            raise ValueError(f"vehicle commands must be local_ok: {cloud_commands}")

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
            "capability": {name: str(c) for name, c in self.capability.items()},
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
            # Intents left out keep their default, so an OEM file lists only what it widens.
            capability={
                **DEFAULT_CAPABILITY,
                **{name: Capability(c) for name, c in data.get("capability", {}).items()},
            },
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        return cls.from_dict(json.loads(Path(path).read_text()))


DEFAULT_POLICY = Policy()
