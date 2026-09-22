"""Types shared by the benchmark: context vector, routes, and intent tags.

Every enum here is a closed vocabulary. The context fields mirror the PRD's
input table; the intent tags are what the rulebook reads. Nothing in this
module contains policy; policy lives in ``policy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# --------------------------------------------------------------------------- #
# Context vector (the seven fields the router sees beside the utterance)
# --------------------------------------------------------------------------- #


class SpeedBucket(StrEnum):
    PARKED = "parked"
    LOW = "low"  # < 30 km/h
    MEDIUM = "medium"
    HIGH = "high"  # > 80 km/h


class Connectivity(StrEnum):
    NONE = "none"
    WEAK = "weak"
    GOOD = "good"


class Workload(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PrivacyMode(StrEnum):
    STANDARD = "standard"
    STRICT = "strict"


class LocalModelTier(StrEnum):
    TINY = "tiny"  # <= 2B parameters on the head unit
    SMALL = "small"  # <= 8B parameters on the head unit


@dataclass(frozen=True, slots=True)
class Context:
    """The vehicle context paired with one utterance."""

    speed_bucket: SpeedBucket
    connectivity: Connectivity
    driver_workload: Workload
    privacy_mode: PrivacyMode = PrivacyMode.STANDARD
    passenger_present: bool = False
    local_model_tier: LocalModelTier = LocalModelTier.TINY

    def to_prefix(self) -> str:
        """Serialize as the bracketed prefix the router model consumes."""
        return (
            f"[speed={self.speed_bucket}]"
            f"[conn={self.connectivity}]"
            f"[workload={self.driver_workload}]"
            f"[privacy={self.privacy_mode}]"
            f"[passenger={'yes' if self.passenger_present else 'no'}]"
            f"[local={self.local_model_tier}]"
        )


# --------------------------------------------------------------------------- #
# Routes (the label space)
# --------------------------------------------------------------------------- #


class Route(StrEnum):
    LOCAL = "LOCAL"
    CLOUD = "CLOUD"
    CLOUD_MASKED = "CLOUD_MASKED"
    DEFER = "DEFER"
    REFUSE = "REFUSE"


# --------------------------------------------------------------------------- #
# Derived context level
# --------------------------------------------------------------------------- #


class DrivingDemand(StrEnum):
    """How much attention the driving task currently needs.

    Derived from speed and workload by ``Policy.demand_table``. Ordered:
    PARKED < LOW < MEDIUM < HIGH.
    """

    PARKED = "parked"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return _DEMAND_ORDER.index(self)

    def relaxed(self, steps: int = 1, floor: DrivingDemand | None = None) -> DrivingDemand:
        """Step the demand down, never below ``floor`` (defaults to LOW)."""
        floor = floor or DrivingDemand.LOW
        if self is DrivingDemand.PARKED:
            return self
        target = max(self.rank - steps, floor.rank)
        return _DEMAND_ORDER[target]


_DEMAND_ORDER = [DrivingDemand.PARKED, DrivingDemand.LOW, DrivingDemand.MEDIUM, DrivingDemand.HIGH]


# --------------------------------------------------------------------------- #
# Intent tags
# --------------------------------------------------------------------------- #


class IntentGroup(StrEnum):
    VEHICLE_CONTROL = "vehicle_control"
    NAVIGATION = "navigation"
    COMMUNICATION = "communication"
    MEDIA = "media"
    INFORMATION = "information"
    PLANNING = "planning"
    PRODUCTIVITY = "productivity"


class Capability(StrEnum):
    """What it takes to answer the intent well."""

    LOCAL_OK = "local_ok"  # any head-unit model can do it
    NEEDS_SMALL_LOCAL = "needs_small_local"  # ok on a <= 8B local model, else cloud
    NEEDS_CLOUD = "needs_cloud"  # needs live data or a frontier model


class DistractionLevel(StrEnum):
    """How much of the driver's attention the *response* demands.

    LOW: short spoken answer or a vehicle command.
    MEDIUM: long spoken output or a decision (read messages, compare options).
    HIGH: needs the screen or sustained interaction (photos, video, documents,
    multi-edit dictation, device pairing, settings menus). Mirrors the set that
    Android Automotive's UX restrictions block while moving.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActuationClass(StrEnum):
    """For vehicle-control intents: what the car is allowed to do on request."""

    NONE = "none"  # not a vehicle command
    SAFETY_CRITICAL = "safety_critical"  # defrost, wipers, hazards: always LOCAL, immediately
    COMFORT = "comfort"  # HVAC, seats, volume, drive mode: always allowed
    RESTRICTED = "restricted"  # ADAS off, door unlock, trunk, park brake: parked only


@dataclass(frozen=True, slots=True)
class Intent:
    """One entry in the intent taxonomy."""

    name: str
    group: IntentGroup
    capability: Capability
    distraction: DistractionLevel
    actuation: ActuationClass = ActuationClass.NONE
    description: str = ""

    def __post_init__(self) -> None:
        if (
            self.actuation is not ActuationClass.NONE
            and self.group is not IntentGroup.VEHICLE_CONTROL
        ):
            raise ValueError(
                f"{self.name}: only vehicle_control intents may have an actuation class"
            )
        if self.group is IntentGroup.VEHICLE_CONTROL and self.actuation is ActuationClass.NONE:
            raise ValueError(f"{self.name}: vehicle_control intents need an actuation class")
        if self.actuation is not ActuationClass.NONE and self.capability is not Capability.LOCAL_OK:
            raise ValueError(f"{self.name}: vehicle commands must be local_ok")
