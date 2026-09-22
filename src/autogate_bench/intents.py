"""Seed intents: one or two per (group, capability, distraction, actuation) cell.

This is not the full ~60-intent taxonomy (that is the next task). It is the
minimum set that exercises every branch of the rulebook and every row of the
policy tables, written from public Android Automotive documentation and
common sense only.
"""

from __future__ import annotations

from autogate_bench.schema import (
    ActuationClass as A,
)
from autogate_bench.schema import (
    Capability as C,
)
from autogate_bench.schema import (
    DistractionLevel as L,
)
from autogate_bench.schema import (
    Intent,
)
from autogate_bench.schema import (
    IntentGroup as G,
)

SEED_INTENTS: tuple[Intent, ...] = (
    # Vehicle control: safety-critical
    Intent(
        "defrost_windshield",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.SAFETY_CRITICAL,
        "Turn on windshield defrost / demist",
    ),
    Intent(
        "wipers_on",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.SAFETY_CRITICAL,
        "Turn on or speed up the wipers",
    ),
    Intent(
        "hazard_lights_on",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.SAFETY_CRITICAL,
        "Turn on hazard warning lights",
    ),
    # Vehicle control: comfort
    Intent(
        "set_cabin_temperature",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.COMFORT,
        "Set HVAC temperature",
    ),
    Intent(
        "seat_heater_on", G.VEHICLE_CONTROL, C.LOCAL_OK, L.LOW, A.COMFORT, "Turn on a seat heater"
    ),
    Intent(
        "set_drive_mode",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.COMFORT,
        "Switch between eco / normal / sport",
    ),
    # Vehicle control: restricted
    Intent(
        "disable_lane_assist",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.RESTRICTED,
        "Turn off lane keeping assist",
    ),
    Intent("unlock_doors", G.VEHICLE_CONTROL, C.LOCAL_OK, L.LOW, A.RESTRICTED, "Unlock all doors"),
    Intent(
        "open_trunk", G.VEHICLE_CONTROL, C.LOCAL_OK, L.LOW, A.RESTRICTED, "Open the boot / trunk"
    ),
    Intent(
        "pair_bluetooth_device",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.HIGH,
        A.RESTRICTED,
        "Pair a new phone over Bluetooth (setup flow)",
    ),
    # Navigation
    Intent("next_turn", G.NAVIGATION, C.LOCAL_OK, L.LOW, description="What is the next manoeuvre"),
    Intent(
        "navigate_to_contact_address",
        G.NAVIGATION,
        C.NEEDS_CLOUD,
        L.LOW,
        description="Start navigation to a saved contact's address",
    ),
    Intent(
        "find_nearby_place",
        G.NAVIGATION,
        C.NEEDS_CLOUD,
        L.MEDIUM,
        description="Find and compare nearby restaurants / fuel",
    ),
    # Communication
    Intent(
        "call_contact",
        G.COMMUNICATION,
        C.LOCAL_OK,
        L.LOW,
        description="Place a call to a phonebook contact",
    ),
    Intent(
        "send_text_message",
        G.COMMUNICATION,
        C.NEEDS_SMALL_LOCAL,
        L.LOW,
        description="Dictate a short text message",
    ),
    Intent(
        "read_messages",
        G.COMMUNICATION,
        C.NEEDS_SMALL_LOCAL,
        L.MEDIUM,
        description="Read recent messages aloud",
    ),
    Intent(
        "compose_long_email",
        G.COMMUNICATION,
        C.NEEDS_CLOUD,
        L.HIGH,
        description="Dictate and edit a long email",
    ),
    # Media
    Intent("play_music", G.MEDIA, C.LOCAL_OK, L.LOW, description="Play a song or playlist"),
    Intent("play_video", G.MEDIA, C.LOCAL_OK, L.HIGH, description="Play a video on the screen"),
    Intent("show_photos", G.MEDIA, C.LOCAL_OK, L.HIGH, description="Browse photos on the screen"),
    # Information
    Intent("weather_now", G.INFORMATION, C.NEEDS_CLOUD, L.LOW, description="Current weather"),
    Intent(
        "general_knowledge_question",
        G.INFORMATION,
        C.NEEDS_SMALL_LOCAL,
        L.LOW,
        description="Short factual question",
    ),
    Intent(
        "explain_topic_in_depth",
        G.INFORMATION,
        C.NEEDS_CLOUD,
        L.MEDIUM,
        description="Long explanation of a topic",
    ),
    # Planning
    Intent(
        "plan_multi_day_trip",
        G.PLANNING,
        C.NEEDS_CLOUD,
        L.MEDIUM,
        description="Plan a multi-day trip",
    ),
    # Productivity
    Intent(
        "calendar_today", G.PRODUCTIVITY, C.LOCAL_OK, L.MEDIUM, description="Read today's calendar"
    ),
    Intent(
        "read_document",
        G.PRODUCTIVITY,
        C.NEEDS_CLOUD,
        L.HIGH,
        description="Read a document on screen",
    ),
)

INTENTS_BY_NAME: dict[str, Intent] = {i.name: i for i in SEED_INTENTS}
