"""The AutoGate intent taxonomy: 65 intents in seven groups.

Every benchmark utterance maps to exactly one intent. The intent's two routing tags
(distraction, actuation), its capability in the labeling policy
(``Policy.capability``, Table 4) and the vehicle context fully determine the
route label via ``rulebook.route``.

Distraction and actuation describe the request and the car, so they live
here. Capability describes what the OEM's onboard model can do, so it lives
in the policy, where an OEM can widen it.

Sources: public Android Automotive OS documentation only (VehiclePropertyIds,
CarUxRestrictions, CarAudioManager, the AAOS voice interaction docs) and
common sense. No proprietary supplier or OEM intent list was used.

Invariants (checked in ``tests/test_intents.py``):

* Vehicle-control intents are LOW distraction: the command itself is
  short. They are refused, if at all, by their actuation class. (Every
  policy must also keep them local_ok; ``Policy.validate`` enforces it.)
* Everything else is refused, if at all, by its distraction level. No intent
  is both RESTRICTED and above LOW distraction, so the two REFUSE mechanisms
  never overlap.
* Every intent carries a ``consequence`` tag (none / reversible /
  irreversible). It does not affect the route. The deterministic policy
  gate reads it to decide whether to ask for confirmation before an
  execute route runs: irreversible intents and restricted actuation are
  confirmed; safety-critical commands never are.
* Screen-heavy setup flows (device pairing) live in productivity with HIGH
  distraction and no actuation class, mirroring ``UX_RESTRICTIONS_NO_SETUP``.

A trailing ``#`` comment marks a tag that is a judgment call. The per-intent
rationale, with the AAOS property or UX restriction each maps to, is in
``docs/taxonomy.md``.
"""

from __future__ import annotations

from autogate_bench.schema import ActuationClass as A
from autogate_bench.schema import Consequence as Q
from autogate_bench.schema import DistractionLevel as L
from autogate_bench.schema import Intent
from autogate_bench.schema import IntentGroup as G

INTENTS: tuple[Intent, ...] = (
    # ======================================================================= #
    # vehicle_control (22): all LOW distraction
    # ======================================================================= #
    # ---- safety_critical (5): a delay could itself be dangerous ------------ #
    Intent(
        "defrost_windshield",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.SAFETY_CRITICAL,
        "Turn on windshield defrost or demist",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "wipers_on",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.SAFETY_CRITICAL,
        "Turn on or speed up the windshield wipers",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "hazard_lights_on",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.SAFETY_CRITICAL,
        "Turn on the hazard warning lights",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "headlights_on",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.SAFETY_CRITICAL,
        "Turn on the headlights",
        consequence=Q.REVERSIBLE,
    ),  # judgment: headlights are visibility, not comfort; driving unlit at dusk is the hazard
    Intent(
        "emergency_call",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.SAFETY_CRITICAL,
        "Call emergency services from the car",
        consequence=Q.IRREVERSIBLE,  # irreversible, but safety_critical is exempt from confirmation
    ),  # judgment: an eCall-style vehicle function, so it bypasses every gate and never DEFERs
    # ---- comfort (11): adjusted routinely while moving ---------------------- #
    Intent(
        "set_cabin_temperature",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.COMFORT,
        "Set the HVAC temperature",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "set_fan_speed",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.COMFORT,
        "Change the HVAC fan speed",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "climate_on_off",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.COMFORT,
        "Turn the air conditioning or climate system on or off",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "set_recirculation",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.COMFORT,
        "Switch between recirculated and fresh cabin air",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "seat_ventilation_on",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.COMFORT,
        "Turn on or adjust a ventilated (cooled) seat",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "seat_heater_on",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.COMFORT,
        "Turn on or adjust a seat heater",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "adjust_window",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.COMFORT,
        "Open or close a window, fully or partly",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "set_audio_volume",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.COMFORT,
        "Turn the audio volume up or down",
        consequence=Q.REVERSIBLE,
    ),  # judgment: volume is a car-audio actuation (CarAudioManager), so vehicle_control not media
    Intent(
        "set_drive_mode",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.COMFORT,
        "Switch drive mode, such as eco, normal, or sport",
        consequence=Q.REVERSIBLE,
    ),  # judgment: drive-mode selectors work on the move; no standard VHAL property exists
    Intent(
        "set_ambient_lighting",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.COMFORT,
        "Change the interior ambient lighting",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "set_cruise_speed",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.COMFORT,
        "Raise or lower the cruise control set speed",
        consequence=Q.REVERSIBLE,
    ),  # judgment: nudging an engaged system's speed is routine; turning assistance off is not
    # ---- restricted (6): only when parked ---------------------------------- #
    Intent(
        "disable_lane_assist",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.RESTRICTED,
        "Turn off lane keeping assist",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "unlock_doors",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.RESTRICTED,
        "Unlock the doors",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "open_trunk",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.RESTRICTED,
        "Open the trunk or tailgate",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "set_parking_brake",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.RESTRICTED,
        "Apply or release the parking brake",
        consequence=Q.REVERSIBLE,
    ),  # judgment: both directions restricted; applying at speed is an unplanned emergency stop
    Intent(
        "disable_child_locks",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.RESTRICTED,
        "Turn off the rear-door child locks",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "fold_mirrors",
        G.VEHICLE_CONTROL,
        L.LOW,
        A.RESTRICTED,
        "Fold or unfold the side mirrors",
        consequence=Q.REVERSIBLE,
    ),  # judgment: folding removes rear visibility while moving; unfold-only is not split out
    # ======================================================================= #
    # navigation (8)
    # ======================================================================= #
    Intent(
        "next_turn",
        G.NAVIGATION,
        L.LOW,
        description="Ask what the next manoeuvre on the active route is",
        consequence=Q.NONE,
    ),
    Intent(
        "cancel_route",
        G.NAVIGATION,
        L.LOW,
        description="Stop the active navigation",
        consequence=Q.REVERSIBLE,  # reversible: the route can be restarted
    ),
    Intent(
        "navigate_home",
        G.NAVIGATION,
        L.LOW,
        description="Start navigation to a saved place such as home or work",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "eta_to_destination",
        G.NAVIGATION,
        L.LOW,
        description="Ask when the car will arrive at the current destination",
        consequence=Q.NONE,
    ),
    Intent(
        "navigate_to_contact_address",
        G.NAVIGATION,
        L.LOW,
        description="Start navigation to a contact's address",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "find_nearby_place",
        G.NAVIGATION,
        L.MEDIUM,
        description="Find and compare nearby places such as restaurants or fuel stations",
        consequence=Q.NONE,
    ),
    Intent(
        "traffic_on_route",
        G.NAVIGATION,
        L.LOW,
        description="Ask about traffic or delays on the current route",
        consequence=Q.NONE,
    ),
    Intent(
        "add_stop_along_route",
        G.NAVIGATION,
        L.MEDIUM,
        description="Add a stop, such as coffee or a charger, along the current route",
        consequence=Q.REVERSIBLE,
    ),
    # ======================================================================= #
    # communication (9)
    # ======================================================================= #
    Intent(
        "call_contact",
        G.COMMUNICATION,
        L.LOW,
        description="Place a phone call to a contact",
        consequence=Q.IRREVERSIBLE,
    ),
    Intent(
        "lookup_contact_info",
        G.COMMUNICATION,
        L.LOW,
        description="Ask for a contact's phone number or address",
        consequence=Q.NONE,
    ),
    Intent(
        "send_text_message",
        G.COMMUNICATION,
        L.LOW,
        description="Dictate and send a short text message",
        consequence=Q.IRREVERSIBLE,
    ),
    Intent(
        "reply_to_message",
        G.COMMUNICATION,
        L.LOW,
        description="Dictate a short reply to a message just read out",
        consequence=Q.IRREVERSIBLE,
    ),  # judgment: LOW because the reply is one short utterance; reading the thread is separate
    Intent(
        "read_messages",
        G.COMMUNICATION,
        L.MEDIUM,
        description="Read recent unread messages aloud",
        consequence=Q.NONE,
    ),
    Intent(
        "summarize_group_chat",
        G.COMMUNICATION,
        L.MEDIUM,
        description="Summarise what was said in a group conversation",
        consequence=Q.NONE,
    ),
    Intent(
        "play_voicemail",
        G.COMMUNICATION,
        L.MEDIUM,
        description="Play new voicemail messages",
        consequence=Q.NONE,
    ),  # judgment: MEDIUM because listening to voicemail is long output
    Intent(
        "start_video_call",
        G.COMMUNICATION,
        L.HIGH,
        description="Start a video call with a contact",
        consequence=Q.IRREVERSIBLE,
    ),
    Intent(
        "compose_long_email",
        G.COMMUNICATION,
        L.HIGH,
        description="Dictate and edit a long email",
        consequence=Q.IRREVERSIBLE,  # irreversible once sent
    ),
    # ======================================================================= #
    # media (7)
    # ======================================================================= #
    Intent(
        "play_music",
        G.MEDIA,
        L.LOW,
        description="Play a song, artist, album, or playlist",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "pause_playback",
        G.MEDIA,
        L.LOW,
        description="Pause or resume the current media",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "skip_track",
        G.MEDIA,
        L.LOW,
        description="Skip to the next or previous track",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "tune_radio_station",
        G.MEDIA,
        L.LOW,
        description="Tune the radio to a station or frequency",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "play_podcast",
        G.MEDIA,
        L.LOW,
        description="Play a podcast or a specific episode",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "play_video",
        G.MEDIA,
        L.HIGH,
        description="Play a video on the screen",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "show_photos",
        G.MEDIA,
        L.HIGH,
        description="Browse photos on the screen",
        consequence=Q.NONE,
    ),
    # ======================================================================= #
    # information (7)
    # ======================================================================= #
    Intent(
        "weather_now",
        G.INFORMATION,
        L.LOW,
        description="Ask about the current or upcoming weather",
        consequence=Q.NONE,
    ),
    Intent(
        "current_time",
        G.INFORMATION,
        L.LOW,
        description="Ask the time or date",
        consequence=Q.NONE,
    ),
    Intent(
        "sports_score",
        G.INFORMATION,
        L.LOW,
        description="Ask for a live or recent sports score",
        consequence=Q.NONE,
    ),
    Intent(
        "general_knowledge_question",
        G.INFORMATION,
        L.LOW,
        description="Ask a short factual question",
        consequence=Q.NONE,
    ),
    Intent(
        "explain_topic_in_depth",
        G.INFORMATION,
        L.MEDIUM,
        description="Ask for a long explanation of a topic",
        consequence=Q.NONE,
    ),
    Intent(
        "vehicle_manual_question",
        G.INFORMATION,
        L.LOW,
        description="Ask how a car feature works or what a warning light means",
        consequence=Q.NONE,
    ),
    Intent(
        "fuel_range",
        G.INFORMATION,
        L.LOW,
        description="Ask how far the car can go on the remaining fuel or charge",
        consequence=Q.NONE,
    ),
    # ======================================================================= #
    # planning (4): all MEDIUM
    # ======================================================================= #
    Intent(
        "plan_multi_day_trip",
        G.PLANNING,
        L.MEDIUM,
        description="Plan a trip spanning several days",
        consequence=Q.NONE,
    ),
    Intent(
        "plan_day_itinerary",
        G.PLANNING,
        L.MEDIUM,
        description="Plan a sequence of stops for today",
        consequence=Q.NONE,
    ),
    Intent(
        "book_restaurant",
        G.PLANNING,
        L.MEDIUM,
        description="Find and reserve a table at a restaurant",
        consequence=Q.IRREVERSIBLE,
    ),  # judgment: MEDIUM not HIGH; confirming time and party size is a spoken decision
    Intent(
        "plan_charging_stops",
        G.PLANNING,
        L.MEDIUM,
        description="Plan charging stops for an electric-vehicle journey",
        consequence=Q.NONE,
    ),
    # ======================================================================= #
    # productivity (8)
    # ======================================================================= #
    Intent(
        "calendar_today",
        G.PRODUCTIVITY,
        L.MEDIUM,
        description="Hear today's calendar",
        consequence=Q.NONE,
    ),
    Intent(
        "add_reminder",
        G.PRODUCTIVITY,
        L.LOW,
        description="Set a reminder for a time or place",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "add_calendar_event",
        G.PRODUCTIVITY,
        L.LOW,
        description="Add an event to the calendar",
        consequence=Q.IRREVERSIBLE,  # irreversible: attendees receive invites
    ),
    Intent(
        "take_note",
        G.PRODUCTIVITY,
        L.LOW,
        description="Dictate a short note to save",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "edit_note",
        G.PRODUCTIVITY,
        L.HIGH,
        description="Make several spoken edits to an existing note",
        consequence=Q.REVERSIBLE,  # reversible: notes keep history
    ),  # judgment: multi-edit dictation needs the transcript on screen to check each edit
    Intent(
        "read_document",
        G.PRODUCTIVITY,
        L.HIGH,
        description="Open and read a document on the screen",
        consequence=Q.NONE,
    ),
    Intent(
        "pair_bluetooth_device",
        G.PRODUCTIVITY,
        L.HIGH,
        description="Pair a new phone or device over Bluetooth",
        consequence=Q.REVERSIBLE,
    ),
    Intent(
        "browse_web",
        G.PRODUCTIVITY,
        L.HIGH,
        description="Open and browse a web page",
        consequence=Q.NONE,
    ),
)

INTENTS_BY_NAME: dict[str, Intent] = {i.name: i for i in INTENTS}
