"""The AutoGate intent taxonomy: 62 intents in seven groups.

Every benchmark utterance maps to exactly one intent. The intent's three tags
(capability, distraction, actuation) plus the vehicle context fully determine
the route label via ``rulebook.route``, so these tags determine every label
in the dataset.

Sources: public Android Automotive OS documentation only (VehiclePropertyIds,
CarUxRestrictions, CarAudioManager, the AAOS voice interaction docs) and
common sense. No proprietary supplier or OEM intent list was used.

Invariants (checked in ``tests/test_intents.py``):

* Vehicle-control intents are LOCAL_OK and LOW distraction: the command
  itself is short. They are refused, if at all, by their actuation class.
* Everything else is refused, if at all, by its distraction level. No intent
  is both RESTRICTED and above LOW distraction, so the two REFUSE mechanisms
  never overlap.
* Screen-heavy setup flows (device pairing) live in productivity with HIGH
  distraction and no actuation class, mirroring ``UX_RESTRICTIONS_NO_SETUP``.

A trailing ``#`` comment marks a tag that is a judgment call. The per-intent
rationale, with the AAOS property or UX restriction each maps to, is in
``docs/taxonomy.md``.
"""

from __future__ import annotations

from autogate_bench.schema import ActuationClass as A
from autogate_bench.schema import Capability as C
from autogate_bench.schema import DistractionLevel as L
from autogate_bench.schema import Intent
from autogate_bench.schema import IntentGroup as G

INTENTS: tuple[Intent, ...] = (
    # ======================================================================= #
    # vehicle_control (19): all LOCAL_OK, all LOW distraction
    # ======================================================================= #
    # ---- safety_critical (5): a delay could itself be dangerous ------------ #
    Intent(
        "defrost_windshield",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.SAFETY_CRITICAL,
        "Turn on windshield defrost or demist",
    ),
    Intent(
        "wipers_on",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.SAFETY_CRITICAL,
        "Turn on or speed up the windshield wipers",
    ),
    Intent(
        "hazard_lights_on",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.SAFETY_CRITICAL,
        "Turn on the hazard warning lights",
    ),
    Intent(
        "headlights_on",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.SAFETY_CRITICAL,
        "Turn on the headlights",
    ),  # judgment: headlights are visibility, not comfort; driving unlit at dusk is the hazard
    Intent(
        "emergency_call",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.SAFETY_CRITICAL,
        "Call emergency services from the car",
    ),  # judgment: an eCall-style vehicle function, so it bypasses every gate and never DEFERs
    # ---- comfort (8): adjusted routinely while moving ---------------------- #
    Intent(
        "set_cabin_temperature",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.COMFORT,
        "Set the HVAC temperature",
    ),
    Intent(
        "set_fan_speed",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.COMFORT,
        "Change the HVAC fan speed",
    ),
    Intent(
        "seat_heater_on",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.COMFORT,
        "Turn on or adjust a seat heater",
    ),
    Intent(
        "adjust_window",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.COMFORT,
        "Open or close a window, fully or partly",
    ),
    Intent(
        "set_audio_volume",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.COMFORT,
        "Turn the audio volume up or down",
    ),  # judgment: volume is a car-audio actuation (CarAudioManager), so vehicle_control not media
    Intent(
        "set_drive_mode",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.COMFORT,
        "Switch drive mode, such as eco, normal, or sport",
    ),  # judgment: drive-mode selectors work on the move; no standard VHAL property exists
    Intent(
        "set_ambient_lighting",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.COMFORT,
        "Change the interior ambient lighting",
    ),
    Intent(
        "set_cruise_speed",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.COMFORT,
        "Raise or lower the cruise control set speed",
    ),  # judgment: nudging an engaged system's speed is routine; turning assistance off is not
    # ---- restricted (6): only when parked ---------------------------------- #
    Intent(
        "disable_lane_assist",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.RESTRICTED,
        "Turn off lane keeping assist",
    ),
    Intent(
        "unlock_doors",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.RESTRICTED,
        "Unlock the doors",
    ),
    Intent(
        "open_trunk",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.RESTRICTED,
        "Open the trunk or tailgate",
    ),
    Intent(
        "set_parking_brake",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.RESTRICTED,
        "Apply or release the parking brake",
    ),  # judgment: both directions restricted; applying at speed is an unplanned emergency stop
    Intent(
        "disable_child_locks",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.RESTRICTED,
        "Turn off the rear-door child locks",
    ),
    Intent(
        "fold_mirrors",
        G.VEHICLE_CONTROL,
        C.LOCAL_OK,
        L.LOW,
        A.RESTRICTED,
        "Fold or unfold the side mirrors",
    ),  # judgment: folding removes rear visibility while moving; unfold-only is not split out
    # ======================================================================= #
    # navigation (8)
    # ======================================================================= #
    Intent(
        "next_turn",
        G.NAVIGATION,
        C.LOCAL_OK,
        L.LOW,
        description="Ask what the next manoeuvre on the active route is",
    ),
    Intent(
        "cancel_route",
        G.NAVIGATION,
        C.LOCAL_OK,
        L.LOW,
        description="Stop the active navigation",
    ),
    Intent(
        "navigate_home",
        G.NAVIGATION,
        C.LOCAL_OK,
        L.LOW,
        description="Start navigation to a saved place such as home or work",
    ),  # judgment: saved destination plus onboard maps; no search or live data needed
    Intent(
        "eta_to_destination",
        G.NAVIGATION,
        C.LOCAL_OK,
        L.LOW,
        description="Ask when the car will arrive at the current destination",
    ),  # judgment: the active nav engine already holds the ETA; the assistant only reads it
    Intent(
        "navigate_to_contact_address",
        G.NAVIGATION,
        C.NEEDS_CLOUD,
        L.LOW,
        description="Start navigation to a contact's address",
    ),  # judgment: geocoding a free-text address uses online maps; the address is a sensitive span
    Intent(
        "find_nearby_place",
        G.NAVIGATION,
        C.NEEDS_CLOUD,
        L.MEDIUM,
        description="Find and compare nearby places such as restaurants or fuel stations",
    ),
    Intent(
        "traffic_on_route",
        G.NAVIGATION,
        C.NEEDS_CLOUD,
        L.LOW,
        description="Ask about traffic or delays on the current route",
    ),
    Intent(
        "add_stop_along_route",
        G.NAVIGATION,
        C.NEEDS_CLOUD,
        L.MEDIUM,
        description="Add a stop, such as coffee or a charger, along the current route",
    ),
    # ======================================================================= #
    # communication (9)
    # ======================================================================= #
    Intent(
        "call_contact",
        G.COMMUNICATION,
        C.LOCAL_OK,
        L.LOW,
        description="Place a phone call to a contact",
    ),
    Intent(
        "lookup_contact_info",
        G.COMMUNICATION,
        C.LOCAL_OK,
        L.LOW,
        description="Ask for a contact's phone number or address",
    ),
    Intent(
        "send_text_message",
        G.COMMUNICATION,
        C.NEEDS_SMALL_LOCAL,
        L.LOW,
        description="Dictate and send a short text message",
    ),
    Intent(
        "reply_to_message",
        G.COMMUNICATION,
        C.NEEDS_SMALL_LOCAL,
        L.LOW,
        description="Dictate a short reply to a message just read out",
    ),  # judgment: LOW because the reply is one short utterance; reading the thread is separate
    Intent(
        "read_messages",
        G.COMMUNICATION,
        C.NEEDS_SMALL_LOCAL,
        L.MEDIUM,
        description="Read recent unread messages aloud",
    ),
    Intent(
        "summarize_group_chat",
        G.COMMUNICATION,
        C.NEEDS_SMALL_LOCAL,
        L.MEDIUM,
        description="Summarise what was said in a group conversation",
    ),
    Intent(
        "play_voicemail",
        G.COMMUNICATION,
        C.LOCAL_OK,
        L.MEDIUM,
        description="Play new voicemail messages",
    ),  # judgment: playback is trivial (LOCAL_OK) but the listening is long output (MEDIUM)
    Intent(
        "start_video_call",
        G.COMMUNICATION,
        C.LOCAL_OK,
        L.HIGH,
        description="Start a video call with a contact",
    ),
    Intent(
        "compose_long_email",
        G.COMMUNICATION,
        C.NEEDS_CLOUD,
        L.HIGH,
        description="Dictate and edit a long email",
    ),
    # ======================================================================= #
    # media (7)
    # ======================================================================= #
    Intent(
        "play_music",
        G.MEDIA,
        C.LOCAL_OK,
        L.LOW,
        description="Play a song, artist, album, or playlist",
    ),
    Intent(
        "pause_playback",
        G.MEDIA,
        C.LOCAL_OK,
        L.LOW,
        description="Pause or resume the current media",
    ),
    Intent(
        "skip_track",
        G.MEDIA,
        C.LOCAL_OK,
        L.LOW,
        description="Skip to the next or previous track",
    ),
    Intent(
        "tune_radio_station",
        G.MEDIA,
        C.LOCAL_OK,
        L.LOW,
        description="Tune the radio to a station or frequency",
    ),
    Intent(
        "play_podcast",
        G.MEDIA,
        C.LOCAL_OK,
        L.LOW,
        description="Play a podcast or a specific episode",
    ),  # judgment: the media app resolves the catalog search; the assistant only hands it off
    Intent(
        "play_video",
        G.MEDIA,
        C.LOCAL_OK,
        L.HIGH,
        description="Play a video on the screen",
    ),
    Intent(
        "show_photos",
        G.MEDIA,
        C.LOCAL_OK,
        L.HIGH,
        description="Browse photos on the screen",
    ),
    # ======================================================================= #
    # information (7)
    # ======================================================================= #
    Intent(
        "weather_now",
        G.INFORMATION,
        C.NEEDS_CLOUD,
        L.LOW,
        description="Ask about the current or upcoming weather",
    ),
    Intent(
        "current_time",
        G.INFORMATION,
        C.LOCAL_OK,
        L.LOW,
        description="Ask the time or date",
    ),
    Intent(
        "sports_score",
        G.INFORMATION,
        C.NEEDS_CLOUD,
        L.LOW,
        description="Ask for a live or recent sports score",
    ),
    Intent(
        "general_knowledge_question",
        G.INFORMATION,
        C.NEEDS_SMALL_LOCAL,
        L.LOW,
        description="Ask a short factual question",
    ),
    Intent(
        "explain_topic_in_depth",
        G.INFORMATION,
        C.NEEDS_CLOUD,
        L.MEDIUM,
        description="Ask for a long explanation of a topic",
    ),
    Intent(
        "vehicle_manual_question",
        G.INFORMATION,
        C.NEEDS_SMALL_LOCAL,
        L.LOW,
        description="Ask how a car feature works or what a warning light means",
    ),  # judgment: retrieval over an onboard owner's manual fits an 8B model, not a 2B one
    Intent(
        "fuel_range",
        G.INFORMATION,
        C.LOCAL_OK,
        L.LOW,
        description="Ask how far the car can go on the remaining fuel or charge",
    ),
    # ======================================================================= #
    # planning (4): all NEEDS_CLOUD, all MEDIUM
    # ======================================================================= #
    Intent(
        "plan_multi_day_trip",
        G.PLANNING,
        C.NEEDS_CLOUD,
        L.MEDIUM,
        description="Plan a trip spanning several days",
    ),
    Intent(
        "plan_day_itinerary",
        G.PLANNING,
        C.NEEDS_CLOUD,
        L.MEDIUM,
        description="Plan a sequence of stops for today",
    ),
    Intent(
        "book_restaurant",
        G.PLANNING,
        C.NEEDS_CLOUD,
        L.MEDIUM,
        description="Find and reserve a table at a restaurant",
    ),  # judgment: MEDIUM not HIGH; confirming time and party size is a spoken decision
    Intent(
        "plan_charging_stops",
        G.PLANNING,
        C.NEEDS_CLOUD,
        L.MEDIUM,
        description="Plan charging stops for an electric-vehicle journey",
    ),
    # ======================================================================= #
    # productivity (8)
    # ======================================================================= #
    Intent(
        "calendar_today",
        G.PRODUCTIVITY,
        C.LOCAL_OK,
        L.MEDIUM,
        description="Hear today's calendar",
    ),  # judgment: LOCAL_OK because it is a templated read-out of the synced calendar
    Intent(
        "add_reminder",
        G.PRODUCTIVITY,
        C.LOCAL_OK,
        L.LOW,
        description="Set a reminder for a time or place",
    ),  # judgment: time plus free-text slot filling, which a tiny model handles
    Intent(
        "add_calendar_event",
        G.PRODUCTIVITY,
        C.NEEDS_SMALL_LOCAL,
        L.LOW,
        description="Add an event to the calendar",
    ),  # judgment: dates, durations, and attendees together exceed a tiny model
    Intent(
        "take_note",
        G.PRODUCTIVITY,
        C.LOCAL_OK,
        L.LOW,
        description="Dictate a short note to save",
    ),  # judgment: LOCAL_OK because the note is saved verbatim, unlike a message that is rephrased
    Intent(
        "edit_note",
        G.PRODUCTIVITY,
        C.NEEDS_SMALL_LOCAL,
        L.HIGH,
        description="Make several spoken edits to an existing note",
    ),  # judgment: multi-edit dictation needs the transcript on screen to check each edit
    Intent(
        "read_document",
        G.PRODUCTIVITY,
        C.NEEDS_CLOUD,
        L.HIGH,
        description="Open and read a document on the screen",
    ),  # judgment: NEEDS_CLOUD because documents live in cloud storage and exceed local context
    Intent(
        "pair_bluetooth_device",
        G.PRODUCTIVITY,
        C.LOCAL_OK,
        L.HIGH,
        description="Pair a new phone or device over Bluetooth",
    ),
    Intent(
        "browse_web",
        G.PRODUCTIVITY,
        C.NEEDS_CLOUD,
        L.HIGH,
        description="Open and browse a web page",
    ),
)

INTENTS_BY_NAME: dict[str, Intent] = {i.name: i for i in INTENTS}
