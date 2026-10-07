# AutoGate intent taxonomy

Every utterance in the AutoGate benchmark maps to exactly one intent. Each
intent carries two routing tags: **distraction** (how much of the driver's
attention the response demands) and **actuation** (what the car may do on
request). Its third input to the route, **capability** (what it takes to
answer), depends on the OEM's onboard model, so it lives in the labeling
policy as Table 4 (`autogate_bench.policy.DEFAULT_CAPABILITY`) where an OEM
can widen it. The capability column on this page is the default policy's.
Tags, capability and the vehicle context fully determine the route label
through the rulebook (`autogate_bench.rulebook.route`). The tags live in
`src/autogate_bench/intents.py`; this page gives the reason behind each tag
and default capability so it can be checked by hand.

**Sources.** Public Android Automotive OS documentation only:
[`VehiclePropertyIds`](https://developer.android.com/reference/android/car/VehiclePropertyIds)
for vehicle controls,
[`CarUxRestrictions`](https://developer.android.com/reference/android/car/drivingstate/CarUxRestrictions)
for what is blocked while moving, `CarAudioManager` for volume, and the AAOS
voice interaction docs (`CarVoiceInteractionSession` read and reply
notification actions) for messaging. Everything else is common sense. No
proprietary supplier or OEM intent list was used or reconstructed.

**Rules the tags follow.**

- `needs_cloud` means the request cannot be answered without the cloud, so
  offline it is DEFER. `cloud_preferred` means the cloud answers it better but
  onboard data can answer it now, so offline it is LOCAL. The degradation
  study relies on this split: forcing connectivity to none must shift the
  first group to DEFER and the second to LOCAL.
- Every intent also carries a `consequence` tag: `none` (read-only),
  `reversible` (the driver can undo it with one more utterance), or
  `irreversible` (another party sees it: a sent message, a placed call, a
  booking, an invite). It never changes the route. The deterministic policy
  gate reads it after the router: an execute route runs only after
  confirmation when the intent is irreversible or its actuation is
  restricted, and safety-critical commands are never held for confirmation.
  Confirmation never relaxes a REFUSE.

- Vehicle-control intents are always `low` distraction: the command itself
  is short. They are refused, if at all, by their actuation class. Every
  policy must keep them `local_ok`; `Policy.validate` rejects one that
  does not.
- Every other intent has actuation `none` and is refused, if at all, by its
  distraction level. No intent is both `restricted` and above `low`
  distraction, so the two REFUSE mechanisms never overlap.
- `high` distraction mirrors the AAOS UX restrictions that apply while moving
  (`NO_VIDEO`, `NO_SETUP`, `NO_KEYBOARD`, `LIMIT_CONTENT`,
  `NO_VOICE_TRANSCRIPTION`). Screen-heavy setup flows such as Bluetooth
  pairing are therefore productivity intents with `high` distraction, not
  vehicle commands.

Rows marked *(judgment)* are tags that could reasonably go the other way.

## Summary

65 intents: vehicle_control 22, navigation 8, communication 9, media 7,
information 7, planning 4, productivity 8.

| capability \ distraction | low | medium | high | total |
| --- | --- | --- | --- | --- |
| local_ok | 37 | 2 | 4 | 43 |
| needs_small_local | 5 | 2 | 1 | 8 |
| needs_cloud | 3 | 4 | 3 | 10 |
| cloud_preferred | 1 | 3 | 0 | 4 |
| total | 46 | 11 | 8 | 65 |

Actuation: safety_critical 5, comfort 11, restricted 6, none 43.

Consequence: none 22, reversible 35, irreversible 8. The eight irreversible
intents are `emergency_call`, `call_contact`, `send_text_message`,
`reply_to_message`, `start_video_call`, `compose_long_email`,
`book_restaurant`, and `add_calendar_event`.

Which tag makes the label depend on context:

| tag | what it flips on | intents |
| --- | --- | --- |
| distraction `medium` | driving demand (refused at high demand) | 11 |
| distraction `high` | parked vs moving | 8 |
| actuation `restricted` | parked vs moving (raw demand, no passenger relaxation) | 6 |
| capability `needs_cloud` | connectivity (DEFER when offline) | 10 |
| capability `cloud_preferred` | connectivity (LOCAL when offline) | 4 |
| capability `needs_small_local` | local model tier (LOCAL on small, cloud on tiny) | 8 |

## vehicle_control

| name | capability | distraction | actuation | consequence | description | rationale |
| --- | --- | --- | --- | --- | --- | --- |
| defrost_windshield | local_ok | low | safety_critical | reversible | Turn on windshield defrost or demist | `HVAC_DEFROSTER` / `HVAC_MAX_DEFROST_ON`; fogged glass blocks the view, so any wait is itself a hazard. |
| wipers_on | local_ok | low | safety_critical | reversible | Turn on or speed up the windshield wipers | `WINDSHIELD_WIPERS_SWITCH`; water on the glass is a visibility hazard that cannot wait for a round-trip. |
| hazard_lights_on | local_ok | low | safety_critical | reversible | Turn on the hazard warning lights | `HAZARD_LIGHTS_SWITCH`; warns following traffic of a breakdown or sudden queue, which only helps if immediate. |
| headlights_on | local_ok | low | safety_critical | reversible | Turn on the headlights | `HEADLIGHTS_SWITCH`; seeing and being seen is visibility, not comfort, so it is safety-critical *(judgment)*. |
| emergency_call | local_ok | low | safety_critical | irreversible | Call emergency services from the car | No `VehiclePropertyIds` entry; treated as an eCall-style vehicle function so the safety branch guarantees it is never refused or deferred *(judgment)*. |
| set_cabin_temperature | local_ok | low | comfort | reversible | Set the HVAC temperature | `HVAC_TEMPERATURE_SET`; routinely adjusted while moving. |
| set_fan_speed | local_ok | low | comfort | reversible | Change the HVAC fan speed | `HVAC_FAN_SPEED`; routinely adjusted while moving. |
| climate_on_off | local_ok | low | comfort | reversible | Turn the air conditioning or climate system on or off | `HVAC_AC_ON` / `HVAC_POWER_ON`; routinely toggled while moving. |
| set_recirculation | local_ok | low | comfort | reversible | Switch between recirculated and fresh cabin air | `HVAC_RECIRC_ON`; routinely toggled while moving, for example in a tunnel. |
| seat_ventilation_on | local_ok | low | comfort | reversible | Turn on or adjust a ventilated (cooled) seat | `HVAC_SEAT_VENTILATION`; the cooling counterpart of the seat heater. |
| seat_heater_on | local_ok | low | comfort | reversible | Turn on or adjust a seat heater | `HVAC_SEAT_TEMPERATURE`; routinely adjusted while moving. |
| adjust_window | local_ok | low | comfort | reversible | Open or close a window, fully or partly | `WINDOW_MOVE` (with `WINDOW_POS` for state); drivers open windows on the move and closing has pinch protection. |
| set_audio_volume | local_ok | low | comfort | reversible | Turn the audio volume up or down | `CarAudioManager.setGroupVolume`; the README's actuation table lists volume under comfort, so it is a vehicle command, not media *(judgment)*. |
| set_drive_mode | local_ok | low | comfort | reversible | Switch drive mode, such as eco, normal, or sport | No standard `VehiclePropertyIds` entry (OEM vendor property); physical drive-mode selectors work on the move *(judgment)*. |
| set_ambient_lighting | local_ok | low | comfort | reversible | Change the interior ambient lighting | Closest public properties are `CABIN_LIGHTS_SWITCH`, `SEAT_FOOTWELL_LIGHTS_SWITCH`, `STEERING_WHEEL_LIGHTS_SWITCH`; cosmetic. |
| set_cruise_speed | local_ok | low | comfort | reversible | Raise or lower the cruise control set speed | `CRUISE_CONTROL_COMMAND` (increase or decrease target speed); the stalk already does this on the move, unlike switching assistance off *(judgment)*. |
| disable_lane_assist | local_ok | low | restricted | reversible | Turn off lane keeping assist | `LANE_KEEP_ASSIST_ENABLED`; switching off a driver-assistance system mid-drive removes a safety net. |
| unlock_doors | local_ok | low | restricted | reversible | Unlock the doors | `DOOR_LOCK`; unlocked doors while moving invite accidental opening and intrusion at stops. |
| open_trunk | local_ok | low | restricted | reversible | Open the trunk or tailgate | `DOOR_POS` on the rear door area; an open trunk while moving can drop cargo and blocks the rear view. |
| set_parking_brake | local_ok | low | restricted | reversible | Apply or release the parking brake | `PARKING_BRAKE_ON` is read-only in the public API, so actuation is OEM-specific; applying at speed is an unplanned emergency stop, so both directions are parked-only *(judgment)*. |
| disable_child_locks | local_ok | low | restricted | reversible | Turn off the rear-door child locks | `DOOR_CHILD_LOCK_ENABLED`; would let a child open a rear door while moving. |
| fold_mirrors | local_ok | low | restricted | reversible | Fold or unfold the side mirrors | `MIRROR_FOLD`; folded mirrors remove side and rear view; unfolding is harmless but not split out, so both are parked-only *(judgment)*. |

## navigation

| name | capability | distraction | actuation | consequence | description | rationale |
| --- | --- | --- | --- | --- | --- | --- |
| next_turn | local_ok | low | none | none | Ask what the next manoeuvre on the active route is | Read from the running navigation engine; one short sentence. |
| cancel_route | local_ok | low | none | reversible | Stop the active navigation | A single command to the navigation app. |
| navigate_home | local_ok | low | none | reversible | Start navigation to a saved place such as home or work | Saved destination plus onboard maps; no search or live data needed *(judgment)*. |
| eta_to_destination | local_ok | low | none | none | Ask when the car will arrive at the current destination | The active navigation engine already holds the ETA; the assistant only reads it *(judgment)*. |
| navigate_to_contact_address | cloud_preferred | low | none | reversible | Start navigation to a contact's address | Online maps geocode a free-text address best, but onboard maps can do it offline, so LOCAL rather than DEFER when there is no signal. The address is a sensitive span, so connected this is the standard CLOUD_MASKED example. |
| find_nearby_place | cloud_preferred | medium | none | none | Find and compare nearby places such as restaurants or fuel stations | Live ratings and opening hours need the cloud, but onboard POI data answers offline, so LOCAL rather than DEFER when there is no signal. |
| traffic_on_route | needs_cloud | low | none | none | Ask about traffic or delays on the current route | Live traffic data, answered in one sentence. |
| add_stop_along_route | cloud_preferred | medium | none | reversible | Add a stop, such as coffee or a charger, along the current route | Cloud search finds the best stop; the onboard POI database can still add one offline, so LOCAL rather than DEFER. |

## communication

| name | capability | distraction | actuation | consequence | description | rationale |
| --- | --- | --- | --- | --- | --- | --- |
| call_contact | local_ok | low | none | irreversible | Place a phone call to a contact | Phonebook match plus dial; voice replaces the dialpad that `UX_RESTRICTIONS_NO_DIALPAD` blocks. |
| lookup_contact_info | local_ok | low | none | none | Ask for a contact's phone number or address | Local phonebook lookup with a short spoken answer. |
| send_text_message | needs_small_local | low | none | irreversible | Dictate and send a short text message | The dictation must be rephrased and addressed ("tell her I'm late"), which needs more than a tiny model; spoken only, as `UX_RESTRICTIONS_NO_TEXT_MESSAGE` blocks showing it. |
| reply_to_message | needs_small_local | low | none | irreversible | Dictate a short reply to a message just read out | `CarVoiceInteractionSession` reply-notification action; the reply is one short utterance, reading the thread is a separate intent *(judgment)*. |
| read_messages | needs_small_local | medium | none | none | Read recent unread messages aloud | `CarVoiceInteractionSession` read-notification action; long spoken output, since `UX_RESTRICTIONS_NO_TEXT_MESSAGE` blocks showing them. |
| summarize_group_chat | needs_small_local | medium | none | none | Summarise what was said in a group conversation | Summarising several senders needs a small model; the spoken summary is long. |
| play_voicemail | local_ok | medium | none | none | Play new voicemail messages | Playback is trivial, but listening to several messages is long output *(judgment)*. |
| start_video_call | local_ok | high | none | irreversible | Start a video call with a contact | `UX_RESTRICTIONS_NO_VIDEO`; placing the call is a plain command, watching it is not allowed while moving. |
| compose_long_email | needs_cloud | high | none | irreversible | Dictate and edit a long email | Long composition needs a frontier model; multi-edit dictation needs the draft on screen, which `UX_RESTRICTIONS_NO_VOICE_TRANSCRIPTION` blocks. |

## media

| name | capability | distraction | actuation | consequence | description | rationale |
| --- | --- | --- | --- | --- | --- | --- |
| play_music | local_ok | low | none | reversible | Play a song, artist, album, or playlist | Hand-off to a media app's `MediaSession`; audio media apps are distraction-optimised. |
| pause_playback | local_ok | low | none | reversible | Pause or resume the current media | `MediaSession` transport control. |
| skip_track | local_ok | low | none | reversible | Skip to the next or previous track | `MediaSession` transport control. |
| tune_radio_station | local_ok | low | none | reversible | Tune the radio to a station or frequency | A tune command to the broadcast radio service. |
| play_podcast | local_ok | low | none | reversible | Play a podcast or a specific episode | The media app resolves the catalog search; the assistant only hands it off *(judgment)*. |
| play_video | local_ok | high | none | reversible | Play a video on the screen | `UX_RESTRICTIONS_NO_VIDEO`. |
| show_photos | local_ok | high | none | none | Browse photos on the screen | Photo browsing is not distraction-optimised and is capped by `UX_RESTRICTIONS_LIMIT_CONTENT`. |

## information

| name | capability | distraction | actuation | consequence | description | rationale |
| --- | --- | --- | --- | --- | --- | --- |
| weather_now | needs_cloud | low | none | none | Ask about the current or upcoming weather | Live data; one-sentence answer. |
| current_time | local_ok | low | none | none | Ask the time or date | System clock. |
| sports_score | needs_cloud | low | none | none | Ask for a live or recent sports score | Live data; one-sentence answer. |
| general_knowledge_question | needs_small_local | low | none | none | Ask a short factual question | An 8B model answers common facts reliably; a 2B model hallucinates too often. |
| explain_topic_in_depth | needs_cloud | medium | none | none | Ask for a long explanation of a topic | Needs a frontier model; the answer is long spoken output. |
| vehicle_manual_question | needs_small_local | low | none | none | Ask how a car feature works or what a warning light means | Retrieval over an onboard owner's manual fits an 8B model, not a 2B one *(judgment)*. |
| fuel_range | local_ok | low | none | none | Ask how far the car can go on the remaining fuel or charge | `RANGE_REMAINING`, `FUEL_LEVEL`, `EV_BATTERY_LEVEL`. |

## planning

| name | capability | distraction | actuation | consequence | description | rationale |
| --- | --- | --- | --- | --- | --- | --- |
| plan_multi_day_trip | needs_cloud | medium | none | none | Plan a trip spanning several days | Live data plus multi-constraint reasoning; the spoken plan is long and invites decisions. |
| plan_day_itinerary | needs_cloud | medium | none | none | Plan a sequence of stops for today | Opening hours and travel times are live; ordering stops is a decision. |
| book_restaurant | needs_cloud | medium | none | irreversible | Find and reserve a table at a restaurant | Live availability and a booking service; confirming time and party size is a spoken decision, not a screen form *(judgment)*. |
| plan_charging_stops | cloud_preferred | medium | none | none | Plan charging stops for an electric-vehicle journey | Live charger availability needs the cloud; the onboard charger database gives a usable plan offline, so LOCAL rather than DEFER. |

## productivity

| name | capability | distraction | actuation | consequence | description | rationale |
| --- | --- | --- | --- | --- | --- | --- |
| calendar_today | local_ok | medium | none | none | Hear today's calendar | A templated read-out of the synced calendar, so local_ok; a full day is long output *(judgment)*. |
| add_reminder | local_ok | low | none | reversible | Set a reminder for a time or place | Time plus free-text slot filling, which a tiny model or command handler does *(judgment)*. |
| add_calendar_event | needs_small_local | low | none | irreversible | Add an event to the calendar | Resolving date, duration and attendees together exceeds a tiny model *(judgment)*. |
| take_note | local_ok | low | none | reversible | Dictate a short note to save | Saved verbatim, unlike a text message that must be rephrased *(judgment)*. |
| edit_note | needs_small_local | high | none | reversible | Make several spoken edits to an existing note | Multi-edit dictation needs the transcript on screen to check each edit, which `UX_RESTRICTIONS_NO_VOICE_TRANSCRIPTION` blocks *(judgment)*. |
| read_document | needs_cloud | high | none | none | Open and read a document on the screen | `UX_RESTRICTIONS_LIMIT_STRING_LENGTH`; documents live in cloud storage and exceed a local model's context *(judgment)*. |
| pair_bluetooth_device | local_ok | high | none | reversible | Pair a new phone or device over Bluetooth | `UX_RESTRICTIONS_NO_SETUP` (setup with external devices); moved out of vehicle_control because it is a screen flow, not an actuation. |
| browse_web | needs_cloud | high | none | none | Open and browse a web page | `UX_RESTRICTIONS_NO_KEYBOARD` and `LIMIT_CONTENT`; a browser is not distraction-optimised. |

## Suggested OOD holdout

Six intents, one from each of six groups, that could be held out entirely from
training for the out-of-distribution test set. Each one's tag combination is
still represented by other training intents, so a router that has learned the
tags rather than the surface wording should label them correctly.

- **disable_child_locks** (vehicle_control): new vocabulary for a restricted
  actuation; five other restricted intents teach the parked-only pattern.
- **add_stop_along_route** (navigation): an en-route variant of place search;
  tests whether cloud plus medium distraction generalises beyond
  `find_nearby_place`.
- **summarize_group_chat** (communication): tests needs_small_local plus
  medium distraction from `read_messages` alone, so both the tier flip and the
  demand flip must transfer.
- **vehicle_manual_question** (information): car-specific wording that looks
  like a vehicle command but is a small-local question with no actuation.
- **plan_charging_stops** (planning): EV vocabulary (charge, kWh, chargers)
  absent from the other planning intents.
- **add_calendar_event** (productivity): close in wording to `add_reminder`
  (local_ok) but needs a small model, a hard boundary for the tier flip.
