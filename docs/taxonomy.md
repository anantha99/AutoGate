# AutoGate intent taxonomy

Every utterance in the AutoGate benchmark maps to exactly one intent. Each
intent carries three routing tags: **capability** (what it takes to answer), **distraction**
(how much of the driver's attention the response demands), and **actuation**
(what the car may do on request). Those three tags plus the vehicle context
fully determine the route label through the rulebook
(`autogate_bench.rulebook.route`), so this taxonomy determines every label in
the dataset. The code lives in `src/autogate_bench/intents.py`; this page gives
the reason behind each tag so it can be checked by hand.

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

- Vehicle-control intents are always `local_ok` and `low` distraction: the
  command itself is short. They are refused, if at all, by their actuation
  class.
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

81 intents: vehicle_control 28, navigation 10, communication 11, media 7,
information 11, planning 5, productivity 9.

| capability \ distraction | low | medium | high | total |
| --- | --- | --- | --- | --- |
| local_ok | 44 | 2 | 4 | 50 |
| needs_small_local | 7 | 3 | 1 | 11 |
| needs_cloud | 3 | 5 | 4 | 12 |
| cloud_preferred | 4 | 4 | 0 | 8 |
| total | 58 | 14 | 9 | 81 |

Actuation: safety_critical 8, comfort 11, restricted 9, none 53.

Consequence: none 29, reversible 42, irreversible 10. The ten irreversible
intents are `emergency_call`, `call_contact`, `send_text_message`,
`reply_to_message`, `start_video_call`, `compose_long_email`,
`share_live_location`, `book_restaurant`, `add_calendar_event`, and
`pay_bill`.

**2026-10 expansion.** Sixteen intents were added to the 65 of v0 so every
tag value has enough distinct intents to learn from once the OOD holdout is
removed (at least 6 training intents per value; see the holdout section
below). Run R2 showed why: `cloud_preferred` had 4 intents, 2 of them held
out, and the capability head could not learn a class from 2 examples. The
new intents are `fog_lights_on`, `dip_high_beams`, `cancel_cruise_control`,
`open_fuel_door`, `disable_stability_control`, `adjust_steering_wheel`,
`find_parking`, `route_avoiding_tolls`, `read_email_aloud`,
`share_live_location`, `fuel_prices_nearby`, `place_opening_hours`,
`translate_phrase`, `read_news_headlines`, `estimate_trip_cost`, and
`pay_bill`. Each is a distinct user goal, not a slot value of an existing
intent.

Which tag makes the label depend on context:

| tag | what it flips on | intents |
| --- | --- | --- |
| distraction `medium` | driving demand (refused at high demand) | 14 |
| distraction `high` | parked vs moving | 9 |
| actuation `restricted` | parked vs moving (raw demand, no passenger relaxation) | 9 |
| capability `needs_cloud` | connectivity (DEFER when offline) | 12 |
| capability `cloud_preferred` | connectivity (LOCAL when offline) | 8 |
| capability `needs_small_local` | local model tier (LOCAL on small, cloud on tiny) | 11 |

## vehicle_control

| name | capability | distraction | actuation | consequence | description | rationale |
| --- | --- | --- | --- | --- | --- | --- |
| defrost_windshield | local_ok | low | safety_critical | reversible | Turn on windshield defrost or demist | `HVAC_DEFROSTER` / `HVAC_MAX_DEFROST_ON`; fogged glass blocks the view, so any wait is itself a hazard. |
| wipers_on | local_ok | low | safety_critical | reversible | Turn on or speed up the windshield wipers | `WINDSHIELD_WIPERS_SWITCH`; water on the glass is a visibility hazard that cannot wait for a round-trip. |
| hazard_lights_on | local_ok | low | safety_critical | reversible | Turn on the hazard warning lights | `HAZARD_LIGHTS_SWITCH`; warns following traffic of a breakdown or sudden queue, which only helps if immediate. |
| headlights_on | local_ok | low | safety_critical | reversible | Turn on the headlights | `HEADLIGHTS_SWITCH`; seeing and being seen is visibility, not comfort, so it is safety-critical *(judgment)*. |
| emergency_call | local_ok | low | safety_critical | irreversible | Call emergency services from the car | No `VehiclePropertyIds` entry; treated as an eCall-style vehicle function so the safety branch guarantees it is never refused or deferred *(judgment)*. |
| fog_lights_on | local_ok | low | safety_critical | reversible | Turn on the fog lights | `FOG_LIGHTS_SWITCH` / `FRONT_FOG_LIGHTS_SWITCH` / `REAR_FOG_LIGHTS_SWITCH`; being seen in fog or a ghat-road cloud is visibility, so it cannot wait. A different goal from `headlights_on` (fog, not darkness) and a different switch. |
| dip_high_beams | local_ok | low | safety_critical | reversible | Switch the high beams off and drop to low beam | `HIGH_BEAM_LIGHTS_SWITCH`; dazzling an oncoming driver is an immediate hazard. Only the dipping direction is in scope; asking for high beam on is not split out *(judgment)*. |
| cancel_cruise_control | local_ok | low | safety_critical | reversible | Disengage cruise control | `CRUISE_CONTROL_COMMAND` (cancel) / `CRUISE_CONTROL_ENABLED`; handing speed back to the driver when traffic bunches up must never be delayed. `set_cruise_speed` stays comfort: the same system, opposite safety class *(judgment)*. |
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
| open_fuel_door | local_ok | low | restricted | reversible | Open the fuel filler door or the charging port cover | `FUEL_DOOR_OPEN` / `EV_CHARGE_PORT_OPEN`; refuelling and charging happen only when stopped, and an open flap at speed can be torn off and an unlatched cap lost. |
| disable_stability_control | local_ok | low | restricted | reversible | Turn off electronic stability or traction control | `ELECTRONIC_STABILITY_CONTROL_ENABLED`; switching off skid protection mid-drive removes a safety net; the real use (getting unstuck in sand or mud) starts from a standstill. |
| adjust_steering_wheel | local_ok | low | restricted | reversible | Move the steering wheel up, down, closer, or further away | `STEERING_WHEEL_HEIGHT_MOVE` / `STEERING_WHEEL_DEPTH_MOVE`; a power column moving under the driver's hands while steering is unsafe, which is why manual columns are adjusted parked *(judgment)*. |

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
| find_parking | cloud_preferred | medium | none | none | Find a place to park near here or near the destination | Live occupancy needs the cloud; onboard parking POIs give an answer offline, so LOCAL rather than DEFER. Comparing lots is a decision, so medium like `find_nearby_place`, but the goal (leave the car) differs *(judgment)*. |
| route_avoiding_tolls | cloud_preferred | low | none | reversible | Re-route the active navigation to avoid tolls or highways | Online routing compares alternatives with live traffic; the onboard router can still honour avoid-tolls offline, so LOCAL rather than DEFER. One short command; undone by asking for the fastest route *(judgment)*. |

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
| read_email_aloud | needs_small_local | medium | none | none | Hear recent emails read out or condensed | Emails are too long to read verbatim, so a small model condenses them for speech; long spoken output, since `UX_RESTRICTIONS_LIMIT_STRING_LENGTH` caps showing them. Differs from `read_messages` in source and length *(judgment)*. |
| share_live_location | local_ok | low | none | irreversible | Share the car's live location with a contact | A contact match plus a share command, like placing a call. Irreversible because the contact sees where the driver is *(judgment)*. |

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
| fuel_prices_nearby | cloud_preferred | low | none | none | Ask what fuel costs at stations nearby | Today's prices are live data; the last-synced prices in the onboard POI database still answer offline, so LOCAL rather than DEFER *(judgment)*. |
| place_opening_hours | cloud_preferred | low | none | none | Ask whether a place is open now or when it closes | Holiday closures and live hours need the cloud; the onboard POI database stores regular hours, so LOCAL rather than DEFER offline. |
| translate_phrase | needs_small_local | low | none | none | Translate a short phrase into another language | An 8B model translates a short phrase between Indian languages and English; a 2B model is unreliable. One phrase spoken back. |
| read_news_headlines | needs_cloud | medium | none | none | Hear the latest news headlines | Live data; a run of headlines is long spoken output. |

## planning

| name | capability | distraction | actuation | consequence | description | rationale |
| --- | --- | --- | --- | --- | --- | --- |
| plan_multi_day_trip | needs_cloud | medium | none | none | Plan a trip spanning several days | Live data plus multi-constraint reasoning; the spoken plan is long and invites decisions. |
| plan_day_itinerary | needs_cloud | medium | none | none | Plan a sequence of stops for today | Opening hours and travel times are live; ordering stops is a decision. |
| book_restaurant | needs_cloud | medium | none | irreversible | Find and reserve a table at a restaurant | Live availability and a booking service; confirming time and party size is a spoken decision, not a screen form *(judgment)*. |
| plan_charging_stops | cloud_preferred | medium | none | none | Plan charging stops for an electric-vehicle journey | Live charger availability needs the cloud; the onboard charger database gives a usable plan offline, so LOCAL rather than DEFER. |
| estimate_trip_cost | needs_small_local | low | none | none | Estimate the fuel or charging cost of a trip | Route distance (onboard maps), the car's efficiency and the last fill-up price combined in a short sum: more than a tiny model, less than the cloud. One number spoken back *(judgment)*. |

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
| pay_bill | needs_cloud | high | none | irreversible | Pay a utility, phone, or other bill | A payment service; checking the payee and amount on screen and entering a PIN or OTP is keyboard entry that `UX_RESTRICTIONS_NO_KEYBOARD` blocks. Irreversible: money leaves the account *(judgment)*. |

## Suggested OOD holdout

Eight intents held out entirely from training for the out-of-distribution
test set (`OOD_INTENTS` in `src/autogate_bench/splits.py`). They were
re-picked in the 2026-10 expansion by tag, because the first holdout took two
of the four `cloud_preferred` intents and three of the eight
`needs_small_local` ones and tested no `needs_cloud`, high-distraction,
comfort or safety-critical intent (run R2). The rules, checked in
`tests/test_splits.py`:

- every value of every tag keeps at least 6 training intents and at least
  70% of its intents in training;
- the holdout covers every value of capability, distraction, actuation and
  consequence at least once;
- at least five of the seven groups are represented (six are; media is not);
- no held-out intent is a rewording of a training intent: each has a
  different goal, so only what it shares with training intents is its tags.

| tag value | training intents | OOD intents |
| --- | --- | --- |
| capability `local_ok` | 46 | 4 |
| capability `needs_small_local` | 10 | 1 |
| capability `needs_cloud` | 10 | 2 |
| capability `cloud_preferred` | 7 | 1 |
| distraction `low` | 53 | 5 |
| distraction `medium` | 13 | 1 |
| distraction `high` | 7 | 2 |
| actuation `none` | 48 | 5 |
| actuation `comfort` | 10 | 1 |
| actuation `restricted` | 8 | 1 |
| actuation `safety_critical` | 7 | 1 |
| consequence `none` | 27 | 2 |
| consequence `reversible` | 37 | 5 |
| consequence `irreversible` | 9 | 1 |

- **set_ambient_lighting** (vehicle_control): tests `comfort` actuation;
  lighting words also appear in safety-critical intents, so the class must
  come from what the light is for, not from "lights".
- **fog_lights_on** (vehicle_control): tests `safety_critical` actuation with
  new vocabulary (fog, ghat, fog lamps); `headlights_on`, `dip_high_beams`
  and `hazard_lights_on` teach that visibility lights cannot wait. Same
  family as `headlights_on`, but a different situation and switch.
- **open_fuel_door** (vehicle_control): tests `restricted` actuation;
  `open_trunk` and `unlock_doors` teach that opening the body is parked-only,
  and the fuel words (petrol bunk, attendant) also occur in local_ok
  information intents.
- **route_avoiding_tolls** (navigation): tests `cloud_preferred` capability
  (LOCAL, not DEFER, offline) with `low` distraction and `reversible`
  consequence; no training intent changes route preferences.
- **compose_long_email** (communication): tests `irreversible` consequence
  together with `needs_cloud` and `high` distraction; `pay_bill` trains the
  same tag combination with a different goal, and `read_email_aloud` shares
  the email vocabulary with different tags (small model, medium, read-only),
  so "email" alone does not decide. Its `{contact}` seeds are the holdout's
  only CLOUD_MASKED rows.
- **read_news_headlines** (information): tests `needs_cloud` capability
  (DEFER offline) and `medium` distraction (refused at high demand) with
  `none` consequence; `weather_now` and `sports_score` teach live data,
  `explain_topic_in_depth` long output.
- **estimate_trip_cost** (planning): tests `needs_small_local` capability
  (the local-tier flip) with `none` consequence; fuel words overlap with
  `fuel_range` (local_ok) and `fuel_prices_nearby` (cloud_preferred), a hard
  boundary for the capability tag.
- **pair_bluetooth_device** (productivity): tests `high` distraction
  (parked vs moving) through `UX_RESTRICTIONS_NO_SETUP`; no other intent is a
  setup flow, so it must transfer from the screen-heavy intents.
