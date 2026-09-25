# Paraphrase review notes

Lines the generation agents flagged as worth a human's eye, collected from
their reports. Every file passes strict validation (0 errors); these are
judgment calls, not rule violations. Fix lines in place (never reorder or
delete; line position is part of the row id) and re-run
`uv run python -m autogate_bench.paraphrases validate --dir data/paraphrases --strict`.

## Needs a native speaker

All Kannada-English lines were written by the agents from common Bengaluru
spellings. Words they were least sure of: muchchi bidtira, chuchta ide,
kelisode illa, malgbittide, koothkotaane, eleyuttide, hidkotide, jaaruttide,
nillolla, etthi, hechchu, sahaaya, kettogide, bisilalli, odisbahudu, sigakke,
mooru dina raja, thale ge hogalla, arthaa maadsi, maadbahuda, tegiyoke aagutta,
talupbeku, ilidakke aagtilla, muchchu, torisu, torsthiya, tegdiro, yethara
betta, janasankhye, tegiri, kaagada, saakagalla, nenapisthiya, madhyaahna,
gottagbeku, marethogthini, nilsida kade inda, munche bandidda haadu.
Hinglish lines are more standard; "dipper" and "batti" (headlights) are
regional.

## Regional English the reviewer may want to confirm

- "dickey" (boot), "petrol bunk", "on the wipers / on the hazards" (turn on),
  "put the ac", "{contact}'s side" (someone's place), "block a table".
- "parking lights" and "emergency lights" used for hazards.

## Self-corrections that name a neighbouring intent first

The spec allows a restart that names the wrong thing and ends on the right
one. The keyword matcher scores some of these 1.00 against the wrong intent.
Examples: "wipers no no i mean hazards turn them on" (hazard_lights_on),
"seat cooling er no seat heating" (seat_heater_on), "warm up no i mean cool
down the {seat} seat" (seat_ventilation_on), "cruise off no no not cruise i
mean lane assist off", "hazards on no sorry i meant handbrake on", "remind me
no sorry calendar {event_title} for {date} {time}" (add_calendar_event),
"fuel stops no sorry charging stops for {city}", "remind me no wait just
write a note {note_text}", "skip the uh no just pause the song",
"take me to the office no home" (navigate_home). Keep or rewrite as a policy
decision: they are realistic and hard, but they put a sibling intent's
keyword into this intent's training data.

## Lines that only weakly imply the request (indirect / outcome style)

- adjust_window: "this is the plaza they need the ticket", "the barrier's not
  lifting he needs to see the pass".
- pause_playback: "the guard just walked up to the car".
- play_video: "the kids are restless and mom's shopping is taking forever".
- read_document: "i must be ready with the slides for the meeting".
- cancel_route: "that's off the table now", "that's been called off".
- send_text_message: "{contact} needs {otp} to finish the booking".
- reply_to_message: "the ball's in my court with {contact}", "i've left
  {contact} on read".
- start_video_call: "i want {contact} to see the new house", "the children
  keep calling out for their granny".
- add_stop_along_route: "we should have a tea pause before we reach".
- find_nearby_place: "i need a clean washroom", "the kids want ice cream"
  (from the seeds themselves).

## Thin fragments

Short seeds forced some very short lines: "home now / home again / just home /
saved home / home time" (navigate_home); "{contact} quickly", "try {contact}"
(call_contact); "medicine time {time}" (add_reminder, no reminder word);
"that abs warning" (vehicle_manual_question); "{temperature} flat",
"{temperature} that's it" (set_cabin_temperature, read like answers);
"time sir", "timing", "dash clock" (current_time); "rescue", "get help fast"
(emergency_call); "play", "go on" (pause_playback resume seed, could read as
play_music).

## Possible slot or scope stretches

- set_audio_volume: "louder till {number}" assumes direction.
- set_cabin_temperature: "put the ac on {temperature} degrees" overlaps
  climate_on_off wording.
- add_calendar_event: "lunch is {date} {time}" adds a meal type; some
  fragments add "team" or "office".
- book_restaurant: "for tonight's dinner" adds a time the seed lacks.
- weather_now: "{city} humidity now", "{city} heat today" narrow the seed.
- play_video: "{topic} film / documentary / reel / vlog / on youtube" narrow
  "video".
- plan_multi_day_trip: "for um three no {duration}" has a literal number
  before the correction.
- emergency_call: 108 and 112 deliberately never appear, though drivers say
  "call one oh eight".

## Deliberate hard cases the agents kept

- unlock_doors: "central lock off", "open the locks on every door" (near
  disable_child_locks).
- vehicle_manual_question: "go through how auto hold works when i'm on a
  slope" (near explain_topic_in_depth).
- disable_child_locks seed 2 lines about a rear passenger who cannot open the
  door (near unlock_doors; the seed itself has this overlap).
