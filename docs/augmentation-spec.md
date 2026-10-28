# Augmentation specification: synthetic goals and minimal pairs

You are a generation agent. You will write **training-only** data of two kinds
into `data/augment/`, check it with the validator, review the risk page, and
commit. Read this page and `docs/paraphrase-spec.md` (sections 3 to 6: the
voice, the variants and the text rules are the same) before writing anything.

## 1. Why

Run R4 (`docs/experiment-log.md`) left three gaps:

- On the 8 OOD intents, 63 to 74% of high-distraction rows are predicted low
  or medium: the model has seen only 7 high-distraction intents.
- The adversarial slice (restricted commands asked as questions) has 25 to
  30% critical misses.
- Capability accuracy on OOD is 0.50.

Two sources address them:

- **Synthetic goals** ("reverse queries"): new in-car goals, invented for a
  given tag combination, each with 8 to 12 utterances. More distinct goals
  per tag value, so the tag heads learn what *high distraction* or
  *restricted* means instead of memorising 7 intents.
- **Minimal pairs** (hard negatives): two utterances that share most of
  their words but belong to two existing intents whose tags differ so that
  the route changes ("open the rear window" / "open the rear door").

Every row from these files lands in **train**. Val, test and OOD are never
touched, and the generator guarantees that the rest of the dataset stays
byte-identical.

## 2. How the rows are made and labelled

`python -m autogate_bench.generate ... --augment data/augment` appends, after
all other rows:

- per goal utterance and per pair side: the same slot filling as a seed
  (sensitive fills, relation-word fills for `{contact}`), 3 contexts sampled
  toward the route boundary, **no ASR copies**;
- a goal's route is `rulebook.route_tags(capability, distraction, actuation,
  context, has_sensitive_spans)` from the tags you give it. Its `intent`
  column is the goal name and `is_synthetic` is true; the router's intent
  head ignores these rows, the tag and route heads learn from them;
- a pair side is an ordinary row of its intent, labelled exactly like every
  other row of that intent, with `pair_id` set.

So **the tags you write are the labels**. A wrong tag teaches the router a
wrong rule. When unsure, write the rationale first and pick the tags from
it, using the rules in `docs/taxonomy.md`.

## 3. `data/augment/synthetic_goals.yaml`

A list of goals:

```yaml
- name: start_karaoke_session        # snake_case, unique, not an existing intent name
  description: Start a karaoke session with the lyrics on the screen
  group: media                        # vehicle_control, navigation, communication,
                                      # media, information, planning, productivity
  capability: local_ok                # local_ok, needs_small_local, needs_cloud, cloud_preferred
  distraction: high                   # low, medium, high
  actuation: none                     # none, safety_critical, comfort, restricted
  consequence: reversible             # none, reversible, irreversible
  rationale: on-screen lyrics are sustained visual content (UX_RESTRICTIONS_NO_VIDEO)
  utterances:
    - {text: "start karaoke with the lyrics on the screen", style: command, variant: plain}
    - {text: "karaoke chalu karo lyrics screen pe dikhao", style: command, variant: hinglish}
    - ...
```

Each goal:

- **A real goal a driver or passenger would ask a car assistant for**, not
  a rewording of an existing intent (`docs/taxonomy.md` lists all 81). If
  the nearest existing intent has the same goal, drop yours.
- **Tags follow the taxonomy's rules.** `actuation` other than `none` only
  for `group: vehicle_control`, and then `capability: local_ok`,
  `distraction: low` and a `consequence` other than `none` (the command is
  short; it is refused, if at all, by its actuation class).
  `vehicle_control` goals always have an actuation class. Every other
  goal has `actuation: none` and is refused, if at all, by its distraction.
  `high` distraction means the response needs the screen or sustained
  interaction (the set that Android Automotive's UX restrictions block while
  moving). `cloud_preferred` means onboard data can answer now, less well;
  `needs_cloud` means it cannot.
- **`rationale`**: one line naming the AAOS restriction, vehicle property or
  property of the task that justifies the tags.
- **8 to 12 utterances**, at least **3 styles** (command, outcome,
  question, indirect, fragment; same meanings as in the paraphrase spec), at
  least **one hinglish or kannada_english** line. Aim for about 6 plain,
  2 code-mixed and 1 to 2 disfluent.
- **Slots** only from the existing slot set (`{contact}`, `{phone}`,
  `{address}`, `{otp}`, `{card}`, `{place}`, `{city}`, `{temperature}`,
  `{fan_level}`, `{song}`, `{artist}`, `{station}`, `{podcast}`, `{time}`,
  `{date}`, `{duration}`, `{team}`, `{topic}`, `{note_text}`,
  `{message_text}`, `{event_title}`, `{drive_mode}`, `{color}`, `{speed}`,
  `{seat}`, `{window}`, `{number}`). Slots may differ between the
  utterances of one goal. A sensitive slot makes the row sensitive (it can
  route CLOUD_MASKED), so use one only where the goal really carries one.

### What to generate first

Run `python -m autogate_bench.augment stats` to see goals per tag value
beside the number of training intents with that value. Spend goals where the
training intents are few and R4 failed:

| priority | tag value | training intents | why |
| --- | --- | --- | --- |
| 1 | distraction `high` | 7 | OOD high-distraction rows called low/medium 63 to 74% of the time |
| 2 | actuation `restricted` (with question-style lines) | 8 | adversarial slice, 25 to 30% critical misses |
| 3 | capability `cloud_preferred`, `needs_small_local`, `needs_cloud` | 7, 10, 10 | capability 0.50 on OOD |
| 4 | distraction `medium`, consequence `irreversible`, actuation `safety_critical` | 13, 9, 7 | rare values |

Vary the other tags inside each priority: a `high` goal can be
`local_ok` or `needs_cloud`, `reversible` or `irreversible`, in any non-vehicle
group. Use many groups.

## 4. `data/augment/minimal_pairs.yaml`

A list of pairs:

```yaml
- id: rear_window_vs_rear_door_01    # snake_case, unique
  differs_in: [actuation]            # exactly the tags that differ between the two intents
  a: {intent: adjust_window, text: "open the rear left window for my friend", style: command, variant: plain}
  b: {intent: unlock_doors, text: "open the rear left door for my friend", style: command, variant: plain}
  note: a window is comfort and always runs; a door is restricted and is refused while moving
```

Each pair:

- **Two different existing intents, neither of them OOD.** `differs_in`
  lists exactly the tags (capability, distraction, actuation, consequence)
  in which the two intents differ; the validator computes it and rejects a
  mismatch.
- **The texts look alike**: token Jaccard of at least 0.4 under the rules
  baseline's tokenizer (lowercased, slots and stopwords dropped). Change as
  few words as possible; the changed words should be the ones that carry the
  difference.
- **The route must differ in some context.** Two intents whose tags differ
  only in ways the rulebook ignores are not a hard negative for routing and
  are rejected. Examples that are rejected: `set_cruise_speed` vs
  `cancel_cruise_control` (comfort vs safety_critical: both always LOCAL),
  `unlock_doors` vs `disable_child_locks` (same tags), anything that differs
  only in `consequence`.
- **Each side is an ordinary utterance of its intent**: it must read as
  that intent's goal (check `docs/taxonomy.md`), in its own right, with the
  same voice rules as paraphrases.
- **Style.** Use the same style on both sides unless the style is the point.
  Many pairs should put a restricted action in a question
  ("can you open the rear door for my friend") opposite a comfort or
  low-distraction look-alike: that is the adversarial slice.

Good targets: comfort vs restricted vehicle controls (window vs door, seat
vs steering wheel, window vs mirrors), low vs high distraction (voice call
vs video call, music vs music video, save a note vs edit a note), medium vs
high (hear today's calendar vs open a document), capability
(`current_time` vs `weather_now`, `fuel_range` vs `fuel_prices_nearby`),
medium vs low (hear messages vs send one).

## 5. Hard rules (validator errors)

All of the paraphrase spec's text rules: lowercase, ASCII, no punctuation
except apostrophes and one trailing `?`, at most 24 words, no literal names,
numbers (4+ digits) or values where a slot belongs, no received-message
content. Plus:

1. **Unique**: no utterance identical (after lowercasing and dropping
   apostrophes, spacing and a trailing `?`) to any seed, any paraphrase, or
   any other augmentation utterance.
2. **No OOD goals.** A goal utterance may not have token Jaccard above 0.5
   with any seed or paraphrase of an OOD intent, and a goal's name and
   description may not contain an OOD intent's key words. The list, matched
   as word prefixes (all words of one entry must appear):

   | OOD intent | key words |
   | --- | --- |
   | `set_ambient_lighting` | ambient; mood + light |
   | `fog_lights_on` | fog |
   | `open_fuel_door` | fuel + door / flap / lid / cap; petrol + flap / lid; tank + flap / lid; filler |
   | `route_avoiding_tolls` | toll |
   | `compose_long_email` | long / compose / draft / write + email |
   | `read_news_headlines` | news; headline |
   | `estimate_trip_cost` | trip / journey / drive + cost |
   | `pair_bluetooth_device` | bluetooth; pair |

   Do not write goals in these families at all, even under other words:
   the OOD split measures whether the router generalises to them.
3. **No val/test leak.** No augmentation utterance (goal or pair) may have
   token Jaccard above 0.7 with any seed or paraphrase of a val or test
   seed. Do not copy or lightly edit existing paraphrases.
4. **Goals per section 3, pairs per section 4.** Unknown group, tag, style
   or variant values are errors.
5. With `--strict`: 8 to 12 utterances, 3 styles, one code-mixed line per
   goal.

Warnings point at what to reread: two near-identical utterances in one goal;
a goal utterance that looks like an existing intent with different routing
tags (is the tag right, or is it that intent?); a pair side that looks more
like another intent than its own.

## 6. Checklist before committing

```
uv run python -m autogate_bench.augment validate --strict          # must print 0 errors
uv run python -m autogate_bench.augment stats                      # coverage per tag value
uv run python -m autogate_bench.augment review --out docs/augment-review.md
```

1. Fix every error. Read every warning.
2. Open `docs/augment-review.md`. Items are sorted by risk, highest first.
   For the top 20, check: is the nearest existing intent really a different
   goal? If its tags differ from yours, which are right? Is the OOD or
   val/test similarity just shared function words?
3. Do not reorder or delete committed goals, utterances or pairs (row ids
   use positions); fix them in place, like paraphrase lines.
4. Commit `data/augment/*.yaml` and `docs/augment-review.md`. Never commit
   generated rows.
