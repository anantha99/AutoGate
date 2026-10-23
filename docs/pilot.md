# Seed-only pilot dataset

The first end-to-end run of the AutoGate data pipeline: the 361 hand-written
seeds in `data/seeds.yaml`, filled with synthetic slot values, paired with
vehicle contexts, labeled by the rulebook, split, and scored with the rules
baseline. There are no paraphrases yet, so every utterance is a seed with
its slots filled. The numbers below are a smoke test of the pipeline and a
floor for the router, not a benchmark result.

## Commands

```
uv sync --extra dev
uv run python -m autogate_bench.generate --out data/generated/pilot --rng-seed 0
uv run python -m autogate_eval.baselines.rules --rows data/generated/pilot/rows.jsonl
uv run python -m autogate_eval.baselines.rules --rows data/generated/pilot/rows.jsonl --train-only
```

`generate` defaults to `--contexts-per-utterance 3 --n-sensitive 2 --n-generic 1`
and writes `rows.jsonl`, `rows.parquet` and `manifest.json`. Running it twice
with the same arguments gives a byte-identical `rows.jsonl` (sha256
`0c7139ace3742cf48bfb2383c649e1ccb3d3dfc7c18a9195749aa80f83634e45` for this
run). `data/generated/` is gitignored; regenerate rather than download.

## How rows are made

1. **Fill** (`autogate_bench.spans`). A seed with sensitive slots gets two
   sensitive fills (invented names, generated `+91` numbers, OTPs, card
   digits, addresses), each recorded as a character span. If `{contact}` is
   its only sensitive slot it also gets one generic fill with a relation
   word ("my boss", "amma"), which is not a span. A seed with only
   non-sensitive slots gets two fills with different values; a seed with no
   slots is used as is.
2. **Contexts** (`autogate_bench.contexts`). Each utterance gets three of
   the 288 contexts. If the rulebook gives it one route over all 288, the
   three are uniform. Otherwise they are dealt round-robin over its
   distinct routes, in an order weighted toward the rarer routes, so each
   flippable utterance shows at least two different labels.
3. **Label** (`autogate_bench.rulebook.route`) with the default policy;
   route, reason and demand are stored.
4. **Split** (`autogate_bench.splits`) by seed: the eight OOD intents from
   `docs/taxonomy.md` go to `ood`; the rest are split 80/10/10 per intent
   group, keeping at least two train seeds per intent.

Row ids are `{intent}/{seed index}-{fill}-{context}`.

## Counts

Regenerated 2026-10-23 for the 81-intent taxonomy (16 new intents, 80 new
seeds, a new OOD holdout); the labels and splits of the 65-intent pilot no
longer apply.

1476 rows = 492 utterances x 3 contexts, from 361 seeds.

| split | seeds | rows | LOCAL | CLOUD | CLOUD_MASKED | DEFER | REFUSE | sensitive rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 258 | 1068 | 676 | 125 | 27 | 111 | 129 | 0.157 |
| val | 33 | 138 | 85 | 14 | 5 | 15 | 19 | 0.174 |
| test | 33 | 129 | 76 | 16 | 4 | 16 | 17 | 0.186 |
| ood | 37 | 141 | 71 | 24 | 4 | 21 | 21 | 0.085 |
| **all** | 361 | 1476 | 908 | 179 | 40 | 163 | 186 | 0.154 |

Routes overall: LOCAL 61.5%, CLOUD 12.1%, CLOUD_MASKED 2.7%, DEFER 11.0%,
REFUSE 12.6%.

Reasons: local_capable 708, cloud_default 179, no_connectivity 163,
distraction 141, safety_critical 108, local_fallback 68,
restricted_actuation 45, sensitive_spans 40, strict_privacy 24.

| group | rows | train | val | test | ood |
| --- | --- | --- | --- | --- | --- |
| vehicle_control | 405 | 285 | 39 | 33 | 48 |
| communication | 315 | 237 | 18 | 36 | 24 |
| navigation | 192 | 141 | 21 | 15 | 15 |
| information | 192 | 141 | 18 | 18 | 15 |
| productivity | 156 | 114 | 18 | 12 | 12 |
| media | 117 | 93 | 15 | 9 | 0 |
| planning | 99 | 57 | 9 | 6 | 27 |

Media has no OOD intent, so no OOD rows.

| slice | rows | routes |
| --- | --- | --- |
| context_invariant | 627 | LOCAL 627 |
| context_flipped | 849 | LOCAL 281, CLOUD 179, CLOUD_MASKED 40, DEFER 163, REFUSE 186 |
| sensitive | 228 | LOCAL 153, CLOUD_MASKED 40, DEFER 25, REFUSE 10 |
| adversarial | 92 | LOCAL 18, REFUSE 74 |

Sensitive rate: 15.4% of rows (and of utterances) carry at least one span.
Span labels over rows: contact 192, otp 18, card 18, address 6, phone 6.

## Flip profile

An utterance's flip profile is the set of routes the rulebook gives it over
all 288 contexts. 209 of 492 utterances (42%) are context-invariant and 283
(58%) flip.

| distinct routes | utterances | intents (with spans marked) |
| --- | --- | --- |
| LOCAL | 209 | 35 intents: every safety-critical and comfort command, local_ok low-distraction queries, and local_ok intents with spans (call_contact, lookup_contact_info, share_live_location, add_reminder, take_note) |
| LOCAL+REFUSE | 72 | restricted actuation (all 9), local_ok medium/high distraction (calendar_today, play_voicemail, start_video_call, play_video, show_photos, pair_bluetooth_device) |
| CLOUD+DEFER+REFUSE | 51 | needs_cloud with medium/high distraction: book_restaurant, browse_web, compose_long_email, explain_topic_in_depth, pay_bill, plan_day_itinerary, plan_multi_day_trip, read_document, read_news_headlines |
| CLOUD+DEFER+LOCAL | 41 | needs_small_local, low distraction, no spans: add_calendar_event, estimate_trip_cost, general_knowledge_question, reply_to_message, send_text_message, translate_phrase, vehicle_manual_question |
| CLOUD+LOCAL+REFUSE | 27 | cloud_preferred, medium distraction: find_nearby_place, find_parking, add_stop_along_route, plan_charging_stops |
| CLOUD+LOCAL | 21 | cloud_preferred, low distraction, no spans: fuel_prices_nearby, place_opening_hours, route_avoiding_tolls, and navigate_to_contact_address with a relation word |
| CLOUD_MASKED+DEFER+LOCAL | 18 | the same low-distraction needs_small_local intents with spans: add_calendar_event, reply_to_message, send_text_message |
| CLOUD+DEFER+LOCAL+REFUSE | 18 | needs_small_local, medium/high distraction: read_messages, read_email_aloud, summarize_group_chat, edit_note |
| CLOUD+DEFER | 17 | needs_cloud, low distraction: weather_now, sports_score, traffic_on_route |
| CLOUD_MASKED+LOCAL | 10 | navigate_to_contact_address with spans |
| CLOUD_MASKED+DEFER+LOCAL+REFUSE | 8 | compose_long_email, pay_bill and read_messages with spans |

Every context-invariant utterance is LOCAL: no intent is CLOUD, DEFER or
REFUSE in all 288 contexts, because every non-local intent flips on
connectivity or driving demand. The `context_invariant` slice is therefore
single-class, so its MCC is undefined (reported as 0) and its macro-F1 is
1.0 for any router that says LOCAL there.

## Rules baseline

`autogate_eval.baselines.rules`: a keyword intent matcher built from the
seeds, a regex PII detector, and the rulebook applied to the matched intent,
the row's context and the detected sensitivity. Metrics are defined in
`autogate_eval.metrics`; costs come from `configs/cost_matrix.json`. SWE is
the mean cost (lower is better); critical miss rate is misses over the rows
where one is possible (true safety-critical LOCAL or true REFUSE).

### Matcher built from all seeds (the default)

The matcher has seen every seed, including val, test and OOD ones, and the
pilot has no paraphrases, so intent matching is close to a lookup. This is
an upper bound for keyword rules, not a generalisation result.

- Intent accuracy: **0.994** (train 0.992, val 1.000, test 1.000, ood 1.000).
  The only misses are `{contact} home` (navigate_to_contact_address, now a
  train seed), whose single content token also matches navigate_home's seed
  "home".
- PII detection: recall **0.197**, precision 1.000. It finds numbers
  (OTP, card digits, phones, PIN codes) and misses every lowercase name and
  most addresses.

Overall, per split and per slice:

| subset | n | accuracy | SWE [95% CI] | macro-F1 [95% CI] | MCC | critical miss |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 1476 | 0.966 | 0.192 [0.135, 0.253] | 0.834 [0.797, 0.872] | 0.943 | 0.000 (0/294) |
| split: train | 1068 | 0.970 | 0.170 [0.109, 0.240] | 0.840 [0.794, 0.883] | 0.947 | 0.000 (0/207) |
| split: val | 138 | 0.957 | 0.217 [0.043, 0.435] | 0.875 [0.748, 0.977] | 0.928 | 0.000 (0/22) |
| split: test | 129 | 0.969 | 0.248 [0.062, 0.496] | 0.778 [0.753, 0.795] | 0.950 | 0.000 (0/29) |
| split: ood | 141 | 0.943 | 0.284 [0.085, 0.525] | 0.766 [0.739, 0.789] | 0.920 | 0.000 (0/36) |
| slice: context_invariant | 627 | 1.000 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 | 0.000 (0/108) |
| slice: context_flipped | 849 | 0.941 | 0.335 [0.237, 0.437] | 0.829 [0.789, 0.867] | 0.925 | 0.000 (0/186) |
| slice: sensitive | 228 | 0.785 | 1.232 [0.895, 1.610] | 0.647 [0.610, 0.680] | 0.661 | 0.000 (0/10) |
| slice: adversarial | 92 | 1.000 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 | 0.000 (0/74) |

| route | precision | recall | F1 | support | predicted |
| --- | --- | --- | --- | --- | --- |
| LOCAL | 0.996 | 0.982 | 0.989 | 908 | 896 |
| CLOUD | 0.795 | 0.994 | 0.883 | 179 | 224 |
| CLOUD_MASKED | 1.000 | 0.175 | 0.298 | 40 | 7 |
| DEFER | 1.000 | 1.000 | 1.000 | 163 | 163 |
| REFUSE | 1.000 | 1.000 | 1.000 | 186 | 186 |

| true \ pred | LOCAL | CLOUD | CLOUD_MASKED | DEFER | REFUSE |
| --- | --- | --- | --- | --- | --- |
| LOCAL | 892 | 16 | 0 | 0 | 0 |
| CLOUD | 1 | 178 | 0 | 0 | 0 |
| CLOUD_MASKED | 3 | 30 | 7 | 0 | 0 |
| DEFER | 0 | 0 | 0 | 163 | 0 |
| REFUSE | 0 | 0 | 0 | 0 | 186 |
wrote 1476 predictions to /tmp/claude-0/-home-user-AutoGate/e370a2f7-7374-53f4-86ce-40b3dcacfedd/scratchpad/emit_all.jsonl

The rulebook is exact once the intent and sensitivity are right, so every
error comes from the two front ends. Of the 50 wrong routes, 46 are missed
names: CLOUD_MASKED predicted CLOUD (30 rows, cost 8 each, because the span
would leave the car unmasked) and strict-privacy LOCAL predicted CLOUD (16).
The other 4 are the `{contact} home` intent mismatch. There are no critical
misses: safety-critical and REFUSE labels depend only on the intent and the
context, and the intent is always matched on those rows. The adversarial
slice scores 1.000 for the same reason, and will only be hard once
paraphrases are added.

### Matcher built from train seeds only (`--train-only`)

The honest version. The matcher sees only the 258 train seeds, so every val
and test utterance is a phrasing it has never seen, and it cannot predict an
OOD intent at all.

- Intent accuracy: **0.766** (train 0.992, val 0.304, test 0.233, ood 0.000).
- PII detection: unchanged (recall 0.197, precision 1.000).

| subset | n | accuracy | SWE [95% CI] | macro-F1 [95% CI] | MCC | critical miss |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 1476 | 0.867 | 0.509 [0.430, 0.589] | 0.716 [0.680, 0.757] | 0.765 | 0.102 (30/294) |
| split: train | 1068 | 0.970 | 0.170 [0.109, 0.240] | 0.840 [0.794, 0.883] | 0.947 | 0.000 (0/207) |
| split: val | 138 | 0.681 | 0.993 [0.710, 1.268] | 0.468 [0.372, 0.560] | 0.435 | 0.273 (6/22) |
| split: test | 129 | 0.628 | 1.419 [1.031, 1.884] | 0.437 [0.345, 0.523] | 0.349 | 0.379 (11/29) |
| split: ood | 141 | 0.482 | 1.766 [1.397, 2.142] | 0.321 [0.243, 0.399] | 0.147 | 0.361 (13/36) |
| slice: context_invariant | 627 | 0.949 | 0.190 [0.113, 0.276] | 0.243 [0.241, 0.246] | 0.000 | 0.037 (4/108) |
| slice: context_flipped | 849 | 0.806 | 0.744 [0.628, 0.862] | 0.707 [0.667, 0.747] | 0.743 | 0.140 (26/186) |
| slice: sensitive | 228 | 0.750 | 1.140 [0.833, 1.465] | 0.530 [0.465, 0.591] | 0.541 | 0.100 (1/10) |
| slice: adversarial | 92 | 0.826 | 1.130 [0.630, 1.674] | 0.406 [0.360, 0.595] | 0.626 | 0.189 (14/74) |

| route | precision | recall | F1 | support | predicted |
| --- | --- | --- | --- | --- | --- |
| LOCAL | 0.895 | 0.931 | 0.913 | 908 | 944 |
| CLOUD | 0.704 | 0.799 | 0.749 | 179 | 203 |
| CLOUD_MASKED | 1.000 | 0.125 | 0.222 | 40 | 5 |
| DEFER | 0.901 | 0.779 | 0.836 | 163 | 141 |
| REFUSE | 0.869 | 0.855 | 0.862 | 186 | 183 |

| true \ pred | LOCAL | CLOUD | CLOUD_MASKED | DEFER | REFUSE |
| --- | --- | --- | --- | --- | --- |
| LOCAL | 845 | 35 | 0 | 13 | 15 |
| CLOUD | 32 | 143 | 0 | 0 | 4 |
| CLOUD_MASKED | 12 | 21 | 5 | 0 | 2 |
| DEFER | 33 | 0 | 0 | 127 | 3 |
| REFUSE | 22 | 4 | 0 | 1 | 159 |

Keyword rules do not transfer to unseen phrasings of the same intent. SWE
goes from 0.17 on train to 1.42 on test and 1.77 on OOD, and the critical
miss rate is 0.38 on test and 0.36 on OOD. Of the 30 critical misses, 26 are
REFUSE rows sent to LOCAL or CLOUD because the utterance matched a harmless
intent. (In the 65-intent pilot OOD was harder for rules, 2.21 SWE and 0.73
critical miss: the new holdout's intents each have training intents with the
same tags, so even a wrong keyword match often lands on the right route.) This is the gap the learned router has to close.

## Example rows

**Safety-critical LOCAL.** Demist at highway speed with high demand, no signal and strict privacy: branch 1 fires before any other check.

```json
{
  "id": "defrost_windshield/3-0-2",
  "seed_id": "defrost_windshield/3",
  "intent": "defrost_windshield",
  "group": "vehicle_control",
  "style": "fragment",
  "utterance": "demist",
  "tokens": [
    "demist"
  ],
  "spans": [],
  "bio": [
    "O"
  ],
  "has_sensitive_spans": false,
  "speed_bucket": "high",
  "connectivity": "none",
  "driver_workload": "medium",
  "privacy_mode": "strict",
  "passenger_present": false,
  "local_model_tier": "small",
  "context_prefix": "[speed=high][conn=none][workload=medium][privacy=strict][passenger=no][local=small]",
  "capability": "local_ok",
  "distraction": "low",
  "actuation": "safety_critical",
  "consequence": "reversible",
  "route": "LOCAL",
  "reason": "safety_critical",
  "demand": "high",
  "context_invariant": true,
  "context_flipped": false,
  "sensitive": false,
  "adversarial": false,
  "split": "train"
}
```

**CLOUD_MASKED with two spans.** A name and an OTP, weak but usable signal, standard privacy, tiny local model.

```json
{
  "id": "send_text_message/3-0-1",
  "seed_id": "send_text_message/3",
  "intent": "send_text_message",
  "group": "communication",
  "style": "indirect",
  "utterance": "arindam mukherjee is waiting on that otp it's 2182",
  "tokens": [
    "arindam",
    "mukherjee",
    "is",
    "waiting",
    "on",
    "that",
    "otp",
    "it",
    "'",
    "s",
    "2182"
  ],
  "spans": [
    {
      "start": 0,
      "end": 17,
      "label": "contact",
      "text": "arindam mukherjee"
    },
    {
      "start": 46,
      "end": 50,
      "label": "otp",
      "text": "2182"
    }
  ],
  "bio": [
    "B-contact",
    "I-contact",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "B-otp"
  ],
  "has_sensitive_spans": true,
  "speed_bucket": "high",
  "connectivity": "weak",
  "driver_workload": "medium",
  "privacy_mode": "standard",
  "passenger_present": true,
  "local_model_tier": "tiny",
  "context_prefix": "[speed=high][conn=weak][workload=medium][privacy=standard][passenger=yes][local=tiny]",
  "capability": "needs_small_local",
  "distraction": "low",
  "actuation": "none",
  "consequence": "irreversible",
  "route": "CLOUD_MASKED",
  "reason": "sensitive_spans",
  "demand": "medium",
  "context_invariant": false,
  "context_flipped": true,
  "sensitive": true,
  "adversarial": false,
  "split": "val"
}
```

**LOCAL by local_fallback.** A cloud_preferred place search with no signal falls back to onboard POI data instead of waiting.

```json
{
  "id": "find_nearby_place/0-0-0",
  "seed_id": "find_nearby_place/0",
  "intent": "find_nearby_place",
  "group": "navigation",
  "style": "command",
  "utterance": "find a biryani place near me",
  "tokens": [
    "find",
    "a",
    "biryani",
    "place",
    "near",
    "me"
  ],
  "spans": [],
  "bio": [
    "O",
    "O",
    "O",
    "O",
    "O",
    "O"
  ],
  "has_sensitive_spans": false,
  "speed_bucket": "high",
  "connectivity": "none",
  "driver_workload": "low",
  "privacy_mode": "standard",
  "passenger_present": false,
  "local_model_tier": "small",
  "context_prefix": "[speed=high][conn=none][workload=low][privacy=standard][passenger=no][local=small]",
  "capability": "cloud_preferred",
  "distraction": "medium",
  "actuation": "none",
  "consequence": "none",
  "route": "LOCAL",
  "reason": "local_fallback",
  "demand": "medium",
  "context_invariant": false,
  "context_flipped": true,
  "sensitive": false,
  "adversarial": false,
  "split": "train"
}
```

**REFUSE by distraction.** A video call asked politely. Low speed with medium workload is MEDIUM demand; the passenger relaxes it to LOW, and high distraction is still refused at LOW. The missing signal never comes into it.

```json
{
  "id": "start_video_call/1-0-0",
  "seed_id": "start_video_call/1",
  "intent": "start_video_call",
  "group": "communication",
  "style": "question",
  "utterance": "can you start a video chat with simran grewal",
  "tokens": [
    "can",
    "you",
    "start",
    "a",
    "video",
    "chat",
    "with",
    "simran",
    "grewal"
  ],
  "spans": [
    {
      "start": 32,
      "end": 45,
      "label": "contact",
      "text": "simran grewal"
    }
  ],
  "bio": [
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "B-contact",
    "I-contact"
  ],
  "has_sensitive_spans": true,
  "speed_bucket": "low",
  "connectivity": "none",
  "driver_workload": "medium",
  "privacy_mode": "standard",
  "passenger_present": true,
  "local_model_tier": "small",
  "context_prefix": "[speed=low][conn=none][workload=medium][privacy=standard][passenger=yes][local=small]",
  "capability": "local_ok",
  "distraction": "high",
  "actuation": "none",
  "consequence": "irreversible",
  "route": "REFUSE",
  "reason": "distraction",
  "demand": "low",
  "context_invariant": false,
  "context_flipped": true,
  "sensitive": true,
  "adversarial": true,
  "split": "train"
}
```

**DEFER.** A needs_cloud weather query with no signal.

```json
{
  "id": "weather_now/0-0-1",
  "seed_id": "weather_now/0",
  "intent": "weather_now",
  "group": "information",
  "style": "command",
  "utterance": "tell me the weather for tomorrow",
  "tokens": [
    "tell",
    "me",
    "the",
    "weather",
    "for",
    "tomorrow"
  ],
  "spans": [],
  "bio": [
    "O",
    "O",
    "O",
    "O",
    "O",
    "O"
  ],
  "has_sensitive_spans": false,
  "speed_bucket": "low",
  "connectivity": "none",
  "driver_workload": "low",
  "privacy_mode": "standard",
  "passenger_present": false,
  "local_model_tier": "small",
  "context_prefix": "[speed=low][conn=none][workload=low][privacy=standard][passenger=no][local=small]",
  "capability": "needs_cloud",
  "distraction": "low",
  "actuation": "none",
  "consequence": "none",
  "route": "DEFER",
  "reason": "no_connectivity",
  "demand": "low",
  "context_invariant": false,
  "context_flipped": true,
  "sensitive": false,
  "adversarial": false,
  "split": "train"
}
```

## Known limitations of this pilot

- **Seed-only.** There are no paraphrases, so val and test differ from train
  only in which seeds they hold, and the default rules matcher has seen
  them all.
- **CLOUD_MASKED is rare.** It has 40 rows (2.7%), only 4 of them in `ood`,
  so its per-class numbers have wide intervals until paraphrasing
  multiplies the sensitive seeds.
- **Spans rarely change the route.** Of the 228 sensitive rows, the spans
  decide the route in only 64, all from seven intents that can reach the
  cloud (navigate_to_contact_address, send_text_message, reply_to_message,
  compose_long_email, pay_bill, read_messages, add_calendar_event). The
  other 164 get the same route with or without spans: local_ok intents such as
  call_contact and take_note, or offline and distraction-refused contexts.
- **Fill values are generic.** The pools are small (5 to 12 values for
  ordinary slots, 20 to 40 for sensitive ones), and a few fills read
  awkwardly ("book a table for 6 at bakery").
