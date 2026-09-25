# Seed-only pilot dataset

The first end-to-end run of the AutoGate data pipeline: the 281 hand-written
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
`d055b4c97072ac0f62415544d72cf2282f2b2d890510c885420862799e8b14a9` for this
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
4. **Split** (`autogate_bench.splits`) by seed: the six OOD intents from
   `docs/taxonomy.md` go to `ood`; the rest are split 80/10/10 per intent
   group, keeping at least two train seeds per intent.

Row ids are `{intent}/{seed index}-{fill}-{context}`.

## Counts

1182 rows = 394 utterances x 3 contexts, from 281 seeds.

| split | seeds | rows | LOCAL | CLOUD | CLOUD_MASKED | DEFER | REFUSE | sensitive rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 204 | 855 | 565 | 77 | 21 | 87 | 105 | 0.161 |
| val | 26 | 114 | 68 | 14 | 8 | 17 | 7 | 0.263 |
| test | 26 | 108 | 66 | 9 | 7 | 12 | 14 | 0.167 |
| ood | 25 | 105 | 38 | 26 | 2 | 17 | 22 | 0.057 |
| **all** | 281 | 1182 | 737 | 126 | 38 | 133 | 148 | 0.162 |

Routes overall: LOCAL 62.4%, CLOUD 10.7%, CLOUD_MASKED 3.2%, DEFER 11.3%,
REFUSE 12.5%.

Reasons: local_capable 619, no_connectivity 133, cloud_default 126,
distraction 121, safety_critical 63, sensitive_spans 38, local_fallback 33,
restricted_actuation 27, strict_privacy 22.

| group | rows | train | val | test | ood |
| --- | --- | --- | --- | --- | --- |
| vehicle_control | 315 | 240 | 27 | 36 | 12 |
| communication | 255 | 192 | 27 | 24 | 12 |
| navigation | 159 | 102 | 21 | 15 | 21 |
| productivity | 138 | 90 | 12 | 9 | 27 |
| information | 126 | 90 | 15 | 9 | 12 |
| media | 117 | 99 | 9 | 9 | 0 |
| planning | 72 | 42 | 3 | 6 | 21 |

Media has no OOD intent, so no OOD rows.

| slice | rows | routes |
| --- | --- | --- |
| context_invariant | 537 | LOCAL 537 |
| context_flipped | 645 | LOCAL 200, CLOUD 126, CLOUD_MASKED 38, DEFER 133, REFUSE 148 |
| sensitive | 192 | LOCAL 121, CLOUD_MASKED 38, DEFER 23, REFUSE 10 |
| adversarial | 72 | LOCAL 12, REFUSE 60 |

Sensitive rate: 16.2% of rows (and of utterances) carry at least one span.
Span labels over rows: contact 162, otp 18, card 12, address 6, phone 6.

## Flip profile

An utterance's flip profile is the set of routes the rulebook gives it over
all 288 contexts. 179 of 394 utterances (45%) are context-invariant and 215
(55%) flip.

| distinct routes | utterances | intents (with spans marked) |
| --- | --- | --- |
| LOCAL | 179 | 35 intent variants: every safety-critical and comfort command, local_ok low-distraction queries, and local_ok intents with spans (call_contact, lookup_contact_info, add_reminder, take_note) |
| LOCAL+REFUSE | 57 | restricted actuation (5 + disable_child_locks), local_ok medium/high distraction (calendar_today, play_voicemail, start_video_call, play_video, show_photos, pair_bluetooth_device) |
| CLOUD+DEFER+REFUSE | 42 | needs_cloud with medium/high distraction: book_restaurant, browse_web, compose_long_email, explain_topic_in_depth, plan_day_itinerary, plan_multi_day_trip, read_document |
| CLOUD+DEFER+LOCAL | 27 | needs_small_local, low distraction, no spans: add_calendar_event, general_knowledge_question, reply_to_message, send_text_message, vehicle_manual_question |
| CLOUD+LOCAL+REFUSE | 21 | cloud_preferred, medium distraction: find_nearby_place, add_stop_along_route, plan_charging_stops |
| CLOUD_MASKED+DEFER+LOCAL | 18 | the same needs_small_local intents with spans |
| CLOUD+DEFER | 17 | needs_cloud, low distraction: weather_now, sports_score, traffic_on_route |
| CLOUD+DEFER+LOCAL+REFUSE | 13 | needs_small_local, medium/high distraction: read_messages, summarize_group_chat, edit_note |
| CLOUD_MASKED+LOCAL | 10 | navigate_to_contact_address with spans |
| CLOUD_MASKED+DEFER+LOCAL+REFUSE | 6 | compose_long_email and read_messages with spans |
| CLOUD+LOCAL | 4 | navigate_to_contact_address with a relation word |

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

- Intent accuracy: **0.992** (train 1.000, val 0.921, test 1.000, ood 1.000).
  The only misses are `{contact} home` (navigate_to_contact_address),
  whose single content token also matches navigate_home's seed "home".
- PII detection: recall **0.203**, precision 1.000. It finds numbers
  (OTP, card digits, phones, PIN codes) and misses every lowercase name and
  most addresses.

Overall, per split and per slice:

| subset | n | accuracy | SWE [95% CI] | macro-F1 [95% CI] | MCC | critical miss |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 1182 | 0.958 | 0.240 [0.168, 0.319] | 0.812 [0.774, 0.851] | 0.928 | 0.000 (0/211) |
| split: train | 855 | 0.965 | 0.204 [0.131, 0.290] | 0.800 [0.760, 0.852] | 0.936 | 0.000 (0/153) |
| split: val | 114 | 0.886 | 0.526 [0.254, 0.860] | 0.733 [0.692, 0.764] | 0.815 | 0.000 (0/16) |
| split: test | 108 | 0.954 | 0.315 [0.056, 0.611] | 0.875 [0.753, 0.974] | 0.924 | 0.000 (0/20) |
| split: ood | 105 | 0.981 | 0.152 [0.000, 0.381] | 0.793 [0.781, 1.000] | 0.975 | 0.000 (0/22) |
| slice: context_invariant | 537 | 1.000 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 | 0.000 (0/63) |
| slice: context_flipped | 645 | 0.922 | 0.440 [0.313, 0.577] | 0.805 [0.767, 0.844] | 0.903 | 0.000 (0/148) |
| slice: sensitive | 192 | 0.745 | 1.464 [1.094, 1.860] | 0.630 [0.596, 0.665] | 0.638 | 0.000 (0/10) |
| slice: adversarial | 72 | 1.000 | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 | 0.000 (0/60) |

| route | precision | recall | F1 | support | predicted |
| --- | --- | --- | --- | --- | --- |
| LOCAL | 0.994 | 0.978 | 0.986 | 737 | 725 |
| CLOUD | 0.731 | 0.992 | 0.842 | 126 | 171 |
| CLOUD_MASKED | 1.000 | 0.132 | 0.233 | 38 | 5 |
| DEFER | 1.000 | 1.000 | 1.000 | 133 | 133 |
| REFUSE | 1.000 | 1.000 | 1.000 | 148 | 148 |

| true \ pred | LOCAL | CLOUD | CLOUD_MASKED | DEFER | REFUSE |
| --- | --- | --- | --- | --- | --- |
| LOCAL | 721 | 16 | 0 | 0 | 0 |
| CLOUD | 1 | 125 | 0 | 0 | 0 |
| CLOUD_MASKED | 3 | 30 | 5 | 0 | 0 |
| DEFER | 0 | 0 | 0 | 133 | 0 |
| REFUSE | 0 | 0 | 0 | 0 | 148 |

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

The honest version. The matcher sees only the 204 train seeds, so every val
and test utterance is a phrasing it has never seen, and it cannot predict an
OOD intent at all.

- Intent accuracy: **0.766** (train 1.000, val 0.263, test 0.194, ood 0.000).
- PII detection: unchanged (recall 0.203, precision 1.000).

| subset | n | accuracy | SWE [95% CI] | macro-F1 [95% CI] | MCC | critical miss |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 1182 | 0.848 | 0.602 [0.510, 0.705] | 0.669 [0.635, 0.705] | 0.727 | 0.128 (27/211) |
| split: train | 855 | 0.965 | 0.204 [0.131, 0.290] | 0.800 [0.760, 0.852] | 0.936 | 0.000 (0/153) |
| split: val | 114 | 0.623 | 1.500 [1.096, 1.939] | 0.458 [0.364, 0.538] | 0.375 | 0.312 (5/16) |
| split: test | 108 | 0.593 | 1.241 [0.944, 1.556] | 0.377 [0.270, 0.470] | 0.268 | 0.300 (6/20) |
| split: ood | 105 | 0.400 | 2.210 [1.781, 2.638] | 0.275 [0.191, 0.392] | 0.142 | 0.727 (16/22) |
| slice: context_invariant | 537 | 0.950 | 0.209 [0.127, 0.300] | 0.244 [0.241, 0.246] | 0.000 | 0.048 (3/63) |
| slice: context_flipped | 645 | 0.763 | 0.929 [0.778, 1.078] | 0.652 [0.619, 0.691] | 0.690 | 0.162 (24/148) |
| slice: sensitive | 192 | 0.693 | 1.417 [1.088, 1.787] | 0.509 [0.442, 0.563] | 0.517 | 0.000 (0/10) |
| slice: adversarial | 72 | 0.847 | 0.944 [0.472, 1.528] | 0.407 [0.356, 0.867] | 0.654 | 0.167 (10/60) |

| route | precision | recall | F1 | support | predicted |
| --- | --- | --- | --- | --- | --- |
| LOCAL | 0.886 | 0.924 | 0.904 | 737 | 769 |
| CLOUD | 0.676 | 0.730 | 0.702 | 126 | 136 |
| CLOUD_MASKED | 1.000 | 0.053 | 0.100 | 38 | 2 |
| DEFER | 0.889 | 0.782 | 0.832 | 133 | 117 |
| REFUSE | 0.778 | 0.831 | 0.804 | 148 | 158 |

| true \ pred | LOCAL | CLOUD | CLOUD_MASKED | DEFER | REFUSE |
| --- | --- | --- | --- | --- | --- |
| LOCAL | 681 | 20 | 0 | 12 | 24 |
| CLOUD | 31 | 92 | 0 | 0 | 3 |
| CLOUD_MASKED | 11 | 23 | 2 | 0 | 2 |
| DEFER | 23 | 0 | 0 | 104 | 6 |
| REFUSE | 23 | 1 | 0 | 1 | 123 |

Keyword rules do not transfer to unseen phrasings of the same intent. SWE
goes from 0.20 on train to 1.24 on test and 2.21 on OOD, and the critical
miss rate is 0.30 on test and 0.73 on OOD. Of the 27 critical misses, 24 are
REFUSE rows sent to LOCAL or CLOUD because the utterance matched a harmless
intent. This is the gap the learned router has to close.

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
  "split": "test"
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
- **CLOUD_MASKED is rare.** It has 38 rows (3.2%), only 2 of them in `ood`,
  so its per-class numbers have wide intervals until paraphrasing
  multiplies the sensitive seeds.
- **Spans rarely change the route.** Of the 192 sensitive rows, the spans
  decide the route in only 60, all from six intents that can reach the
  cloud (navigate_to_contact_address, send_text_message, reply_to_message,
  compose_long_email, read_messages, add_calendar_event). The other 132
  get the same route with or without spans: local_ok intents such as
  call_contact and take_note, or offline and distraction-refused contexts.
- **Fill values are generic.** The pools are small (5 to 12 values for
  ordinary slots, 20 to 40 for sensitive ones), and a few fills read
  awkwardly ("book a table for 6 at bakery").
