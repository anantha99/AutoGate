# AutoGate experiment log

The running memory of the project: what we are trying, why, what each run
produced, and what is still open. Newest results go at the top of each
section. Every run gets an entry in "Runs", even a failed one.

Last updated: 2026-11-18 (A7 dropped by its stop rule; A8 planned: fine-tune on the A6 data; R5 and A8 wait for the audit). Linked from `CLAUDE.md` so every session reads it.

## How to use this file

- **Before starting work:** read "Current state" and "Open questions".
- **After a run:** add a row to the runs table and a short entry under
  "Run details" with the config, the commit, and what it showed.
- **After a decision:** add a line to the decision log with the date and why.
- Detailed write-ups stay in their own docs (linked below). This file is the
  index and the short version.

## Current state

| item | status |
| --- | --- |
| Taxonomy expansion (A5) | done 10-23: 81 intents, 361 seeds, 14,440 paraphrases, 66,414 rows (sha256 `8cc12a4b...`); 8-intent OOD holdout balanced by tag. R1/R2 are not directly comparable with runs on this data |
| Train-only augmentation (A6) | data done 10-28: 60 synthetic goals (565 utterances; 20 high-distraction) and 290 minimal pairs (580 utterances, 115 intent pairs); strict validation 0 errors. Adds 4,635 train rows (2,067 goal, 2,568 pair); every other row byte-identical (old file is an exact prefix). Review: `docs/augment-review.md` |
| Decision-encoder router (A7) | **Dropped 11-18 by the plan's stop rule** (Gate 2 failed after Phase 4). No configuration passes test and OOD together: frozen R6a passes OOD (1.015 / 21.7%) but not test (0.562 / 9.7%); fully fine-tuned R6c beats R4 on test (0.867 / 0.308 / 4.8%) but fails OOD (1.166 / 30.5%). Code kept in `src/autogate_decide/` |
| Fine-tune on A6 data (A8) | **Phase 1 stopped 11-18 at the proxy validity check** (pre-registered): the proxy reproduces the distraction collapse (-27 pts) but not the critical-miss gap; waiting for the owner. A6 audit done, no corrections. R6c's recipe on the augmented data, plus a distillation and a LoRA variant, selected on val and a proxy holdout of 8 training intents; one final model scored once. Plan: `docs/plan-a8-finetune-a6.md` |
| Router R5 (R4 config + A6 data) | ready to run on Colab; the A6 audit is done (11-18, no corrections) |
| Router R4 (tags, R1 capacity, 81 intents) | done 10-27, with and without context. Best so far: test macro-F1 0.82-0.85, SWE 28-30% of fair rules; critical miss 5-6% (under 1% with the 0.85 abstention threshold at 74% coverage). Open: high distraction on unseen intents, adversarial slice, int8 |
| Benchmark v0 | done: 65 intents, 281 seeds, 11,240 paraphrases, 53,184 rows (`docs/dataset-v0.md`); superseded for intents and splits by A5 |
| Rules baseline | done, both all-seeds and fair train-only versions |
| Router v0 (route + intent + span heads) | trained once on Colab T4; beats rules in distribution, fails on unseen intents (`docs/results-v0.md`) |
| Router v1 (tag heads + regularisation) | run R2 done 10-23: tags route cuts test critical misses to 3.1% but over-regularised (underfit); OOD still poor because capability does not transfer |
| No-context ablation | not run yet; included in the v1 notebook |
| int8 export | broken by dynamic quantization (route agreement 0.915 in R1, 0.730 in R2); recipe fix not started |
| LLM zero-shot baselines, paper outline, Table 4 | built in another session on branch `claude/vibrant-hamilton-1psf5i`; **not merged**; labels verified identical |

## Goal and success criteria

From the PRD (`roadgate-prd.md`). The router reads `(utterance, context)` and
picks one of five routes: LOCAL, CLOUD, CLOUD_MASKED, DEFER, REFUSE.

| metric | target |
| --- | --- |
| macro-F1 over 5 routes, test | > 0.85 |
| safety-weighted error (SWE) | < 40% of rules baseline |
| critical miss rate | < 1% |
| model size | < 1B params, < 600 MB int8 |
| on-device latency p95 | < 50 ms |

**Critical miss** = a safety-critical command not run locally, or a REFUSE
case that executes anyway.

## Approaches

### A1. Policy-first labels (done)

Labels come from a deterministic rulebook, not from an LLM's judgment. Each
intent carries four hand-written tags; the rulebook combines tags with the
car's context.

- **Tags:** capability (local_ok, needs_small_local, cloud_preferred,
  needs_cloud), distraction (low, medium, high), actuation (none, comfort,
  safety_critical, restricted), consequence (none, reversible, irreversible).
- **Policy tables:** driving demand from speed x workload; distraction gate;
  actuation gate. Editable as JSON (`src/autogate_bench/policy.py`).
- **Precedence:** safety-critical -> LOCAL; restricted while moving -> REFUSE;
  too distracting -> REFUSE; local-capable -> LOCAL; no signal -> DEFER (or
  LOCAL for cloud_preferred); sensitive spans -> CLOUD_MASKED (or LOCAL under
  strict privacy); else CLOUD.

### A2. Data: seeds, paraphrases, noise (done)

- 4 to 5 hand-written seeds per intent, each a distinct phrasing style
  (command, outcome, question, indirect, fragment).
- 40 paraphrases per seed: 24 plain, 6 Hinglish, 4 Kannada-English,
  6 disfluent. Written by Claude agents against `docs/paraphrase-spec.md`.
- 4 deterministic ASR-noise variants per seed.
- Slots filled from synthetic pools; sensitive spans recorded as offsets.
- 3 contexts per utterance, sampled toward route boundaries.
- Splits by seed (80/10/10); 6 intents held out entirely as OOD (8, chosen
  by tag, since A5).

### A3. Router v0: end-to-end route head (done, run 1)

Qwen3-0.6B + LoRA r=16 on all projections; heads for route (5-way, cost
weighted), intent (65-way), span (BIO). Route comes straight from the route
head.

### A4. Router v1: tag heads + regularisation (run R2 done)

**Why:** run 1 learned intent identity, so unseen intents defaulted to LOCAL.
Tags are what the rulebook uses and every intent has them, so predicting tags
should transfer to unseen intents.

- Four extra heads predict the tags from the sentence.
- Three route sources from one forward pass:
  - `e2e`: the route head, as in v0
  - `tags`: the rulebook applied to the predicted tags plus context
  - `combined`: the more cautious of the two
- Conservative tag decoding: actuation is safety_critical or restricted if
  p >= 0.3; distraction is high if p >= 0.3.
- Tag losses weighted by danger: actuation 1.0, distraction 1.0, capability
  0.5, consequence 0.3. Class weights are 1/sqrt(freq) x a danger factor.
- Regularisation: 2 epochs, lr 1e-4, LoRA r=8 attention-only, LoRA dropout
  0.1, head dropout 0.3, weight decay 0.05, intent weight 0.2, label
  smoothing 0.1, context dropout 0.2, word dropout 0.1.
- Config: `configs/train_qwen3_0.6b_tags.yaml`. Details: `docs/training.md`,
  "Tag heads".

### A5. Taxonomy expansion and OOD re-pick (done, 2026-10-23)

**Why:** R2's capability head failed OOD mostly because of the split:
`cloud_preferred` had 4 intents and 2 were held out, `needs_small_local` lost
3 of 8, and the holdout tested no `needs_cloud`, high-distraction, comfort or
safety-critical intent.

- **16 new intents** (65 -> 81), each a distinct goal with a rationale in
  `docs/taxonomy.md`: safety_critical `fog_lights_on`, `dip_high_beams`,
  `cancel_cruise_control`; restricted `open_fuel_door`,
  `disable_stability_control`, `adjust_steering_wheel`; cloud_preferred
  `find_parking`, `route_avoiding_tolls`, `fuel_prices_nearby`,
  `place_opening_hours`; needs_small_local `read_email_aloud`,
  `translate_phrase`, `estimate_trip_cost`; needs_cloud
  `read_news_headlines`, `pay_bill`; local_ok `share_live_location`.
  Totals: cloud_preferred 4 -> 8, needs_small_local 8 -> 11, restricted
  6 -> 9, safety_critical 5 -> 8, medium 11 -> 14, high 8 -> 9,
  irreversible 8 -> 10.
- **5 seeds each** (281 -> 361 seeds), same voice rules; no new slots.
- **New OOD holdout (8):** `set_ambient_lighting`, `fog_lights_on`,
  `open_fuel_door`, `route_avoiding_tolls`, `compose_long_email`,
  `read_news_headlines`, `estimate_trip_cost`, `pair_bluetooth_device`.
  Every tag value keeps >= 6 training intents and >= 70% of its intents in
  training (smallest: cloud_preferred 7/8, safety_critical 7/8, high 7/9);
  the holdout covers every value of all four tags and six of seven groups.
  Enforced by `tests/test_splits.py`.
- **Labels changed:** pilot sha256 moved to `0c7139ac...`; splits differ for
  every intent, so **R0-R2 numbers are not directly comparable** with runs
  on the new data. Old checkpoints also have a 65-way intent head.
- **Paraphrases for the 16 new intents: done** (3,200 lines by four Claude
  agents against the spec). Whole directory: 81 files, 14,440 lines, 0
  errors under strict validation. Flagged lines are added to
  `docs/paraphrase-review-notes.md`.
- **Full dataset v1:** 66,414 rows, 22,138 utterances; train 48,054, val
  6,210, test 5,805, OOD 6,345. Routes: LOCAL 40,513, REFUSE 8,468, CLOUD
  8,249, DEFER 7,280, CLOUD_MASKED 1,904. Sensitive rate 15.4%. Every row
  re-checked: labels recompute from the rulebook, spans slice back, no split
  leaks, OOD intents only in OOD.

### A6. Synthetic goals and minimal pairs (train-only augmentation; data written 2026-10-28)

**Data (2026-10-28):** 60 synthetic goals, 565 utterances (high distraction
20 goals vs 7 training intents; needs_cloud 22; irreversible 12; restricted
5; safety_critical 4; no comfort). 290 minimal pairs over 115 intent pairs
(differs in capability 140, distraction 129, consequence 97, actuation 91;
57 with a restricted command phrased as a casual question). Re-checked by the
coordinator: route labels recompute from `rulebook.route` / `route_tags`,
spans slice back, all 4,635 new rows are train, and the 66,414 existing rows
are an exact byte prefix of the augmented file. Judgment calls flagged for
the audit: turn_signal_on, spray_windshield_washer and heat_side_mirrors as
safety_critical; move_driver_seat and fold_rear_seats as restricted;
30 "how do I X" (vehicle_manual_question) vs "do X" pairs.


**Why:** R4's open problems. On the 8 OOD intents 63 to 74% of
high-distraction rows are predicted low or medium (only 7 training intents
are high); the adversarial slice (restricted commands asked as questions)
has 25 to 30% critical misses; capability is 0.50 on OOD. The first two
Zomato ideas below, made concrete.

- **Synthetic goals ("reverse queries"):** new in-car goals invented for a
  tag combination, 8 to 12 utterances each, in
  `data/augment/synthetic_goals.yaml`. Labels come from the rulebook on the
  goal's own tags (`route_tags`); the intent head ignores these rows
  (`IGNORE_INDEX`), the tag and route heads learn from them. Aim: more
  distinct goals per rare tag value.
- **Minimal pairs (hard negatives):** two look-alike utterances (token
  Jaccard >= 0.4) of two existing, non-OOD intents whose tags differ so that
  some context routes them apart, in `data/augment/minimal_pairs.yaml`.
  Each side is an ordinary row of its intent.
- **Train only, nothing else moves.** Rows are appended after all others,
  every utterance with its own rng; the extra columns (`is_synthetic`,
  `pair_id`) appear only on augmentation lines. Checked on the real data:
  without `--augment` the full set is still sha256 `8cc12a4b...` (pilot
  `0c7139ac...`); with the test fixture the old file is a byte prefix of the
  new one (66,414 lines + 114 train lines). Labels of existing rows did not
  change.
- **Leak guards (validator errors):** goal utterance vs any OOD seed or
  paraphrase token Jaccard <= 0.5 (rules-baseline tokenizer); goal name or
  description may not contain an OOD intent's key words (ambient, mood
  light, fog, fuel door/flap/lid/cap, petrol/tank flap or lid, filler, toll,
  long/compose/draft/write email, news, headline, trip/journey/drive cost,
  bluetooth, pair); pairs may not use OOD intents; any augmentation
  utterance vs any val/test seed or paraphrase <= 0.7; no exact duplicate of
  any seed or paraphrase. Goal tags must satisfy the `Intent` invariants.
- **Review:** `python -m autogate_bench.augment review --out
  docs/augment-review.md` lists every goal and pair, risk-sorted, with the
  nearest existing intent (and whether its tags differ) and the highest
  Jaccard to OOD and val/test.
- **Next:** generation agents write the data against
  `docs/augmentation-spec.md`, a human audits the review page, then R5
  (R4's config on the augmented data). R5 also writes val predictions so
  the following round can turn val route-changing tag errors into new
  pairs.

### A7. Decision-encoder router (run 2026-11-17/18; dropped by its stop rule)

**Why:** R4nc showed the model only has to predict context-free tags. That
is the interface of the "decision models" released since September 2026
(TypeSafe's Jev and open reproductions; Google's Gemini Decision 1 and
MediaPipe Decision Maker): one forward pass, typed answers with
probabilities, no generation. Hosted ones cannot route (the router must
decide before anything leaves the car), so the candidate is a small
on-device encoder.

- **Design:** frozen EmbeddingGemma 2 text tower (reported 270M, about
  191 MB at 4-bit) -> five heads (four tags + sensitive) -> expected-cost
  routing through the rulebook and `configs/cost_matrix.json` -> abstain
  above a risk threshold. Optional intent registry (nearest registered
  example phrase; tags from the table) and a keyword safety lane.
- **Why frozen first:** heads train in minutes on cached vectors, and the
  shipped encoder can be Google's quantized file unchanged, which avoids
  our broken int8 export.
- **Runs:** R6z zero-shot, R6a frozen + heads, R6a2 token-state heads, R6b
  registry, R6l safety lane, R6c fine-tuned, R6q 8-bit and latency. No
  augmentation data in reported runs, so they compare with R4.
- **Gates and verdict conditions:** `docs/plan-a7-decision-router.md`.
- **Caveat:** every model fact (size, memory, licence, latency) comes from
  web search summaries on 11-17; no page was opened. Phase 0 of the plan
  verifies them.
- The local generative model (for example Gemma 4 E2B) stays a separate
  model, called only for routes that need an answer written.

**Phase 0 (11-17): verified.** Model card, config and weights were read on
Hugging Face; the encoder loads and the licence allows our use (Gate 0
passed).

| claim | plan said (search) | verified 11-17 |
| --- | --- | --- |
| Model id | `google/embeddinggemma-2` | same; revision `f9972ac22e696a105c482202646979e33e4b5a1e`; **not gated** |
| Text tower size | 270M (130M + 140M) | 271,002,624 parameters loaded text-only (vision and audio configs dropped); card: 130M backbone + 140M embedder; 740M with all encoders |
| Memory at 4-bit | about 191 MB | not on the card; not checked (Phase 5) |
| Output | 768-d, mean pooled, truncatable | 768-d; mean pooling over every non-padding token incl. the prefix (`include_prompt: true`), then L2; MRL to 512/256/128 |
| Context | 8,192 tokens | same (card); 24 layers, d=512, sliding window 512 in the config (1,024 on the card) |
| Licence | Apache 2.0, maybe Gemma policy | Apache 2.0, and deployments must follow the Gemma Prohibited Use Policy; commercial use allowed |
| Half precision | fp16 NaN | confirmed by the card: use bf16 or fp32, never fp16 |
| Task prefix (classification) | - | `task: classification \| query: ` (symmetric tasks use the same prefix for all inputs) |
| Library | - | needs `transformers >= 5.18` (class `EmbeddingGemma2Model`); locked at 5.23.0 |
| On-device runtime, latency | MediaPipe, 60-80 ms | not checked yet (Phase 5) |

- Control encoder: `intfloat/multilingual-e5-small` (revision `614241f6`,
  118M, ungated), prefix `query: `, mean pooling + L2.
- Embedding cache (`python -m autogate_decide.embed`): 23,682 unique texts
  (22,137 in the 66,414-row set plus the A6 rows, kept only for the
  minimal-pair probe) on 4 CPU cores, fp32: Gemma 460 s (768-d), e5 53 s
  (384-d). All finite. Re-embedding 10 texts in a different batch changes
  values by at most 3e-7 (padding), not a determinism problem.
- Network: this needed a personal cloud environment "AutoGate" allowing
  Hugging Face (incl. `us.aws.cdn.hf.co`) and the PyTorch CPU index.

**Implementation choices made before any test scoring** (after an
independent review by a second model; see the decision log):

- Heads train on the 16,018 unique train utterances, not on rows (each
  utterance has three contexts).
- Head shape (linear or one hidden layer of 256) and weight decay are
  chosen per question by NLL on **held-out training intents** (5
  tag-balanced leave-intents-out folds), not on val: val holds only
  training intents, so val loss rewards memorising intents. The MLP wins
  on val and loses badly on held-out intents (e5 capability: val NLL 0.85
  against held-out 2.07).
- Two sets of heads: weighted (R4's class weights) for the conservative
  decoder, unweighted for expected cost and the risk score (a temperature
  cannot undo a class-weight prior shift).
- A joint head over the 14 (capability, distraction, actuation)
  combinations seen in training is mixed with the product of the
  marginals (weight lambda on held-out intents), because tags are fixed
  per intent and the product puts mass on combinations no intent has.
- Expected cost is restricted to routes the rulebook can produce in the
  row's context. Gate 2 stays on the pre-registered conservative decoder;
  expected cost is reported beside it.
- Risk score for abstention = p(the chosen route is a critical miss), not
  the minimum expected cost (which spends coverage on cheap CLOUD vs
  CLOUD_MASKED confusions). Threshold: largest val coverage with covered
  critical miss <= 1%; a Wilson 95% upper bound is reported too.
- Intervals: a bootstrap over **intents** is reported beside the row-level
  one (OOD has 8 intents; rows of one intent are correlated).
- Gate 2 parity excludes masking: the sensitive head is a yes/no answer
  with no spans, so the span leak rate is not measured.

### A8. Fine-tune the decision encoder on the A6 data (planned 2026-11-18, nothing run)

**Why:** A7's fine-tuned encoder (R6c) beat R4 on test (0.867 / 0.308 /
4.8%) but collapsed on unseen intents (OOD 1.166 / 30.5%; high distraction
missed 88%). The bet: the collapse comes from too few distinct goals per
tag value. A6's 60 synthetic goals (20 high distraction, against 7 such
training intents) and 290 minimal pairs are the fix it was built for.

- **Design:** R6c's recipe on `generate --augment data/augment`, with and
  without distillation towards the frozen encoder's vectors (R7a, R7b).
  Controls: R6c's recipe without A6 (R7-ctl) and the frozen encoder
  (R7-frz). LoRA is cut (it lost on val in A7).
- **Selection sees unseen intents:** a proxy holdout of 8 training intents
  with the real OOD set's tag tuples is removed from train and val during
  selection (with A6's leak guards against it); 58 minimal pairs are a
  report-only probe never trained on. The proxy must first reproduce A7's
  ordering (frozen beats fine-tuned). Rule: a val filter relative to
  R7-ctl, then the lowest proxy critical miss. Go on only if the winner
  beats R7-ctl and matches R7-frz on the proxy, on two fine-tune seeds,
  under an intent-clustered paired bootstrap.
- **One final model** (R7), retrained on all training data, scored once
  (the fourth use of the test set; stated beside the result). Gate 2 and
  the verdict thresholds as in A7.
- **Blocked** on the owner's audit of `docs/augment-review.md`; R5 (R4 on
  the same data) is the comparison that tells whether A6 helps any router.
- **Phase 0 prepared (11-18), nothing trained:**
  - Proxy holdout (seed 0): read_document, send_text_message,
    emergency_call, disable_lane_assist, play_video, book_restaurant,
    fuel_prices_nearby, adjust_window. Same capability, distraction and
    actuation as the 8 OOD intents.
  - On the current A6 data, selection runs move 5,535 proxy rows out of
    train and val and drop 102 of the 290 pairs (pairs touching a proxy
    intent). They also drop 7 of the 60 goals: 6 on key words, among them
    two video-streaming goals and a movie one; 1 on token overlap, a
    send-a-message phrasing.
  - 59 probe pairs (465 rows) are held out in every phase.
  - Selection therefore runs on a smaller A6 than the final model; that is
    the price of a clean unseen-intent proxy.
  - Distillation weight beta = 0.0276, from R6c's drift: the fine-tuned
    vectors have mean cosine 0.26 with the frozen ones, so R6c rewrote the
    geometry almost entirely.
  - R7-frz trains its heads through the same fine-tune path with the
    encoder frozen, so all proxy comparisons share one decoder.

### Ideas not yet tried

From Zomato's natural-language search write-up (blog, 2026-10-23), mapped to
AutoGate; the first two are now A6 (plumbing only), the rest not started:

- **Tag-conditioned synthetic utterances ("reverse queries").** Generate
  requests from a target tag combination (e.g. needs_cloud + high
  distraction + irreversible) instead of only paraphrasing fixed seeds.
  More distinct intents per rare tag value, aimed at the OOD distraction
  gap. Generated items need a human audit before use.
- **Hard-negative minimal pairs.** Utterances that share most words but
  differ in one tag (set vs cancel cruise; unlock doors vs child locks;
  read vs summarise messages), plus a loop that turns each run's
  route-changing tag errors on val into new training pairs. Aimed at the
  adversarial slice and confusable safety intents.
- **Fast deterministic lane for safety-critical commands**, run in parallel
  with the model: a high-precision keyword match that routes LOCAL at once
  and can only make a decision more cautious.
- **Multi-constraint and compound requests** ("defrost and read my
  messages"), with the rule that any refused part refuses the whole.
- **Manually verified evaluation set** (they use 1,000+ queries with a
  "critical entity" column): our PRD's 300-item audit, with a critical-tag
  label per item.

Highest priority after R2:

- **R4: tags with R1's capacity.** Keep tag heads; go back to r=16 on all
  projections, lr 2e-4, 3 epochs, head dropout 0.1, intent weight 0.5; drop
  label smoothing; keep context dropout 0.2; word dropout 0.05. Expect the
  route head to recover R1 numbers and the tag heads to sharpen.
- ~~Paraphrases for the 16 A5 intents~~: done; R0f is the new fair rules
  baseline. R4 config and notebook are ready.
- ~~Rebalance the OOD holdout by tag~~ and ~~more intents in rare tag
  values~~: done in A5.
- **Treat capability as policy, not perception** (as on the other branch):
  predict only distraction, actuation, consequence from text; take
  capability from the policy table when the intent is known and default to
  the cautious value (cloud_preferred) when it is not.

- More seeds per intent (6 to 8). Most likely fix for the val-to-test gap.
- On-the-fly slot refill and ASR noise each epoch (currently fixed at
  generation time).
- Early stopping on val SWE.
- int8 fix: per-channel weight-only quantization, or static quantization with
  val calibration; require route agreement > 0.99.
- Cross-validation over seed folds for tighter confidence intervals.
- Raise the sensitive rate from 16% toward the PRD's 35%.
- Real in-car utterances (KVRET) as a second OOD set, to answer the
  single-model-paraphrase concern.

## Runs

All numbers on held-out splits. SWE = safety-weighted error, lower is better.

**Comparability:** R0a-R2 use the 65-intent taxonomy and the old 6-intent
OOD holdout. From A5 (10-23) on, intents, seeds and splits differ, so R1/R2
are not directly comparable with future runs; R0d/R0e are the first rows on
the new splits (seed-only pilot).

### Test split (new phrasings of known intents)

| run | date | macro-F1 | SWE | critical miss | span leak | notes |
| --- | --- | --- | --- | --- | --- | --- |
| R6c Gemma fully fine-tuned + heads, conservative | 11-18 | **0.867** | **0.308** | **4.8%** | n/a | passes Gate 2's test bars, beats R4nc; 5 seeds identical to 0.002; intent-bootstrap crit CI [0.3%, 9.7%] |
| R6c, expected cost | 11-18 | 0.868 | 0.302 | 4.7% | n/a | |
| R6c, conservative + abstention | 11-18 | 0.884* | 0.233* | 0.8%* | n/a | *87.9% coverage (U1 met); abstains on 4.6% of safety-critical rows (U4 fails) |
| R6a2 Gemma frozen, layer-mix token heads, conservative | 11-18 | 0.834 | 0.467 | 10.7% | n/a | **Gate 2 fails** again; 5-seed mean 0.825 / 0.457 / 8.7% (sd 0.019 / 0.019 / 1.8 pts) |
| R6a2, expected cost | 11-18 | 0.804 | 0.461 | 8.3% | n/a | |
| R6a2, conservative + abstention | 11-18 | 0.876* | 0.270* | 0.4%* | n/a | *88.5% coverage (U1's bar met); abstains on 11% of safety-critical rows (U4 fails) |
| R6a EmbeddingGemma 2 frozen + heads, conservative | 11-17 | 0.800 | 0.562 | 9.7% | n/a | **Gate 2 fails** (needs 0.83 / 0.37 / 7.0%); 5 seeds, sd <= 0.003 |
| R6a, expected cost | 11-17 | 0.808 | 0.471 | 11.3% | n/a | lower SWE, more critical misses |
| R6a, conservative + abstention | 11-17 | 0.838* | 0.409* | 0.5%* | n/a | *answered rows, 80% coverage; abstains on 32% of safety-critical rows |
| R6a-ctl e5-small frozen + heads, conservative | 11-17 | 0.744 | 0.659 | 14.0% | n/a | control encoder: worse everywhere |
| R6z-gemma zero-shot, conservative / expected cost | 11-17 | 0.320 / 0.259 | 1.531 / 1.462 | 22.6% / 16.0% | n/a | Gate 1 (informational): prompting alone is not enough |
| R6z-e5 zero-shot, conservative / expected cost | 11-17 | 0.264 / 0.222 | 1.703 / 1.458 | 41.2% / 10.6% | n/a | |
| R4 with context, e2e | 10-27 | 0.821 | 0.360 | 5.6% | 22.9%* | route head now matches the tags route |
| R4 with context, tags | 10-27 | 0.832 | 0.380 | 6.3% | 19.5%* | |
| R4 with context, combined | 10-27 | 0.832 | 0.369 | **5.1%** | 19.6%* | |
| R4nc no context, e2e | 10-27 | 0.359 | 1.105 | 17.4% | 4.3%* | route head blind to speed/signal |
| R4nc no context, **tags** | 10-27 | **0.850** | **0.342** | 6.0% | 20.5%* | context enters through the rulebook |
| R4nc no context, combined | 10-27 | 0.657 | 0.759 | **5.0%** | 17.4%* | over-refuses (REFUSE predicted 2x) |
| R0f rules, train seeds only (**81-intent full**) | 10-23 | 0.458 | 1.224 | 30.3% | - | **the fair baseline from now on** |
| R0d rules, seeds only (81-intent pilot) | 10-23 | 0.778 | 0.248 | 0% | - | A5 sanity check, all-seeds matcher |
| R0e rules, train seeds only (81-intent pilot) | 10-23 | 0.437 | 1.419 | 37.9% | - | A5 fair baseline, pilot only |
| R0a rules, seeds only (pilot) | 09-25 | 0.875 | 0.315 | 0% | - | sanity check; too easy by design |
| R0b rules, all seeds | 09-25 | 0.627 | 0.919 | 22.5% | 85.7% | matcher saw test seeds; optimistic |
| R0c rules, train seeds only | 10-07 | 0.430 | 1.256 | 37.1% | - | the fair baseline |
| R1 router v0, e2e | 10-07 | **0.780** | **0.426** | **7.1%** | 33.4%* | Colab T4, 1h43m |
| R2 router v1, e2e | 10-23 | 0.477 | 0.809 | 10.5% | 32.7%* | route head underfit (val F1 0.58) |
| R2 router v1, tags | 10-23 | 0.728 | 0.504 | **3.1%** | 49.8%* | best critical miss so far |
| R2 router v1, combined | 10-23 | 0.656 | 0.637 | **2.9%** | 50.0%* | most cautious; over-refuses |

### OOD split (intents never seen in training)

6 intents for R0-R2; 8 different intents from R0d on (A5).

| run | macro-F1 | SWE | critical miss | notes |
| --- | --- | --- | --- | --- |
| R6c fully fine-tuned, conservative | 0.611 | 1.166 | 30.5% | **fails Gate 2's OOD bars**; worse than frozen R6a: fine-tuning collapsed unseen intents onto training tag clusters |
| R6c, conservative + abstention | 0.655* | 0.796* | 10.1%* | *82% coverage |
| R6a2 layer-mix token heads, conservative | 0.638 | 1.089 | 27.3% | OOD crit fails Gate 2 (25%); 5-seed mean 1.089 / 26.3% |
| R6a2, conservative + abstention | 0.675* | 0.786* | 9.6%* | *83% coverage (U3a needs <= 5% at >= 50%) |
| R6a EmbeddingGemma 2 frozen + heads, conservative | 0.621 | **1.015** | 21.7% | passes Gate 2's OOD bars (1.12 / 25%); intent-bootstrap crit CI [6%, 50%] |
| R6a, expected cost | 0.584 | 1.129 | 29.3% | |
| R6a, conservative + abstention | 0.653* | 0.734* | 4.2%* | *82% coverage (U3a needs <= 5% at >= 50%) |
| R6a-ctl e5-small, conservative | 0.575 | 1.255 | 33.1% | |
| R6z-gemma zero-shot, conservative / expected cost | 0.294 / 0.224 | 1.734 / 1.658 | 30.0% / 28.6% | |
| R6z-e5 zero-shot, conservative / expected cost | 0.324 / 0.219 | 1.691 / 1.565 | 38.8% / 13.9% | |
| R4 with context, e2e | 0.637 | 1.032 | 22.5% | |
| R4 with context, tags | 0.628 | 1.057 | 23.0% | |
| R4 with context, combined | **0.645** | **0.974** | **19.1%** | 55% of fair-rules SWE |
| R4nc no context, e2e | 0.238 | 1.619 | 36.2% | |
| R4nc no context, **tags** | **0.640** | **1.038** | 22.9% | capability OOD 0.52 (R2 0.33) |
| R4nc no context, combined | 0.423 | 1.270 | **19.1%** | |
| R0f rules, train seeds only (**81-intent full**) | 0.324 | 1.771 | 44.3% | the fair baseline from now on |
| R0d rules, seeds only (81-intent pilot) | 0.766 | 0.284 | 0% | matcher saw OOD seeds |
| R0e rules, train seeds only (81-intent pilot) | 0.321 | 1.766 | 36.1% | old pilot OOD: 2.21 SWE, 72.7% |
| R0b rules, all seeds | 0.583 | 1.044 | 34.1% | not fair: saw OOD seeds |
| R0c rules, train seeds only | 0.231 | 2.231 | 78.6% | |
| R1 router v0, e2e | 0.442 | 1.312 | 25.7% | defaults to LOCAL on unseen intents |
| R2 router v1, e2e | 0.298 | 1.694 | 30.7% | underfit route head |
| R2 router v1, tags | **0.477** | 1.307 | 22.0% | small gain over R1; capability tag fails OOD |
| R2 router v1, combined | 0.396 | 1.526 | **19.8%** | |

\* Span leak is measured on test + OOD combined, on rows the run routed to the cloud, so it is not comparable across rows with different routing.

## Run details

### A8 Phase 1, controls and the proxy check (2026-11-18): STOP, owner asked

- Augmented data regenerated; base set still `8cc12a4b...` and an exact
  prefix (audit made no corrections). Frozen cache for the augmented texts
  rebuilt (501 s). `scripts/a8_phase1.sh`, outputs in `runs/a8/`.
- **R7-frz** (frozen encoder, heads only; epochs 1-3): val SWE 0.81 /
  0.73 / 0.73; proxy critical miss 24.2% / 24.1% / 23.4%, distraction
  dangerous miss 20.8% / 23.3% / 22.5%.
- **R7-ctl** (R6c recipe, no A6; 30 min an epoch): val SWE 0.386 / 0.326 /
  0.344, val critical miss 7.8% / 6.4% / 9.2%; proxy critical miss 26.6% /
  23.6% / 24.0%, distraction dangerous miss 62.8% / 49.7% / 46.4%.
- **Proxy validity check failed** (pre-registered rule: the frozen control
  better than R7-ctl on proxy critical miss *and* distraction dangerous
  miss). R7-frz epoch 3 against R7-ctl epoch 2: distraction dangerous miss
  -27 points (95% intent-bootstrap interval -30 to -25), but critical miss
  -0.2 points (interval -12 to +14). Per proxy intent, critical miss frozen
  vs fine-tuned: play_video 9.1% vs 25.3%, read_document 17.0% vs 33.3%
  (the A7 collapse, reproduced); disable_lane_assist 69.4% vs 38.2% (the
  frozen encoder fails this restricted command); book_restaurant 61.3% vs
  58.7%; emergency_call 2.0% vs 1.1%. The real OOD set had no such
  restricted-intent reversal (open_fuel_door 27% vs 26%).
- As the plan says, Phase 1 stopped before any A6 candidate (R7a, R7b)
  was trained; the owner decides how to go on.

### R6c: A7 Phase 4 result and the end of A7 (2026-11-18)

- Full fine-tune (epoch 2) re-embedded the dataset (412 s); fresh linear
  heads chosen on val NLL (weight decay 1.0 for capability, distraction,
  consequence and joint, 0.1 actuation, 0.01 sensitive); configuration
  frozen in `43088be` (`configs/decide/R6c.json`), then scored once. The
  test set was used for the third time in A7 (R6a, R6a2, R6c).
- **Test: 0.867 / 0.308 / 4.8%** (conservative), better than R4nc tags
  (0.850 / 0.342 / 6.0%) and every earlier run; tag accuracy capability
  0.85, distraction 0.94, actuation 0.92, consequence 0.89; adversarial
  slice critical miss 8.8% (R4 25-30%); minimal pairs 69% right on both
  sides (R6a 35%); per variant critical miss plain 3.9%, ASR 3.4%,
  disfluent 4.5%, Hinglish 7.6%, Kannada-English 8.5% (2.2x plain).
  Sensitive recall 0.997, precision 1.0.
- **OOD: 0.611 / 1.166 / 30.5%**: fails both Gate 2 OOD bars and is
  worse than the frozen encoder (R6a 1.015 / 21.7%). Distraction dangerous
  miss 88% (R6a 80%); OOD ECE capability 0.34, distraction 0.32
  (confidently wrong on unseen intents). Per intent: pair_bluetooth_device
  98% critical miss, compose_long_email 52%, read_news_headlines 27%,
  open_fuel_door 26%, fog_lights_on 4%.
- Abstention (val threshold 0.024): test 0.8% critical miss on 87.9%
  coverage, abstaining on 4.6% of safety-critical rows; OOD 10.1% on 82%.
- **Gate 2 outcome B after Phase 4: the plan says stop.** A7 is dropped
  and the project returns to R5 (waiting for the A6 audit). Verdict table
  (section 7) is therefore not evaluated as such; for the record, on R6c
  U1's numbers are met on test, U7 is met (0.997 / 0.956), and U3a, U4 and
  U6 fail. Phase 3 (registry, lane) and Phase 5 (artifact retraining,
  latency) were built or previewed but not run, as the plan only reaches
  them after a passing gate.
- What A7 showed: the trade-off is in the encoder, not the heads. A frozen
  general-purpose encoder keeps unseen intents apart as well as R4 but
  cannot separate known look-alikes; supervised fine-tuning fixes known
  intents and erases what helped on unseen ones. Nothing in A7 closes the
  OOD distraction gap; that is what A6's synthetic goals are for.

### R6c: A7 Phase 4 pre-registration (2026-11-18, before any fine-tuning)

After a third independent review. `python -m autogate_decide.finetune`:
- EmbeddingGemma 2 text tower (`f9972ac2`), fp32, mean pooling, heads for
  the four tags, sensitive and the 14-way joint; loss weights actuation 1,
  distraction 1, capability 0.5, consequence 0.3, sensitive 0.5, joint 0.5;
  tag class weights as R4. Batch 32, 3 epochs, seed 0.
- **LoRA r16** (alpha 32, dropout 0.05) on q, k, v, o only, encoder lr
  1e-4; **full fine-tune** encoder lr 2e-5. Heads in their own group at lr
  1e-3; 6% warmup, then linear decay.
- Per run, the epoch with the lowest val SWE (in-model heads, conservative
  decoder) is kept. **Between LoRA and full, the lower val SWE goes on; only
  that one is scored on test and OOD** (so the A7 test set is used three
  times in all: R6a, R6a2, R6c).
- The kept encoder is saved merged; the dataset is re-embedded with it and
  scored by `run --method heads --selection val`: linear heads, weight
  decay 0.01 / 0.1 / 1 / 10 chosen by **val NLL** (train-only folds are
  leaky once the encoder has fitted every training seed, so val is the only
  unfitted data; picking one of three epochs on val is a mild reuse of it),
  lambda fixed at 0.5, temperatures and abstention threshold on val.
- Deviations from the plan: CPU (4 cores, fp32) instead of a Colab T4 (no
  difference in precision; torch version and thread count are recorded);
  LoRA on attention only, not "all projections". The reviewer's suggested
  held-out-intent diagnostic fold (about an hour) is not run: epoch choice
  on val cannot see a collapse on unseen intents, and OOD is first seen at
  the single test scoring.

**R6c fine-tunes (11-18; val only, in-model heads, conservative decoder).**
CPU, torch 2.14.0+cpu, 4 threads; 16,018 train utterances, batch 32.

| run | epoch | train loss | val SWE | val macro-F1 | val critical miss | elapsed |
| --- | --- | --- | --- | --- | --- | --- |
| lora16 | 1 | 1.591 | 0.432 | 0.751 | 7.6% | 18 min |
| lora16 | 2 | 0.369 | 0.413 | 0.795 | 9.0% | 36 min |
| lora16 | 3 | 0.195 | 0.408 | 0.782 | 7.8% | 53 min |
| full | 1 | 1.066 | 0.340 | 0.845 | 6.9% | 26 min |
| full | 2 | 0.082 | 0.333 | 0.833 | 6.7% | 53 min |
| full | 3 | 0.009 | 0.341 | 0.833 | 7.7% | 80 min |

Kept: LoRA epoch 3 (val SWE 0.408) and full epoch 2 (val SWE 0.333); **full goes on**, as pre-registered.
For scale, R4's route head on val reached SWE 0.359 / macro-F1 0.808 /
critical miss 4.9% at its last epoch. Full fine-tuning drives train loss to
0.01 by epoch 3 and val SWE turns up: memorisation, which held-out
intents (OOD) will show if anything does. The merged encoders reproduce
the training-time vectors of five probe sentences (max difference 2e-8
full, 2e-7 LoRA).

### R6a2: A7 Phase 2b, layer-mix token heads (2026-11-17/18)

- Frozen EmbeddingGemma 2, per-token states of layers 6, 12, 18, 23
  (407,151 tokens, 3.3 GB fp32 cache, 7.5 min to build); heads as
  pre-registered (commit `7598b2e`); configuration frozen in `072d35f`
  (`configs/decide/R6a2.json`). Selection on CPU took about 2 hours over two
  container restarts (resumed from per-trial checkpoints; the first
  attempt's numbers reproduced exactly).
- Chosen: weight decay 10 (grid edge again) for capability, distraction,
  consequence, sensitive and joint; 1.0 for actuation; 3 or 6 epochs;
  lambda 0.5. The one-standard-error rule pushed most heads to the
  strongest decay.
- **Gate 2: outcome B.** Test 0.834 / 0.467 / 10.7% (five-seed mean 0.825 /
  0.457 / 8.7%): macro-F1 reaches the bar on seed 0 but not on the mean;
  SWE and critical miss fail. OOD 1.089 / 27.3% (mean 1.089 / 26.3%):
  SWE passes, critical miss fails (R6a passed both OOD bars). Seeds now
  differ (sd 1.8 points of test critical miss): the attention heads are
  not convex.
- Gains over R6a: test SWE 0.562 -> 0.467; sensitive recall 0.89 -> 1.00
  on test and OOD (U7 met); minimal pairs 35% -> 46% both sides right;
  test actuation dangerous miss 2.2% -> 0.6%; with val-chosen abstention,
  test critical miss 0.4% on 88.5% coverage (U1's numbers), abstaining on
  11% of safety-critical rows (R6a: 32%).
- Unchanged weak spots: distraction dangerous miss test 41%, OOD 83%;
  OOD capability 0.51 (ECE 0.29); Kannada-English test critical miss 18.8%
  vs plain 9.4% (2.0x: U6 borderline); OOD abstention leaves 9.6% critical
  misses on 83% coverage.

### R6a, R6a-ctl, R6z: A7 Phases 1-2 (2026-11-17)

- Code: `src/autogate_decide/` (commit `ba33c44` froze R6a's
  configuration before test/OOD scoring; R6a-ctl and R6z frozen in
  `8c14a2b`). Frozen choices: `configs/decide/R6a.json`,
  `R6a-ctl.json`, `R6z-*.json`; option texts `configs/decide/questions.yaml`.
  Dataset sha256 `8cc12a4b...` (no `--augment`). CPU only, 4 cores.
- Command: `python -m autogate_decide.run --cache
  data/generated/decide/<encoder> --rows data/generated/full/rows.jsonl
  --pairs-rows data/generated/full_aug/rows.jsonl --out runs/<run>
  --score val,test,ood [--config configs/decide/<run>.json]`.
- **R6a (EmbeddingGemma 2 `f9972ac2`, frozen):** selection took 22 min
  (36 head fits per question). Chosen: linear heads, weight decay 1.0 for
  capability, distraction, consequence and the joint head (the edge of
  the grid), 0.01 for actuation, a 256-unit MLP for sensitive; mixing
  weight 0.5. Five seeds differ by sd <= 0.003 on every gate metric
  (convex heads), so the intent bootstrap is the honest interval.
- **Gate 2: outcome B.** Test fails on all three bars (0.800 / 0.562 /
  9.7% against 0.83 / 0.37 / 7.0%); OOD passes both (SWE 1.015, critical
  miss 21.7%; R4nc 1.038 / 22.9%). The frozen encoder generalises to unseen
  intents about as well as R4 but loses on known ones: test tag accuracy
  capability 0.78, distraction 0.86, actuation 0.84, consequence 0.81
  (R4nc val: 0.78 / 0.92 / 0.95 / 0.83).
- Diagnostics (conservative, seed 0):
  - adversarial slice critical miss: test 11.4% (R4 25-30%), OOD 44.5%;
  - distraction dangerous miss: test 42%, OOD 80% (R4nc 74%); actuation
    dangerous miss test 2.2%, OOD 16% (R4 18.7%);
  - per variant, test critical miss: plain 7.6%, ASR 9.2%, disfluent
    11.3%, Hinglish 12.8%, Kannada-English 16.2% (2.1x plain: U6 fails);
  - sensitive: recall 0.89 test / 0.85 OOD, precision 0.99 (U7 needs 0.95);
  - ECE after val temperature: test <= 0.04 per question; OOD capability
    and distraction 0.20 (overconfident on unseen intents);
  - OOD per intent: fog_lights_on 1% critical miss, open_fuel_door 27%,
    compose_long_email 53%, pair_bluetooth_device 58%;
  - minimal-pair probe: 35% of 290 pairs have every routing tag right on
    both sides (actuation pairs 47%, capability pairs 14%).
- **Expected cost vs conservative:** expected cost lowers SWE (test 0.471)
  but raises critical misses (11.3%; OOD 29.3%), as predicted: it needs
  p(safety_critical) > 0.55 to prefer LOCAL over REFUSE.
- **Abstention** (p(critical miss) threshold 0.044 chosen on val): test
  0.5% critical miss on 80% coverage (U1 needs >= 85%), OOD 4.2% on 82%.
  But it abstains on 32% of test safety-critical rows (U4 needs <= 1%):
  a safety command that is asked again is itself a delay.
- **R6a-ctl (e5-small):** test 0.744 / 0.659 / 14.0%, OOD 0.575 / 1.255 /
  33.1%. The approach depends on the encoder; Gemma is better on every
  metric, most on OOD and on the adversarial slice.
- **R6z (zero-shot):** test macro-F1 0.32 at best; prompting alone is far
  below R4, as expected (Gate 1 informational; no leak suspected).
- **Post-hoc diagnostic (val only, not a gate run):** R6a with the head
  shapes val NLL prefers (MLPs, lower weight decay) scores val 0.716 /
  0.619 / 12.3% (expected cost 0.739 / 0.496 / 8.4%), against 0.659 /
  0.771 / 14.5% as selected. Part of the test gap is the held-out-intent
  criterion trading in-intent fit for OOD; the rest (R4 val SWE 0.36) is
  not closed by capacity on pooled vectors.

**Phase 5 preview (11-17; facts only, not run R6q).** Google ships the
text tower for LiteRT-LM as `litert-community/embeddinggemma-2-text-270m-litert-lm`
(revision `9be6e8b9`, Apache 2.0, not gated): **int4 QAT** (transformer and
embedder), 165 MB for the CPU/GPU build plus per-chip NPU builds. It
returns only pooled vectors (128-768 d): pooled heads (R6a) can run on it;
heads that read token states or intermediate layers (R6a2) cannot. The
model card's own CPU latencies at the 128-token signature: S26 Ultra 27 ms,
Arm Linux 105 ms, Raspberry Pi 5 161 ms; TPU/NPU builds 8-14 ms. Here
(`litert-lm` 0.19.2, dev CPU shared with a training job, so an upper
bound): p50 97 ms, p95 110 ms on one thread. On 300 utterances its vectors
have cosine 0.968 on average (min 0.86) with our fp32 vectors, so heads
must be retrained on the artifact's own vectors (plan step 5.2) and U5's
"8-bit" becomes "int4 as shipped". U5 latency (p95 <= 50 ms) is out of
reach on a generic CPU; it needs a phone-class CPU or an accelerator.

**R6a2 pre-registration (11-17, before any R6a2 scoring).** After the
second review: (1) heads read a learned mix of encoder layers 6, 12, 18 and
23 (each 512-d, layer-normalised) with two attention queries plus the mean;
this goes beyond the plan's "attention pooling" and means Phase 5 needs a
runtime exposing those layers; (2) selection criterion `blend`: mean of
held-out-intent and held-out-seed NLL on train-only folds (3 each), then
the most regularised configuration within one standard error; (3) grid
widened to weight decay 0.01 / 0.1 / 1 / 10 and 3 or 6 epochs, because
R6a's choice sat at the grid edge (a train-only fact). Temperatures and
abstention thresholds still come from val; the conservative decoder uses
the val temperature. R6a is not re-scored for the gate. The BIO span head
is not built: Gate 2 does not score masking, and it is only needed if the
approach survives.

### R4: tags with R1 capacity, 81 intents, with context (2026-10-27)

- `configs/train_qwen3_0.6b_tags_r4.yaml`, Colab T4, 2h05m training.
  10.2M of 606M params trained.
- Val by epoch (route head): SWE 0.452 -> 0.398 -> 0.359; macro-F1 0.747 ->
  0.817 -> 0.808; critical miss 6.2% -> 9.1% -> 4.9%. Train loss 0.16: no
  memorisation (R1: 0.002) and no underfit (R2: 2.6).
- Test: all three route sources land together. e2e 0.821 / SWE 0.360 /
  5.6%; tags 0.832 / 0.380 / 6.3%; combined 0.832 / 0.369 / 5.1%. SWE is
  29-31% of the fair rules baseline (PRD target < 40%: met).
- OOD: combined is best, 0.645 / 0.974 / 19.1% (55% of fair-rules SWE).
- **Context in the model adds nothing over context in the policy.** The
  no-context tags route (R4nc) scores test 0.850 / 0.342 / 6.0% and OOD
  0.640 / 1.038 / 22.9%, within the confidence intervals of every R4
  source. The simplest design, a context-free tag model plus the rulebook,
  is as good as the end-to-end route head that reads the context.
- **Abstention reaches the PRD's critical-miss target.** Calibration
  temperature 2.19, ECE 0.084 -> 0.014. On val, threshold 0.85 gives 0.9%
  critical misses on 74% coverage (R1 needed 0.95 for 54%); 0.90 gives
  0.5% on 66%. This uses the route head's confidence; the tags route has
  no calibrated confidence yet.
- Tags, OOD: capability 0.50 (R4nc 0.52), distraction dangerous miss 63.0%
  (R4nc 73.9%), actuation dangerous miss 18.7%, consequence 73.9%.
- Weak spots unchanged: adversarial slice macro-F1 0.23-0.25 and critical
  miss 25-30% for every source; sensitive-slice critical miss 66% (n=32).
- int8: route agreement 0.795, worse than R4nc (0.888). Still unusable.

### R0f: fair rules on the full 81-intent dataset (2026-10-23)

- Train-seed matcher, 66,414 rows. Intent accuracy: train 0.505, val
  0.243, test 0.254, OOD 0.
- Test: macro-F1 0.458, SWE 1.224, critical miss 403/1,331 (30.3%).
- OOD: macro-F1 0.324, SWE 1.771, critical miss 715/1,615 (44.3%).
- Comparable only with runs on dataset sha256 `8cc12a4b...`.

### R0d/R0e: rules on the 81-intent pilot (2026-10-23)

- After A5. Seed-only pilot (`generate --out data/generated/pilot
  --rng-seed 0`, sha256 `0c7139ac...`), 1,476 rows, 361 seeds; full tables
  in `docs/pilot.md`.
- All seeds: intent accuracy 0.994; test macro-F1 0.778, SWE 0.248; OOD
  0.766 / 0.284; no critical misses (the matcher has seen every seed).
- Train seeds only: intent accuracy val 0.304, test 0.233, OOD 0; test
  macro-F1 0.437, SWE 1.419, critical miss 11/29; OOD 0.321, 1.766, 13/36.
- OOD SWE for the fair rules fell from 2.21 to 1.77 on the pilot: each new
  OOD intent has training intents with the same tags, so a wrong keyword
  match more often lands on the right route.

### R4nc: no-context ablation of R4 (2026-10-27)

- `configs/train_qwen3_0.6b_tags_r4.yaml --no-context`, 81-intent data,
  Colab T4, 55 min. 10.2M of 606M params trained.
- Val by epoch (route head): SWE 1.188 -> 1.203 -> 1.143; macro-F1 0.34 ->
  0.36 -> 0.36. Tag accuracy at the end: capability 0.78, distraction 0.92,
  actuation 0.95, consequence 0.83.
- **Key finding: the tags route does not need context in the model.** The
  route head without context collapses (test macro-F1 0.36): it cannot know
  speed or signal. The tags route scores test macro-F1 0.850, SWE 0.342
  (28% of fair rules, meets the PRD's < 40%), because tags are
  context-free and the rulebook applies the context afterwards.
  Consequence for the paper: "remove the context" only hurts the end-to-end
  design. For the decomposed design the ablation has to remove context from
  the rulebook too, or be framed as "context in the model vs. context in the
  policy".
- OOD (8 held-out intents): tags route macro-F1 0.640, SWE 1.038 (59% of
  fair rules), critical miss 22.9%. Capability OOD accuracy 0.52, up from
  0.33 in R2: A5 helped.
- **New safety gap: high distraction on unseen intents.** OOD distraction
  dangerous-miss rate 73.9% (1,197 of 1,620 high-distraction rows called
  low or medium), from the two high OOD intents, compose_long_email and
  pair_bluetooth_device. Test split: 33.9%. These turn REFUSE into an
  executed task while moving.
- Consequence dangerous misses on OOD 82.8% (compose_long_email's
  irreversible); no route effect, but the confirmation step would be skipped.
- Adversarial slice still weak for the tags route: macro-F1 0.24, critical
  miss 29.6%.
- Calibration: temperature 1.55, ECE 0.104 -> 0.028.
- int8: route agreement 0.888. Still unusable.

### R2: router v1 with tags (2026-10-23)

- Commit `723573c`, `configs/train_qwen3_0.6b_tags.yaml`, Colab T4, 57 min
  training. 2.4M of 598M params trained (LoRA r=8, attention only).
- **Over-regularised: the model underfit.** Train loss stayed at 2.6 (route
  0.56) versus 0.002 in R1. Val route-head macro-F1 0.58 versus 0.86 in R1.
  Calibration temperature came out 0.88 (below 1 means underconfident) and
  ECE got worse after calibration (0.018 -> 0.034), so label smoothing had
  already flattened the probabilities.
- **Tags route is the best route source.** Test critical misses 3.1% versus
  7.1% for R1, with macro-F1 0.728 versus 0.780. It refuses more: REFUSE
  recall 0.83, precision 0.61.
- **Combined** gets the lowest critical misses (2.9% test, 19.8% OOD) but
  over-refuses (REFUSE predicted 2,968 times for 1,615 true).
- **OOD improved only slightly:** macro-F1 0.477 vs 0.442, critical miss
  22.0% vs 25.7%, SWE unchanged (1.307 vs 1.312).
- Tag accuracy, test / OOD:

  | tag | test | OOD | route-changing errors (OOD) | dangerous miss |
  | --- | --- | --- | --- | --- |
  | capability | 0.80 | **0.33** | 1,112 | - |
  | distraction | 0.86 | 0.58 | 592 | 0% of high |
  | actuation | 0.88 | 0.84 | 184 | 14.3% test, 2.4% OOD |
  | consequence | 0.85 | 0.58 | 0 (not a routing tag) | 34.5% of irreversible |

- **Why capability fails OOD: the split, mostly.** Five of the six OOD intents
  have rare capability values. `cloud_preferred` has only 4 intents in the
  whole taxonomy and 2 of them are OOD, so training saw 2. `needs_small_local`
  has 8 and 3 are OOD. The model cannot learn a class from 2 examples.
  Capability is also partly not in the wording: whether onboard data can
  answer depends on the car, which is why the other branch made it a policy
  table.
- Consequence dangerous misses (34.5% of irreversible called reversible or
  none) do not change routes but would skip the "are you sure?" step.
- Span exact-match F1 fell to 0.39 (R1 0.63), likely from underfitting and
  word dropout.
- int8 got worse: route agreement 0.730.
- Conclusion: the tags idea helps safety; the regularisation went too far;
  OOD is limited by how the OOD intents were chosen and by capability being
  partly a deployment fact. See next steps under "Ideas not yet tried".

### R1: router v0 (2026-10-07)

- Qwen3-0.6B, LoRA r=16 all projections, 3 epochs, batch 32, lr 2e-4.
  10.2M of 606M params trained.
- Val by epoch: SWE 0.496 -> 0.388 -> 0.336; macro-F1 0.806 -> 0.828 ->
  0.859. Train loss reached 0.002: memorisation.
- Test intent accuracy 0.503 (no better than keywords); OOD 0.000 by
  construction.
- Errors: LOCAL over-predicted. 276 of 321 critical misses were REFUSE
  predicted as LOCAL.
- Calibration: temperature 3.15, ECE 0.080 -> 0.016. Threshold 0.95 gives
  under 1% critical misses on 54% coverage; 0.80 gives 4.0% on 82%.
- Export: fp32 2,275 MB, exact. int8 571 MB, route agreement 0.915, intent
  0.707: unusable.
- Conclusion: overfit to training phrasings; learned intent identity, not
  transferable semantics. Led to A4.
- Full write-up: `docs/results-v0.md`.

### R0: rules baselines (2026-09-25, 10-07)

- Keyword intent matcher from seeds + regex PII detector + the rulebook.
- 2026-10-22: the v1 notebook's rules cell on Colab reproduced R0b exactly
  (test macro-F1 0.627, critical miss 197/876; OOD 0.583). The dataset
  regenerates identically there, so R2 is directly comparable to R1.
- Seed-only data: intent accuracy 0.99. Paraphrased data: 0.52. Confirms the
  benchmark is not too easy (PRD threshold: rebalance if rules > 0.9 F1).
- Regex finds 19% of sensitive spans; misses lowercase names.
- Write-ups: `docs/pilot.md`, `docs/dataset-v0.md`.

## Decision log

| date | decision | why |
| --- | --- | --- |
| 11-18 | A8 Phase 1 stopped at the proxy validity check (distraction collapse reproduced, critical-miss gap not); owner asked | pre-registered stop |
| 11-18 | Owner reports the A6 audit of `docs/augment-review.md` done, with no corrections (no change to `data/augment` on any branch); A8 Phase 1 starts. R5 is unblocked too (Colab) | owner's word in the session |
| 11-18 | A8 runs on this CPU, started by `scripts/a8_phase1.sh` as soon as the owner reports the A6 audit done | owner's instruction; about 9 hours, resumable after container restarts |
| 11-18 | Plan A8: fine-tune the decision encoder on the A6 data, selected on a proxy holdout of training intents | owner asked for this follow-up; R6c failed only on unseen intents, which A6 targets; A7's selection could not see unseen intents |
| 11-18 | **Drop A7** (the plan's stop rule: Gate 2 failed after Phase 4) and return to R5 | R6c (full fine-tune) passes every test bar and beats R4, but OOD critical miss 30.5% and SWE 1.166 fail; no A7 configuration passes test and OOD together |
| 11-18 | R6c: full fine-tune goes on over LoRA (val SWE 0.333 vs 0.408); score only the winner on test | pre-registered; keeps A7 to three uses of the test set |
| 11-18 | A7 Gate 2 outcome B for R6a2 too (test SWE 0.467, critical miss 10.7%; OOD critical miss 27.3%); go to Phase 4 (fine-tune, R6c) as the plan says | frozen encoders, pooled or token-level, do not reach R4 on known intents |
| 11-17 | A7 Gate 2 outcome B for R6a (test fails, OOD passes); go to Phase 2b (token-state heads, R6a2) as the plan says | test 0.800 / 0.562 / 9.7% vs bars 0.83 / 0.37 / 7.0% |
| 11-17 | Owner confirmed the proposed A7 verdict thresholds (plan section 7) as written, before any test scoring | plan requires it before Phase 2 test scoring |
| 11-17 | A7 heads: choose head shape and weight decay on held-out training intents, not val; two head sets (weighted for conservative, plain for expected cost); joint-combination head; p(critical miss) as the abstention score; Gate 2 stays on the conservative decoder | independent review: val holds only training intents; class weights distort expected cost; product of marginals puts mass on tag combinations no intent has |
| 11-17 | Plan A7: test a Jev-style decision router on a small frozen encoder, in phases with pre-set gates | one-pass typed decisions match what R4nc needs; a 270M encoder fits the size and latency targets where a shared 2B+ assistant model cannot; frozen-first makes iteration minutes and sidesteps int8 |
| 10-28 | Add synthetic goals and minimal pairs as train-only augmentation (A6); R5 = R4's config on that data, scored on val too | R4's gaps (OOD high distraction, adversarial slice, OOD capability) come from too few distinct goals per tag value and no look-alike negatives; train-only keeps every eval split and all R4 numbers comparable |
| 10-27 | Lead design candidate: context-free tag model + rulebook (R4nc tags) | matches the context-aware route head (R4) within CIs, simpler input, policy swappable without retraining |
| 10-27 | The no-context ablation is reinterpreted (see R4nc): it tests context in the model, not context in the decision | the tags route reaches 0.85 test macro-F1 with no context in the model |
| 10-23 | Runs on the 81-intent data write to `runs-v1/`; R4 is the next training run | keep incomparable runs out of one comparison table |
| 10-23 | Expand the taxonomy to 81 intents and re-pick an 8-intent OOD holdout balanced by tag (A5) | R2: capability could not transfer with 2 training `cloud_preferred` intents; the old holdout tested no needs_cloud, high, comfort or safety-critical intent |
| 10-23 | Next run backs off regularisation (R4 above) | R2 underfit: train loss 2.6, val route F1 0.58 |
| 10-22 | Add tag heads; score three route sources | R1 failed on unseen intents; tags transfer |
| 10-22 | Regularise run 2 (see A4) | R1 train loss 0.002, val-test gap |
| 10-07 | Fair rules baseline uses train seeds only | all-seeds matcher saw test and OOD seeds |
| 10-07 | Notebook runs on Colab and Kaggle; outputs can go to Drive | user trains on Colab; disk wiped on disconnect |
| 10-07 | Repo made public | Colab and Kaggle clone without a token |
| 09-25 | Paraphrases written by Claude agents, not a paid API | free under the subscription; API script kept for reproducibility |
| 09-25 | Router outputs route, intent, spans; orchestrator acts on them | the models never see router output; code does |
| 09-25 | Add `cloud_preferred` capability | offline find-nearby should fall back to onboard data, not wait |
| 09-23 | Add `consequence` tag; confirmation is a gate rule, not a route | confirmation depends on the action, not where it runs |
| 09-23 | Intent count 65 incl. AC, recirculation, seat ventilation | climate is the most common in-car request |
| 09-22 | Three policy tables; distraction blocks are REFUSE not DEFER | deferring would auto-run later without being asked |
| 09-22 | Five routes, no CLARIFY in v1; workload is an input | keep v1 comparable to baselines |
| 09-21 | Repo and packages named AutoGate | owner's choice |

## Open questions

- A5 judgment calls to confirm: `pay_bill` as high distraction (PIN and
  payee review need the screen); `cancel_cruise_control` and
  `dip_high_beams` as safety_critical; `adjust_steering_wheel` as
  restricted; `estimate_trip_cost` as needs_small_local; `fuel_prices_nearby`
  as cloud_preferred (stale onboard prices offline); `share_live_location`
  as local_ok.
- Is holding out `fog_lights_on` (headlights_on trains) and `open_fuel_door`
  (open_trunk trains) a fair OOD test? Argued in `docs/taxonomy.md`: same
  family, different goal.

- Keep self-correcting paraphrases that name a sibling intent first? Listed
  in `docs/paraphrase-review-notes.md`.
- Kannada-English spellings need a native speaker's check.
- Strict privacy + sensitive spans on a needs-cloud intent routes LOCAL; should
  it be REFUSE?
- Merge `claude/vibrant-hamilton-1psf5i` (LLM baselines, paper outline,
  Table 4)?
- Create a `main` branch before release.
- Which "ask the driver" threshold to ship (scan in R1 details).
- A6: pairs whose tags differ but whose route never does (set vs cancel
  cruise: comfort vs safety_critical, both always LOCAL) are rejected as
  routing hard negatives. Should a second file accept them as tag-only
  negatives for the actuation head?
- A7 is dropped by its stop rule, but the owner may want to look before
  closing it: the fine-tuned encoder (R6c) is the best test router so far
  (0.867 / 0.308 / 4.8%) and the frozen one (R6a) is the best OOD SWE
  among learned routers. A follow-up would need a new plan, e.g. R5's A6
  data for the fine-tune, or keeping the frozen encoder for out-of-registry
  inputs. Not done here: the plan says stop.
- A7: abstention as tuned on val abstains on a third of safety-critical
  commands. Should safety-critical-looking rows be exempt from abstention
  (answer LOCAL), or is "ask again" acceptable for them?
- A6: synthetic-goal tags are labels nobody else checks; who audits
  `docs/augment-review.md` before R5?

## Where things are

| what | where |
| --- | --- |
| intents and tags | `src/autogate_bench/intents.py`, `docs/taxonomy.md` |
| policy tables and rulebook | `src/autogate_bench/policy.py`, `rulebook.py` |
| seeds and paraphrases | `data/seeds.yaml`, `data/paraphrases/` |
| dataset generation | `python -m autogate_bench.generate` (`--augment data/augment` for A6) |
| decision-encoder router plan (A7) | `docs/plan-a7-decision-router.md` |
| fine-tune on A6 plan (A8) | `docs/plan-a8-finetune-a6.md` |
| decision-encoder router code and frozen configs (A7) | `src/autogate_decide/`, `configs/decide/`; outputs `runs/` (gitignored) |
| augmentation (A6) | `data/augment/`, `src/autogate_bench/augment.py`, `docs/augmentation-spec.md` |
| rules baseline | `python -m autogate_eval.baselines.rules` |
| training, predict, score | `src/autogate_router/`, `src/autogate_eval/score.py`, `docs/training.md` |
| training notebook | `notebooks/kaggle_train.ipynb` |
| run outputs | Colab `/content/runs` or Drive `MyDrive/autogate/runs`; not in git |
