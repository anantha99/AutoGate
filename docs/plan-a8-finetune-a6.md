# Plan A8: fine-tune the decision encoder on the A6 data

A phased plan for an implementing agent. It follows A7
(`docs/plan-a7-decision-router.md`), which was dropped by its own stop rule,
and reuses A7's code (`src/autogate_decide/`).

Status: **Phase 0 prepared, nothing trained.** Written 2026-11-18 and
revised the same day after an independent review. **Blocked** on the
owner's audit of `docs/augment-review.md` (A6 data may not be trained on
before it). The owner chose CPU (2026-11-18): once the audit is done,
`scripts/a8_phase1.sh` runs Phase 1 end to end and stops where this plan
says stop.

Prepared (committed before any A8 run): `configs/decide/a8_holdout.yaml`
(proxy pick with seed 0: read_document, send_text_message, emergency_call,
disable_lane_assist, play_video, book_restaurant, fuel_prices_nearby,
adjust_window; key words; 59 probe pairs), `configs/decide/a8_beta.json`
(beta 0.0276), `autogate_decide.holdout`, `autogate_decide.a8` (the
selection rules) and the fine-tune options, with tests.

## 1. What we are testing

**Where A7 ended.** No A7 configuration passed Gate 2 on test and OOD
together:

| run | test macro-F1 / SWE / critical miss | OOD SWE / critical miss |
| --- | --- | --- |
| R4nc tags (reference) | 0.850 / 0.342 / 6.0% | 1.038 / 22.9% |
| R6a frozen EmbeddingGemma 2 + heads | 0.800 / 0.562 / 9.7% | **1.015 / 21.7%** |
| R6c fully fine-tuned (A7 recipe) | **0.867 / 0.308 / 4.8%** | 1.166 / 30.5% |

Fine-tuning fixed known intents and broke unseen ones. On OOD, R6c misses
88% of high-distraction rows (R6a 80%), and its capability and distraction
probabilities are confidently wrong (ECE 0.34 and 0.32).

**Hypothesis.** The OOD collapse comes from training on too few distinct
goals per tag value (only 7 training intents are high distraction). A6 adds
60 synthetic goals (20 of them high distraction) and 290 minimal pairs.
Fine-tuning on that data, with or without a pull towards the frozen
encoder, keeps R6c's test results and brings unseen intents back to at
least the frozen encoder's level, which Gate 2 needs.

**Why it could fail.**

- 60 synthetic goals written by agents may be too uniform to teach
  "distraction" as a property rather than a list of phrasings.
- The A6 labels are judgement calls that nobody but the owner checks.
- The collapse may come from supervised fine-tuning itself, not from the
  data. That is why there is an anti-drift variant (R7b) and two controls
  (section 5).
- Selection cannot see the real OOD set. The proxy holdout (section 4) is a
  stand-in, and Phase 1 checks it reproduces A7's known ordering before
  any comparison counts.

**What it does not prove.** As in A7: synthetic benchmark, not real
drivers. It also does not show whether A6 helps routers in general. R5
(R4's Qwen router on the same data) answers that, and A8 reports beside it
when R5 exists. **R5 trains on all 290 pairs, A8 on 232 of them** (section
4): say so wherever the two appear together.

## 2. Rules for the implementing agent

Everything in section 3 of the A7 plan still applies, in particular:

- **Labels do not change.** The augmented dataset is
  `generate --augment data/augment`. Check that the first 66,414 lines are
  byte-identical to the sha256 `8cc12a4b...` file, and record the augmented
  file's sha256 in every manifest.
- **The audit comes first.** Do not train on `data/augment` until the
  owner has audited `docs/augment-review.md`. Use any corrections the audit
  makes; regenerate and re-record the sha256.
- **Pre-register, then score once.** Fit on train. Select on the
  selection sets of section 4. Commit the frozen configuration, then score
  test and OOD **once**.
- **Test reuse.** A7 used the test and OOD sets three times (R6a, R6a2,
  R6c). A8 adds exactly one more use, for the single final configuration.
  State that count next to the result. Nothing is tuned after it.
- fp32 everywhere. CPU or Colab T4; record the torch version, device and
  thread count. Checkpoints and caches are gitignored.
- Use the A7 code paths: `autogate_decide.finetune`, `embed`,
  `run --selection val`, and the per-epoch resume and per-trial selection
  checkpoints (container restarts were hourly in A7).
- Every run gets a runs-table row in `docs/experiment-log.md`, including
  failed ones. Run ids: **R7x** (R5 stays reserved for R4 on the A6 data).

**Stop and ask the owner when:**

- the audit is not done, or its corrections change more than a handful of
  labels (that is a new A6; redo the section 4 lists);
- the proxy holdout fails its validity check (Phase 1, step 2);
- no configuration passes the val filter (outcome C);
- Phase 4 needs a quantised artefact of a fine-tuned encoder and neither
  route in section 6 works.

## 3. Reference numbers

Use the A7 table above. Gate 2 and the verdict thresholds are unchanged
from the A7 plan; the owner confirmed the "proposed" ones on 2026-11-17.

## 4. Held-out data for selection

A7's selection saw only val, which holds training intents, so it could not
see the OOD collapse. A8 carves two held-out sets out of the training
intents. Neither is trained on during selection.

**Proxy holdout (8 training intents).** For each of the 8 real OOD intents,
take one training intent with the same capability, distraction and
actuation tags. Every such tuple has at least 3 training intents (checked
2026-11-18). So the proxy has the real OOD set's composition: two
high-distraction intents (needs_cloud and local_ok, both actuation none),
one medium, one safety-critical, one restricted, one comfort, one
cloud_preferred and one needs_small_local/low. Five of the 7
high-distraction training intents stay in training. Among candidates, pick
with a fixed seed.

During selection runs, remove from training and from val:

- the proxy intents' seeds, paraphrases and ASR copies (train and val
  rows);
- A6 minimal pairs that use a proxy intent;
- synthetic goals that would leak, under the same guards `augment.py`
  applies to the real OOD set: token Jaccard > 0.5 against any proxy seed
  or paraphrase, or a proxy intent's key words in the goal's name or
  description (write the key words beside the list, as `OOD_KEYWORDS`
  does).

The **proxy set** is the proxy intents' train and val rows. Their test rows
are never used in selection, so the test split is touched only once, by
the final model. Write the
pick, the key words and the dropped goals and pairs to
`configs/decide/a8_proxy_holdout.yaml`, with a test that checks the tuple
match and the leak guards. Commit it before any run.

**Probe pairs (58 of the 290 minimal pairs).** Hold them out in every
phase, stratified by `differs_in`. Take whole intent pairs, and only from
intent pairs that touch no proxy intent. Probe scores are **report-only**:
they never enter selection. They replace A7's minimal-pair probe, which A6
training would contaminate. Commit the list in
`configs/decide/a8_probe_pairs.yaml`.

Selection scores, per epoch and per configuration:

- **val minus proxy rows** (training intents, new phrasings):
  conservative-decoder SWE and critical miss, as in A7;
- **proxy set** (unseen intents): SWE, critical miss, distraction
  dangerous-miss rate, capability accuracy;
- **probe pairs** (report-only): share with every routing tag right on
  both sides.

## 5. Configurations

Fine-tune recipe from R6c unless stated: EmbeddingGemma 2 text tower
`f9972ac2`, full fine-tune at lr 2e-5, mean pooling, heads for the four
tags, sensitive and joint; loss weights as R6c; batch 32; heads lr 1e-3;
warmup 6%; 3 epochs.

| id | what | role |
| --- | --- | --- |
| R7-frz | frozen encoder, heads trained through the same fine-tune path (`--freeze-encoder`, vectors from the frozen cache), A6 data, proxy held out | the unseen-intent level Gate 2 needs; scored by the same decoder as the others (minutes) |
| R7-ctl | R6c recipe, **no** A6, proxy held out | where R6c already stood on the proxy |
| R7a | R6c recipe + A6 | the data effect alone |
| R7b | R6c recipe + A6 + distillation | data plus a pull towards the frozen geometry |

LoRA is cut: it lost to the full fine-tune on val in A7 (0.408 vs 0.333).
The compute goes on second fine-tune seeds instead (Phase 1, step 5).

**Distillation (R7b).** Add `beta * (1 - cos(z_ft, z_frozen))` to the loss,
where `z_frozen` is the frozen encoder's pooled vector for the same text
(in the A7 embedding cache). Set beta before any run, without labels:

1. Compute R6c's drift from the two existing caches: the mean
   `1 - cos(z_R6c, z_frozen)` on the training texts.
2. Choose beta so that the term equals 25% of R6c's final train loss at
   that drift.

Record both numbers in the configuration. Beta is not tuned afterwards.

**Synthetic goals in the loss.** They have no intent and train the tag,
sensitive and joint heads only, as in A6. The joint vocabulary is
rebuilt from train plus goals, because goals add combinations no intent
has. Goal rows count once per utterance, like every other row.

## 6. Phases

### Phase 0: prerequisites

1. Owner audit of `docs/augment-review.md` done; corrections applied;
   augmented file regenerated; prefix and sha256 checks pass.
2. Proxy holdout and probe-pair lists committed, with their tests.
3. `finetune.py` takes the augmented rows, the proxy list (removed from
   train and val), the probe list and `--distill-weight`. It reports the
   section 4 scores per epoch. Tests on the tiny backbone; no downloads.
4. Beta for R7b computed and committed (section 5).

### Phase 1: controls, proxy check, selection (R7-frz, R7-ctl, R7a, R7b)

1. Run R7-frz and R7-ctl.
2. **Proxy validity check.** The proxy must reproduce what A7 saw on the
   real OOD set: R7-frz better than R7-ctl on proxy critical miss and on
   distraction dangerous miss, each by a clear margin (paired intent
   bootstrap, step 4). If it does not, the proxy is not a stand-in: stop
   and ask the owner.
3. Run R7a and R7b.
4. **Selection rule (pre-registered).**
   - Each fine-tuned run's epoch is chosen by the same rule. R7-ctl is
     treated the same way, so the comparison is best-epoch against
     best-epoch.
   - **Val filter:** val SWE (proxy rows removed) no worse than R7-ctl's
     best epoch + 0.03, and val critical miss no worse than R7-ctl's +1
     point.
   - Among epochs that pass, take the lowest proxy critical miss. Break
     ties (within 1 point) by the lower proxy SWE.
   - Between R7a and R7b, the same rule picks the winner.
5. Run a **second fine-tune seed** of R7-ctl and of the winner, with the
   same epochs. Outcome A is checked on both seeds.

**Outcomes.** All comparisons on the proxy set use a paired bootstrap of
(winner minus control), resampling **intents**. The proxy's eligible rows
come from about five intents, so row-level intervals would be far too
narrow.

| outcome | condition | next |
| --- | --- | --- |
| A | On both seeds: the winner beats R7-ctl on proxy critical miss and proxy SWE (95% interval of the difference excludes 0), and is no worse than R7-frz on either | Phase 2 |
| B | a winner passes the val filter but misses A | stop: A6 does not fix the collapse for this encoder. Log it; R5 decides whether A6 helps at all |
| C | nothing passes the val filter | stop and ask the owner |

### Phase 2: final model (R7)

1. Retrain the winner on **all** training data: the proxy intents return,
   the probe pairs stay out. Use the selected epochs and seed 0, with no
   further selection.
2. **Stop check:** the final run's val scores on non-proxy rows must be
   within the val-filter tolerances of the selection run's. If not, stop
   without scoring test.
3. Re-embed the dataset with the merged encoder. Check that the five probe
   sentences reproduce their training-time vectors.
4. `run --method heads --selection val` on the augmented rows: linear
   heads, weight decay by val NLL, lambda 0.5, temperatures and abstention
   threshold on val. Commit the frozen configuration.
5. Score test and OOD once. Apply Gate 2 (A7 plan) to the conservative
   decoder without abstention, and report expected cost beside it. Also
   report:
   - the probe-pair score;
   - the per-variant table;
   - OOD per intent;
   - the intent-bootstrap intervals, as in A7;
   - the test-use count (4).

**If Gate 2 fails: stop.** Log A8 as dropped with the numbers. A third
fine-tuning plan would need new data, not new knobs.

### Phase 3: registry and safety lane (only after Gate 2 passes)

As A7 Phase 3 (code exists, tested, never run): fair and registered
registry modes, safety lane. Report with and without them.

### Phase 4: device reality (only after Gate 2 passes)

As A7 Phase 5, with one change. Google's shipped int4 LiteRT file holds
the **base** weights, so a fine-tuned encoder needs its own quantised
artefact. Try, in order:

1. a LiteRT-LM bundle built from the merged weights (`litert-lm-builder`
   is in the `litert-lm` 0.19.2 package; check that it supports
   EmbeddingGemma 2);
2. ONNX with weight-only per-channel int8. Do not use dynamic quantisation:
   it broke R1, R2 and R4.

Then re-embed with the artefact, retrain the heads on those vectors,
require route agreement >= 0.99 with fp32, and measure size and latency.

## 7. Verdict

Use the A7 plan's section 7 thresholds unchanged (U1-U7, owner-confirmed).
How to read the result:

- **Gate 2 passes and U1-U7 pass:** adopt; R4 becomes the comparison.
- **Gate 2 passes, U4 or U6 fails:** a better router than R4 that is not
  release-ready. Name the failing rows; the next work is data for them.
- **Gate 2 passes but Phase 4 cannot produce a quantised artefact with
  agreement >= 0.99:** the result stands for fp32 only. Phase 4 becomes
  its own plan.

## 8. Run ids and compute

| id | what |
| --- | --- |
| R7-frz, R7-ctl | controls on the proxy holdout (R7-ctl in two seeds) |
| R7a, R7b | selection runs (winner in two seeds) |
| R7 | final model, scored once on test and OOD |
| R7q | quantised artefact and latency (Phase 4) |

Compute on this 4-core CPU, from A7's timings: a full fine-tune is about
26 minutes an epoch, so about 80 minutes a run. Phase 1 is five fine-tune
runs (R7-ctl and the winner twice each, the loser once), about 7 hours.
Phase 2 adds about 1.5 hours. A Colab T4 is much faster.

For each run: a runs-table row, and a "Run details" entry with config,
commit, encoder revision, augmented-data sha256 and device. Add a
decision-log line at each outcome.

## 9. Open items for the owner

- **The A6 audit** (blocking).
- CPU (about 9 hours in total, run here) or Colab T4.
- Whether a fourth use of the test set is acceptable, or a fresh test set
  should be generated first. That would mean new paraphrases of the
  test seeds, which changes no labels but costs a paraphrase round.
