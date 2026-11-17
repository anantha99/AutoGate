# Plan A7: decision-encoder router

A phased plan for an implementing agent. It builds a Jev-style router (one
forward pass, typed answers with probabilities, no text generation) on a small
on-device encoder, and tests whether it can replace the R4 router.

Status: written 2026-11-17. **Phases 0-2 run the same day: Gate 2 outcome B**
(R6a test 0.800 / 0.562 / 9.7%, OOD 1.015 / 21.7%); Phase 2b (R6a2) in
progress. Results, verified facts and every deviation from this plan are
in `docs/experiment-log.md` (A7 and the R6 run details). Code:
`src/autogate_decide/`; frozen configurations: `configs/decide/`.

## 1. What we are testing

**Hypothesis.** A small frozen text encoder (EmbeddingGemma 2, 270M text
tower) plus tiny heads that answer five fixed questions (the four policy tags
and "does this contain sensitive content"), followed by the rulebook, routes
at least as well as R4, in under 600 MB at 8-bit and under 50 ms.

**Why it could work.** R4nc showed the model needs no context: tags are
context-free and the rulebook applies the context. So the model's whole job is
five classifications of a short utterance.

**Why it could fail.**

- One pooled vector may not separate look-alike requests ("open the trunk"
  against "how do I open the trunk"). This is R4's weakest slice already.
- The encoder is unproven on Hinglish and Kannada-English code-mixing.
- Latency on car hardware is unmeasured.

**What this plan does not prove.** Every utterance in the benchmark is
synthetic. Passing every gate shows the approach works on our benchmark, not
on real drivers.

## 2. Facts to verify first

Everything in this table came from web search summaries on 2026-11-17; no
page was opened. Phase 0 replaces each value with a verified one.

| claim | reported value | source to check |
| --- | --- | --- |
| Model id | `google/embeddinggemma-2` | Hugging Face |
| Text tower size | 270M (130M transformer + 140M embedder) | model card |
| Text-only memory | about 191 MB at 4-bit | paper, arXiv 2610.37364 |
| Output | 768-dim, mean pooled, truncatable to 256 | model card |
| Context | 8,192 tokens | model card |
| Licence | Apache 2.0, possibly with the Gemma use policy | repo and card |
| Half precision | float16 overflows to NaN silently | community fine-tune card |
| On-device runtime | MediaPipe Decision Maker, LiteRT bundle | Google AI Edge docs |
| Latency | 60-80 ms on a phone (one third-party test) | not authoritative |

## 3. Rules for the implementing agent

- Read `docs/experiment-log.md` first. Follow `CLAUDE.md`.
- **Labels do not change.** Do not touch intents, tags, seeds, paraphrases or
  policy tables. After generating data, confirm the full set is still sha256
  `8cc12a4b...` and say so if it is not.
- **Reported runs use the dataset without `--augment`**, so they compare with
  R4. `data/augment` is unaudited: do not train on it until the owner's audit
  of `docs/augment-review.md` is done.
- **Test and OOD are scored once per pre-registered configuration.** Fit
  heads on train. Fit temperatures, thresholds and every other choice on val.
  Freeze the configuration (commit it) before scoring test or OOD.
- Every run gets a row in the runs table, including failed ones. Use the ids
  below. R5 stays reserved for the augmented R4 run.
- Embeddings, heads and checkpoints are generated data: gitignored, never
  committed. Record the encoder's exact revision hash in each run's manifest.
- Use fp32 everywhere. Do not enable fp16 autocast.
- Checks before pushing:
  `uv run ruff format src tests && uv run ruff check src tests && uv run pytest -q`.
- Unit tests must run on CPU with no downloads. Provide a deterministic fake
  encoder for tests (same idea as `autogate_router.tiny`).

**Stop and ask the owner when:**

- the model is gated and needs a Hugging Face token or licence acceptance;
- the licence turns out to restrict commercial use or redistribution;
- a gate fails and the plan says "stop";
- hardware is needed for Phase 5;
- R4's prediction files are needed (they live on the owner's Drive under
  `MyDrive/autogate/runs-v1`).

## 4. Reference numbers

All on dataset `8cc12a4b...`, from the experiment log. The reference router
is **R4nc tags** (the lead design).

| metric | R4nc tags | best of any R4 source | fair rules R0f |
| --- | --- | --- | --- |
| test macro-F1 | 0.850 | 0.850 | 0.458 |
| test SWE | 0.342 | 0.342 | 1.224 |
| test critical miss | 6.0% | 5.0% | 30.3% |
| OOD macro-F1 | 0.640 | 0.645 | 0.324 |
| OOD SWE | 1.038 | 0.974 | 1.771 |
| OOD critical miss | 22.9% | 19.1% | 44.3% |
| adversarial slice critical miss | 29.6% | 25% | - |
| OOD distraction dangerous miss | 73.9% | 63.0% | - |
| OOD capability accuracy | 0.52 | 0.52 | - |
| critical miss with abstention (val only) | - | 0.9% at 74% coverage | - |
| int8 route agreement | 0.888 | 0.888 | - |

## 5. Design

```
utterance (no context)
   -> frozen encoder, one pass -> 768-dim vector
   -> five small heads -> p(capability), p(distraction), p(actuation),
                          p(consequence), p(sensitive)
   -> expected-cost routing with the row's context -> route, risk
   -> risk above threshold -> abstain ("ask the driver")
```

**Expected-cost routing.** For each combination of capability, distraction,
actuation and sensitive (96 combinations), take the product of the four
probabilities, get the route from `rulebook.route_tags` for the row's context,
and look up the cost in `configs/cost_matrix.json`. Pick the route with the
lowest expected cost. The expected cost of that route is the risk score used
for abstention. Precompute the route table over the 288 context values.

This replaces the hand-set 0.3 thresholds in `tag_routing.decode_tags`. Report
both decoders so the comparison with R4 stays like for like.

**New package:** `src/autogate_decide/` (add it to the wheel packages in
`pyproject.toml`, and add a `decide` extra for its dependencies).

**Prediction format:** write the existing `predictions.jsonl` format
(`id`, `route`, `tags`, `tag_probs`, `confidence`) so `autogate_eval.score`
works unchanged. Add an optional boolean `sensitive` field; extend the scorer
to use it where it now uses "any predicted span", with tests.

## 6. Phases

### Phase 0: access and facts

1. Load the encoder text-only, embed ten sentences twice, confirm the vectors
   are identical and finite.
2. Read the model card and licence. Fill in the table in section 2 with
   verified values, in the experiment log.
3. Record the task prefix the model card prescribes for classification and
   use it everywhere.
4. Pick a control encoder that is small, ungated and multilingual (for
   example `intfloat/multilingual-e5-small`). It lets the pipeline be built
   before access is sorted and later shows whether results come from the
   approach or from this particular model.

**Gate 0.** The encoder loads and the licence allows our use. If not, stop and
ask; continue the build on the control encoder meanwhile.

### Phase 1: embedding cache and zero-shot baseline (run R6z)

Build:

- `autogate_decide.embed`: reads `rows.jsonl`, embeds each **unique
  utterance** once (22,138 of them), writes vectors plus an index and a
  manifest (encoder id, revision, prefix, dataset sha256).
- `configs/decide/questions.yaml`: one plain-language description per tag
  value, written from `docs/taxonomy.md` and the README. No OOD intent's name
  or key words may appear (reuse the key-word list in `augment.py`). Commit
  this file before any test scoring and never edit it afterwards for R6z.
- `autogate_decide.zeroshot`: cosine similarity between the utterance vector
  and each option vector, softmax with one temperature per question fitted on
  val, then routing as in section 5.

Run R6z on val, then once on test and OOD.

**Gate 1 is informational.** It answers "is prompting alone enough". Expect
it to be far below R4. If it lands within the Gate 2 margins of R4, suspect a
leak and audit before believing it.

### Phase 2: frozen encoder with trained heads (run R6a)

Build:

- `autogate_decide.heads`: one head per question on the cached vectors. Start
  with a linear layer; try one hidden layer of 256 on val. Class weights from
  `autogate_router.data.tag_class_weights`.
- Per-question temperature on val, reusing `autogate_router.calibrate`.
- `autogate_decide.route`: both decoders (conservative thresholds, expected
  cost) and the abstention scan (coverage against critical miss).
- `autogate_decide.predict`: writes predictions for any split.

Run with five training seeds; report mean and spread. Also run the control
encoder through the same code (R6a-ctl).

Diagnostics to report every time:

- adversarial slice: macro-F1 and critical miss;
- per variant (plain, Hinglish, Kannada-English, disfluent, ASR): tag
  accuracy and critical miss, with row counts;
- per tag on OOD: accuracy and dangerous-miss rate;
- per question: ECE before and after calibration;
- sensitive detection: recall and precision;
- **minimal-pair probe:** embed `data/augment/minimal_pairs.yaml` (never
  trained on in this plan) and report the share of pairs where both sides get
  every routing tag right. These rows are existing intents with rulebook
  labels, so the unaudited-tags concern does not apply.

**Gate 2**, on test and OOD, conservative decoder, no abstention:

| outcome | condition | next |
| --- | --- | --- |
| A. matches R4 | test macro-F1 >= 0.83 and test SWE <= 0.37 and test critical miss <= 7.0% and OOD SWE <= 1.12 and OOD critical miss <= 25% | Phase 3 |
| B. short of R4 | anything else | Phase 2b, then Phase 4 |

### Phase 2b: token-state heads (run R6a2), only after outcome B

Keep the encoder frozen but cache its per-token states. Train an
attention-pooling head per question in place of mean pooling, and a BIO head
for sensitive spans (reuse `autogate_router.data.subword_bio`). Re-apply
Gate 2. Outcome A here goes to Phase 3; otherwise go to Phase 4.

Note for Phase 5: this variant needs a runtime that exposes token states.

### Phase 3: intent registry and safety lane (runs R6b, R6l)

**Registry (R6b).** Nearest-neighbour match of the utterance vector against
registered example phrases; a match takes its tags from the intent table.

- Fair mode: the registry holds train-split seeds only. Fall back to the
  heads when the best similarity is below a threshold. Choose the threshold
  by holding out a tag-balanced set of training intents (as `splits.py` does
  for OOD), never by looking at OOD.
- Registered mode: add the five seeds of each OOD intent to the registry with
  no retraining, and score on OOD rows that are not those seeds. This is a
  **new protocol**: report it in its own table. It is not comparable with
  R4's OOD numbers; its rules analogue is R0d.

**Safety lane (R6l).** A keyword or phrase match for the safety-critical
intents that sets actuation to safety_critical. Build it from train seeds
only. Required on val and test: zero triggers on rows whose true route is
REFUSE. Report its recall on safety-critical rows.

Report every later result both with and without the registry and the lane.

### Phase 4: fine-tune the encoder (run R6c), only if Gate 2 never reached A

Train the text tower with the same heads end to end on a Colab T4 (add a cell
to the notebook). fp32, three epochs, LoRA rank 16 first, full fine-tune
second. Select on val SWE. Re-apply Gate 2.

**If Phase 4 also fails Gate 2: stop.** Log the approach as dropped with the
numbers, and return to R5.

### Phase 5: device reality (run R6q)

1. Obtain or build the 8-bit encoder artifact (LiteRT bundle, or ONNX int8,
   or GGUF Q8). If the encoder was not fine-tuned, use Google's shipped
   quantized file unchanged.
2. **Re-embed the dataset with that exact artifact and retrain the heads on
   those vectors.** This removes any gap between training and serving.
3. Compare against the fp32 run: route agreement, and every Gate 2 metric.
4. Measure latency at batch size 1 over 1,000 val utterances: p50 and p95,
   one thread and four threads, on the development CPU and on a reference
   device the owner supplies. State clearly that neither is a head unit.
5. Write a minimal loader that runs the artifact plus heads plus rulebook end
   to end from an utterance and a context, with a test.

## 7. Verdict: does it work for our use case

Evaluate the best configuration, with abstention, registry and lane as
shipped. All thresholds are chosen on val and applied unchanged.

**The thresholds marked "proposed" are judgment calls. The owner confirms or
changes them before the first test scoring in Phase 2.**

| id | condition | threshold | origin |
| --- | --- | --- | --- |
| U1 | test critical miss on answered rows | <= 1.0% at coverage >= 85% | PRD; coverage proposed |
| U2 | test macro-F1 and SWE on answered rows | > 0.85 and <= 0.49 (40% of R0f) | PRD |
| U3a | OOD, unregistered: critical miss | <= 5% at coverage >= 50% | proposed |
| U3b | OOD, registered with five seeds: critical miss | <= 2% at coverage >= 80% | proposed |
| U4 | safety-critical rows abstained on; lane triggers on REFUSE rows | <= 1%; zero | proposed |
| U5 | 8-bit: route agreement, size, p95 latency | >= 0.99; <= 600 MB; <= 50 ms | PRD; agreement from the log |
| U6 | any language variant's critical miss | <= 2x the plain variant's | proposed |
| U7 | sensitive detection recall at the operating point | >= 0.95 | proposed |

How to read the result:

- **All pass:** adopt as the router; R4 becomes the comparison baseline.
- **U1 to U4, U6, U7 pass, U5 latency between 50 and 100 ms:** adopt, with
  the note that it needs faster hardware or decisions on partial transcripts.
- **Matches R4 (Gate 2 outcome A) but fails U1:** it is a smaller, exportable
  R4, not a release-ready router. Report the coverage against critical-miss
  curve and keep working on the safety gaps.
- **Fails Gate 2 after Phase 4:** drop it.

## 8. Run ids and what to log

| id | what |
| --- | --- |
| R6z | zero-shot, option descriptions only |
| R6a, R6a-ctl | frozen encoder with heads; control encoder |
| R6a2 | frozen encoder, token-state heads |
| R6b | with intent registry (fair and registered modes) |
| R6l | with safety lane |
| R6c | fine-tuned encoder |
| R6q | 8-bit artifact, latency |

For each run: a runs-table row for test and OOD, a "Run details" entry with
the config, commit, encoder revision and what it showed, and a decision-log
line whenever a gate is passed or failed.

## 9. Open items for the owner

- Confirm or change the proposed thresholds in section 7.
- Hugging Face access to the model, if gated.
- A reference device for Phase 5.
- Copies of R4's prediction files, for row-level comparisons.
- The pending audit of `docs/augment-review.md` (needed before any run may
  train on the augmentation data).
