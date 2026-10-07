# Paper outline (working draft)

Mapped out by Socratic dialogue. Decisions already made are stated as
decisions; anything still open is under "Open decisions" at the end. Numbers
marked *[to be shown]* do not exist yet.

## Positioning

- **Contribution type:** the router is the product; the benchmark is the
  evidence that it is safe. Both are released.
- **Venues:** ML and automotive (e.g. EMNLP/ACL Industry Track, IEEE IV,
  AutoUI).
- **Deployment picture:** an OEM keeps its own on-device model and its own
  cloud model. Speech goes through the OEM's ASR, AutoGate reads the text and
  the vehicle context, and decides which endpoint, if any, gets the request.
  AutoGate is model-agnostic: `LOCAL` and `CLOUD` name endpoints, not models.
- **Scope:** text only, sitting after ASR. ASR errors are simulated
  (`autogate_bench.asr_noise`), not modelled end to end.
- **Standards:** we claim *alignment* with automotive safety, distraction and
  privacy standards, not compliance. Everything is open source so anyone can
  audit the alignment.
- **Default policy, adopt or override:** AutoGate ships a default policy
  an OEM can adopt as is or override. Tables 1–3 (driving demand,
  distraction, actuation) and the privacy switch carry the standards
  alignment. Table 4 (capability) is about the car's own model, and its
  default is conservative: with a tiny (≤2B) model, only vehicle commands,
  short answers, media control and templated read-outs stay local. Some
  rules are locked and no policy file can override them. *Companies can tune
  what is convenient, but they cannot configure away what is safe.*
- **Tone:** an honest first attempt at a component the next generation of
  assistants will need. Negative results and limitations are reported up
  front, not tucked into an appendix.

## Thesis

> In-car LLM assistants need a gate that decides, per request and per driving
> context, whether to answer on the head unit, in the cloud, in the cloud
> with personal data masked, later, or not at all. We release AutoGate, a
> 0.6B on-device router, and AutoGateBench, a 53k-row benchmark scored with a
> safety-weighted cost matrix. A small, context-aware router whose
> safety-critical behaviour stays deterministic and whose policy is swappable
> data can match frontier zero-shot LLMs on safety-weighted error, run on
> head-unit hardware, and act as a privacy gate, which a cloud model cannot
> do without first receiving the data.

## Core arguments

| # | Claim | Evidence | Status |
| --- | --- | --- | --- |
| C1 | **The problem.** In-car routing is a five-way, context-dependent, safety-critical decision. A router that sees only the text cannot solve it. | 9,673 of 17,728 utterances (55%) change route with vehicle state; no-context ablation | dataset done; ablation *[to be shown]* |
| C2 | **The architecture.** Splitting the router into learned perception (intent, sensitive spans) and declarative policy keeps safety-critical behaviour deterministic and lets an OEM swap policy, including its capability list, without retraining, at little cost in accuracy compared with an end-to-end route head. Locked rules mean no policy file can configure away safety-critical or restricted-actuation behaviour. | E2E vs. decomposed comparison; policy-variant experiment (RQ4); locked rules enforced by `Policy.validate` and tested | locks done; comparison *[to be shown]* |
| C3 | **The results.** A 0.6B router matches or beats frontier zero-shot LLMs (Claude and non-Claude) on safety-weighted error and critical misses, at a fraction of the size, on-device, and is the only option in the comparison that can gate privacy before data leaves the car. | baseline tiers 1–5; latency on target hardware | *[to be shown]* |
| C4 | **The benchmark.** AutoGateBench and its cost matrix expose failures that accuracy hides: rules reach 0.77 test accuracy yet miss 22% of critical cases, and fall to 0.08 recall on CLOUD_MASKED. | `docs/dataset-v0.md` | done (rules); router *[to be shown]* |

## Standards alignment

Each standard maps to something measured or to a design property. None of
it is a certification claim.

| Standard | Concern | How the paper addresses it |
| --- | --- | --- |
| ISO 26262 | ML cannot realistically be certified to an ASIL level | Decomposition: safety-critical commands take a deterministic branch the model cannot override (architectural argument) |
| ISO 21448 (SOTIF) | Performance shortfalls of an ML component that works as built, including unknown scenarios | Critical-miss rate on test vs. the OOD split ("unknown unsafe") |
| NHTSA Visual-Manual Distraction Guidelines | No screen-heavy tasks while moving | Distraction gate (Table 2); REFUSE recall while moving |
| India DPDP Act 2023, GDPR | Personal data leaving the vehicle | **Leak rate**: sensitive rows routed to plain `CLOUD` |
| EU AI Act | Transparency, oversight | Rulebook returns the branch that fired; policy is inspectable JSON |

**Headline metrics:** critical-miss rate (safety) and leak rate (privacy),
with SWE as the aggregate. Accuracy, macro-F1 and MCC go in the full tables.
A cost-matrix sensitivity analysis shows the ranking does not hinge on the
hand-set weights.

## Experiments

**Baseline tiers**

1. Rules: keyword intent, regex PII, rulebook (exists)
2. Same-size on-device LLM zero-shot: Qwen3-0.6B prompted with the policy tables
3. Larger on-device LLM zero-shot (~3–4B)
4. Frontier cloud LLMs zero-shot, one Claude and at least one non-Claude.
   This is an upper bound that could not be deployed as the gate, because
   asking it already sends the data off the car.
5. AutoGate: end-to-end route head and perception + rulebook, each with and
   without context

**Research questions**

- RQ1: Does the router beat the zero-shot tiers on critical misses, leak rate and SWE?
- RQ2: End-to-end vs. perception + rulebook: which is safer, and which is more accurate?
- RQ3: How much does context buy? (no-context ablation, per slice)
- RQ4: Policy transfer. Train on the default policy and relabel the test
  and OOD splits under each variant (`generate --policy`):
  - **default**: the headline numbers;
  - **tiny local model**: Table 4 narrowed, e.g. `needs_small_local` to
    cloud and templated read-outs to `needs_small_local`;
  - **strong local model**: Table 4 widened, e.g. `needs_small_local` and
    `general_knowledge_question` to `local_ok`;
  - **safety-table change**: e.g. a stricter distraction gate (Table 2)
    or restricted actuation allowed at low demand (Table 3).

  Compare the decomposed router with the swapped policy (no retraining)
  against the E2E router as trained and retrained on relabelled data.
  Report the default as the headline and the variants as transfer.
- RQ5: Robustness by slice and variant: Hinglish, Kannada-English,
  disfluent, ASR, adversarial, OOD intents.
- RQ6: Deployment: parameters, ONNX latency on the target hardware, calibration (ECE).
- Cost-matrix sensitivity: perturb the weights and report how stable the ranking is.

## Section outline

1. **Introduction.** Open with a concrete scene: at 80 km/h the driver says
   "read me Priya's last message, the OTP is in it". Each route would be
   wrong somewhere. Then the gap: routers built for cost or quality ignore
   safety and context, and guardrails ignore where the request runs. End
   with contributions C1–C4.
2. **Background and related work.** LLM routers and cascades (RouteLLM,
   FrugalGPT, Hybrid LLM); on-device/cloud hybrid inference; content-safety
   guardrails (Llama Guard, NeMo Guardrails); in-car dialogue data (KVRET);
   driver-distraction research and guidelines; PII detection and masking.
   *(Verify every citation before submission.)*
3. **Problem formulation.** Routes; context (speed, workload, passenger,
   connectivity, privacy mode); the three policy tables; the precedence
   tree; the cost matrix; definitions of critical miss and leak rate; the
   standards-alignment table.
4. **AutoGateBench.** Taxonomy and seeds; paraphrases (plain, Hinglish,
   Kannada-English, disfluent) and ASR noise; span injection; context
   sampling; rulebook labelling; seed-level and OOD splits; slices; counts;
   the rules baseline as evidence the benchmark is not too easy.
5. **AutoGate router.** Qwen3-0.6B + LoRA; route, intent and span heads; the
   two decision modes; the deterministic safety bypass; calibration; ONNX
   export.
6. **Experiments.** Setup, baselines, metrics, then RQ1–RQ6 and sensitivity.
7. **Discussion.** What an OEM plugs in (endpoints, policy, capability
   table) and what it cannot change (locked rules); when to prefer each
   decision mode; failure analysis.
8. **Limitations and ethics.** See below. Written plainly.
9. **Conclusion.**

Appendix: full policy tables, intent taxonomy, cost matrix, prompts for the
zero-shot baselines, per-slice and per-variant tables, paraphrase spec.

## Limitations to state plainly

- **Labels encode one policy.** The benchmark measures how faithfully a
  router follows a policy, not real-world safety. There is no user study.
- **Single-family synthetic paraphrases.** Claude agents wrote them all, so
  train and test share a style. Mitigations: a non-Claude frontier baseline
  and KVRET real utterances.
- Kannada-English lines have not been checked by a native speaker.
- The sensitive rate is 16%, below the 35% target.
- The cost matrix was set by hand (hence the sensitivity analysis).
- Text only; ASR errors are simulated.
- Alignment with standards, not compliance.

## Open decisions

1. ~~**Capability table.**~~ **Resolved.** Capability is Table 4 of
   `Policy` (`DEFAULT_CAPABILITY`), its default is conservative, and an OEM
   file lists only the intents it widens. Locked in `Policy.validate`:
   vehicle commands are always `local_ok` and safety-critical actuation is
   never blocked, so the capability list cannot move safety-critical or
   restricted behaviour. The default reproduces the earlier labels
   exactly; `rows.jsonl` sha256 is unchanged.
   Still to do: ship the RQ4 variant policies as JSON files under `configs/`.
2. **Circularity.** The labels come from the rulebook, so perception +
   rulebook reproduces them exactly when perception is perfect. How does the
   paper keep this from reading as a benchmark rigged for the decomposed
   design?
3. **Latency target:** which SoC or CPU, and what budget?
4. **Which frontier models** in tier 4, and which ~3–4B model in tier 3.
5. **KVRET real utterances:** in v1 or listed as future work?
6. **Sensitive rate:** add slot-bearing seeds, or raise `--n-sensitive`?
7. **If the router does not beat tier 4:** the paper still stands on
   privacy, latency and policy transfer, but the abstract changes. Decide
   on the fallback wording now.
