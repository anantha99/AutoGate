# AutoGateBench v0: the paraphrased dataset

Generated from the 281 seeds plus 11,240 hand-written paraphrases
(`data/paraphrases/`, 40 per seed) and 4 deterministic ASR variants per seed.
Same pipeline as the seed-only pilot (`docs/pilot.md`); only the input set
differs. Generated data is not committed; regenerate with:

```
uv run python -m autogate_bench.generate --out data/generated/full --rng-seed 0 --paraphrases data/paraphrases
uv run python -m autogate_eval.baselines.rules --rows data/generated/full/rows.jsonl
```

Deterministic: `rows.jsonl` sha256 is
`2d9583cf5bd7ebd8132b8bfe3ea2036911054cba84beeedd3e464790fd5e204f`
at paraphrase commit time. Every row was independently re-checked: the route
and reason recompute from the rulebook, every span offset slices back to its
text, every paraphrase inherits its seed's split, and the six OOD intents
appear only in the `ood` split.

## Counts

| | count |
| --- | --- |
| rows | 53,184 |
| utterances | 17,728 (394 seed fills, 15,760 paraphrase fills, 1,574 ASR) |
| train / val / test / ood | 38,469 / 5,130 / 4,860 / 4,725 |
| seeds per split | 204 / 26 / 26 / 25 |

| route | rows |
| --- | --- |
| LOCAL | 32,881 |
| REFUSE | 6,753 |
| DEFER | 5,966 |
| CLOUD | 5,765 |
| CLOUD_MASKED | 1,819 |

| variant | utterances |
| --- | --- |
| plain | 9,850 |
| hinglish | 2,364 |
| disfluent | 2,364 |
| kannada_english | 1,576 |
| asr | 1,574 |

Slices (rows): context_invariant 24,165; context_flipped 29,019; sensitive
8,640; adversarial 3,210. Sensitive rate 16.2%. Utterances: 8,055 invariant
(all LOCAL) and 9,673 flipped.

## Rules baseline

Keyword intent matcher built from the seeds, regex PII detector, then the
rulebook. On seed-only data this scored 0.99 intent accuracy and 0.81
macro-F1 (`docs/pilot.md`). On the paraphrased data:

- intent accuracy 0.521 (train 0.522, val 0.485, test 0.533, ood 0.535)
- PII detection recall 0.192, precision 1.000

| subset | n | accuracy | SWE [95% CI] | macro-F1 [95% CI] | MCC | critical miss |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 53184 | 0.788 | 0.805 [0.789, 0.820] | 0.610 [0.604, 0.616] | 0.610 | 0.246 (2359/9588) |
| split: train | 38469 | 0.805 | 0.735 [0.717, 0.753] | 0.604 [0.597, 0.611] | 0.614 | 0.238 (1668/7010) |
| split: val | 5130 | 0.742 | 1.005 [0.947, 1.060] | 0.549 [0.535, 0.562] | 0.545 | 0.216 (150/693) |
| split: test | 4860 | 0.770 | 0.919 [0.863, 0.974] | 0.627 [0.609, 0.645] | 0.588 | 0.225 (197/876) |
| split: ood | 4725 | 0.717 | 1.044 [0.990, 1.099] | 0.583 [0.572, 0.593] | 0.628 | 0.341 (344/1009) |
| slice: context_invariant | 24165 | 0.919 | 0.249 [0.236, 0.262] | 0.192 [0.191, 0.192] | 0.000 | 0.043 (121/2835) |
| slice: context_flipped | 29019 | 0.679 | 1.268 [1.241, 1.293] | 0.595 [0.589, 0.601] | 0.593 | 0.331 (2238/6753) |
| slice: sensitive | 8640 | 0.662 | 1.442 [1.391, 1.493] | 0.464 [0.454, 0.474] | 0.424 | 0.373 (165/442) |
| slice: adversarial | 3210 | 0.674 | 2.078 [1.961, 2.178] | 0.321 [0.312, 0.331] | 0.437 | 0.358 (982/2741) |

| route | precision | recall | F1 | support | predicted |
| --- | --- | --- | --- | --- | --- |
| LOCAL | 0.810 | 0.916 | 0.860 | 32881 | 37153 |
| CLOUD | 0.578 | 0.603 | 0.591 | 5765 | 6016 |
| CLOUD_MASKED | 0.863 | 0.083 | 0.151 | 1819 | 175 |
| DEFER | 0.863 | 0.627 | 0.726 | 5966 | 4335 |
| REFUSE | 0.803 | 0.654 | 0.721 | 6753 | 5505 |

| true \ pred | LOCAL | CLOUD | CLOUD_MASKED | DEFER | REFUSE |
| --- | --- | --- | --- | --- | --- |
| LOCAL | 30108 | 1501 | 24 | 497 | 751 |
| CLOUD | 2124 | 3479 | 0 | 0 | 162 |
| CLOUD_MASKED | 829 | 806 | 151 | 0 | 33 |
| DEFER | 2084 | 0 | 0 | 3741 | 141 |
| REFUSE | 2008 | 230 | 0 | 97 | 4418 |

## What this says

- **The benchmark is not too easy.** The PRD's week-1 check was "if rules
  score above 0.9 macro-F1, rebalance before scaling". Rules score 0.63
  macro-F1 on test and miss 22% of critical cases, so the paraphrases did
  their job. No rebalancing is needed before training.
- **Intent recognition is where rules fail**, exactly as the PRD predicted:
  0.99 on canonical phrasings, 0.52 on paraphrases. The router's gain, if any,
  will come from there.
- **CLOUD_MASKED is the hardest class for rules** (recall 0.08) because the
  regex detector cannot see lowercase names. This is the span head's job.
- **The adversarial slice is now real**: SWE 2.08 and macro-F1 0.32 for rules,
  versus a perfect score on seed-only data.
- **The context-invariant slice is all LOCAL**, so its macro-F1 is degenerate
  (0.19) and its MCC is 0. Report accuracy or SWE on that slice, not F1.

## Known gaps

- Sensitive rate is 16%, below the PRD's 35% target, because only 32 seeds
  carry sensitive slots and paraphrases preserve slot sets. Options: add
  slot-bearing seeds, or raise `--n-sensitive`.
- All paraphrases come from one model family (written by Claude agents
  against `docs/paraphrase-spec.md`). Train and test share that style. The
  KVRET real-utterance set in the task list is the antidote and should be
  added before the paper's claims are final.
- Kannada-English spellings were written by the agents and have not been
  checked by a native speaker. See `docs/paraphrase-review-notes.md`.
