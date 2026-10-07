# Results v0: first router run

Qwen3-0.6B + LoRA (r=16), three heads, 3 epochs, batch 32, on
AutoGateBench v0 (`docs/dataset-v0.md`). Trained on a Colab T4 in 1h43m with
`notebooks/kaggle_train.ipynb`. Rules numbers come from
`python -m autogate_eval.baselines.rules`; the "train-only" column builds the
keyword matcher from train seeds only, which is the fair comparison because
the router never saw val, test or OOD seeds either.

## Headline table (held-out test split, 4,860 rows)

| metric | rules (all seeds) | rules (train-only) | router | PRD target |
| --- | --- | --- | --- | --- |
| macro-F1 | 0.627 | 0.430 | **0.780** | > 0.85 |
| safety-weighted error | 0.919 | 1.256 | **0.426** | < 40% of rules |
| critical miss rate | 22.5% | 37.1% | **7.1%** | < 1% |
| accuracy | 0.770 | 0.655 | **0.857** | |
| intent accuracy | 0.533 | 0.233 | 0.503 | |

Router SWE is 46% of the all-seeds rules baseline and 34% of the fair
train-only baseline. Against the fair baseline the SWE target is met; macro-F1
and critical miss are not.

## Out-of-distribution split (6 held-out intents, 4,725 rows)

| metric | rules (all seeds)* | rules (train-only) | router |
| --- | --- | --- | --- |
| macro-F1 | 0.583 | 0.231 | 0.442 |
| safety-weighted error | 1.044 | 2.231 | 1.312 |
| critical miss rate | 34.1% | 78.6% | 25.7% |
| intent accuracy | 0.535 | 0.000 | 0.000 |

\* The all-seeds matcher had the OOD intents' own seeds, so this column is
not a fair OOD number; it is kept because `docs/dataset-v0.md` reports it.

Intent accuracy on OOD is 0.000 by construction for both learned and
train-only systems: those six classes have no training examples.

## Spans and masking (test + ood)

| | rules | router |
| --- | --- | --- |
| span F1 (exact match) | 0.280 | 0.632 |
| span recall | 0.167 | 0.682 |
| leak rate on cloud-routed rows | 85.7% | 33.4% |

## Calibration (val)

Temperature 3.15; ECE 0.080 -> 0.016. Threshold scan: the smallest threshold
with under 1% critical misses on covered rows is 0.95, at 54% coverage. At
0.80 coverage is 82% with 4.0% critical misses on covered rows.

## Export

fp32 ONNX 2,275 MB, matches torch (max diff 5.5e-5, argmax agreement 1.000).
int8 (dynamic quantization) 571 MB but **argmax agreement drops to route
0.915, intent 0.707**, with logit differences up to 25. The int8 artifact is
not usable as exported; see next steps.

## Training curve

| epoch | train loss (end) | val SWE | val macro-F1 | val critical miss | val intent acc | val span F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.085 | 0.496 | 0.806 | 9.7% | 0.644 | 0.931 |
| 2 | 0.027 | 0.388 | 0.828 | 5.2% | 0.606 | 0.967 |
| 3 | 0.002 | 0.336 | 0.859 | 6.5% | 0.608 | 0.976 |

Train loss reaches ~0.002 while val keeps improving slowly: the model
memorises the training seeds' phrasings. Val-to-test gap (0.859 vs 0.780
macro-F1) is seed variance with only 26 seeds per split.

## Reading

1. **In distribution, the router beats rules on every routing metric** and
   cuts span leakage by 2.6x. The gain is where the PRD predicted: paraphrase,
   code-mix and ASR noise on known intents.
2. **It does not generalise to unseen intents.** On OOD it defaults to LOCAL
   (DEFER recall 0.34, CLOUD recall 0.43 on test+ood combined, almost all of
   the misses are OOD rows). It learned intent identity, not the
   capability/distraction/actuation semantics that transfer.
3. **The intent head is weak** (0.50 on test, no better than keywords): 65-way
   over mean-pooled tokens, memorised to zero train loss, and each test seed is
   a phrasing the model never saw.
4. **Critical misses are REFUSE predicted as LOCAL** (276 of 321 on
   test+ood). The calibrated gate recovers much of this at the cost of
   coverage.
5. **The int8 export is broken by dynamic quantization**, not by the model.

## Next steps, in order

1. Run the `--no-context` ablation (the paper's thesis test).
2. Replace or supplement the intent head with **tag heads** (capability,
   distraction, actuation, consequence). Tags are what the rulebook consumes
   and what transfers to unseen intents; the intent head can stay as an
   auxiliary.
3. Regularise: 2 epochs, LoRA dropout 0.1, weight decay, label smoothing on
   the route head; compare on val.
4. Fix int8: per-channel weight-only quantization of MatMul with reduce_range
   off, or static quantization with a val calibration set; score the int8
   file on the test split and require route agreement > 0.99.
5. More seeds per intent (6 to 8) so held-out seeds are less of a surprise;
   this is the cheapest lever on the test/val gap.
