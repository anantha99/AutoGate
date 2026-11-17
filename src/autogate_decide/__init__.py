"""Decision-encoder router (approach A7, ``docs/plan-a7-decision-router.md``).

A frozen text encoder embeds the utterance once; small heads answer five
fixed questions (capability, distraction, actuation, consequence, and
whether the utterance carries sensitive content); the rulebook turns the
answers and the row's context into a route.

Modules:

* :mod:`autogate_decide.encoders`: encoder specs, the Hugging Face mean-pool
  encoder and a deterministic fake encoder for tests.
* :mod:`autogate_decide.embed`: the embedding cache (one vector per unique
  utterance text).
* :mod:`autogate_decide.data`: utterance-level examples joined to the cache.
* :mod:`autogate_decide.heads`: the five trained heads and their temperatures.
* :mod:`autogate_decide.route`: conservative and expected-cost decoders, and
  the abstention scan.
* :mod:`autogate_decide.zeroshot`: option-description baseline (run R6z).
* :mod:`autogate_decide.diagnostics`: per-variant, calibration, sensitive
  and minimal-pair reports.
* :mod:`autogate_decide.run`: one run end to end, writing predictions and a
  report.
"""
