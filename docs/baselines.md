# Zero-shot LLM baselines

`autogate_eval.baselines.llm` prompts an LLM with the full labeling policy
and asks it, for each row, for the intent, the sensitive spans and the
route. One run writes two predictions files, the paper's two decision modes:

| file | route | what its errors are |
| --- | --- | --- |
| `predictions_e2e.jsonl` | the route the model chose | perception and policy reasoning |
| `predictions_rulebook.jsonl` | `rulebook.route(answered intent, context, any answered span)` | perception only |

The prompt is built from the `Policy` in use (`--policy`, default built-in),
so relabelling for an OEM variant (RQ4) changes what the model is told as
well as what it is scored against. `--dry-run` prints it (about 2,700
tokens) and the first request.

```
uv sync --extra baselines
```

## The baseline tiers

Every command below scores the test and OOD splits (9,585 rows) and writes
`responses.jsonl`, both predictions files, `score_e2e.json`,
`score_rulebook.json` and `summary.json` (answer counts, tokens, latency,
estimated cost) to `--out-dir`. Try `--sample 200` first to check the
setup and the cost; a run is resumable with `--resume`.

**Oracle-perception ceiling** (no model; gold intent and spans):

```
uv run python -m autogate_eval.baselines.llm --rows data/generated/full/rows.jsonl \
    --provider oracle --out-dir runs/oracle
```

Its rulebook file scores perfectly by construction. Report it anyway: it
shows that every error of the decomposed router is a perception error.

**Tiers 2 and 3: on-device models, served locally by vLLM** (a GPU such as
Kaggle's or Colab's). vLLM caches the shared system prompt across requests.

```
vllm serve Qwen/Qwen3-0.6B            # tier 2; in another shell:
uv run python -m autogate_eval.baselines.llm --rows data/generated/full/rows.jsonl \
    --provider openai --base-url http://localhost:8000/v1 --model Qwen/Qwen3-0.6B \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' --temperature 0 \
    --out-dir runs/qwen3-0.6b

vllm serve Qwen/Qwen3-4B-Instruct-2507   # tier 3 (a non-thinking model)
uv run python -m autogate_eval.baselines.llm --rows data/generated/full/rows.jsonl \
    --provider openai --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-4B-Instruct-2507 --temperature 0 --out-dir runs/qwen3-4b
```

The default `--response-format json_schema` makes vLLM constrain the output
to the answer schema. For a server without schema support use
`json_object` or `none`; answers are then parsed leniently and anything
unusable is counted as invalid.

**Tier 4: frontier cloud models.**

```
export ANTHROPIC_API_KEY=...
uv run python -m autogate_eval.baselines.llm --rows data/generated/full/rows.jsonl \
    --provider anthropic --model claude-opus-5-5 --out-dir runs/claude-opus-5-5

export OPENAI_API_KEY=...
uv run python -m autogate_eval.baselines.llm --rows data/generated/full/rows.jsonl \
    --provider openai --model <GPT-6 Astra id from OpenAI's model list> \
    --price-in 10 --price-out 50 --out-dir runs/gpt-6-astra

export GEMINI_API_KEY=...
uv run python -m autogate_eval.baselines.llm --rows data/generated/full/rows.jsonl \
    --provider openai --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
    --api-key-env GEMINI_API_KEY --model <Gemini 3.1 Pro id from Google's model list> \
    --price-in 2 --price-out 12 --out-dir runs/gemini-3.1-pro
```

Check the model ids, the OpenAI-compatible base URL for Gemini and the
prices on the providers' own pages on the day of the run, and record that
date in the paper. Claude prices are built in. Optional ceiling check on a
subset: `--model claude-fable-5-1 --sample 2000`.

Claude runs use `output_config.effort` `low` (`--effort` to change; Opus
5.5 cannot turn thinking off) and cache the system prompt. They also opt
into server-side refusal fallbacks, so a declined request is answered by
another Claude model. Each answer records the model that served it and
`summary.json` counts any served by a fallback; for a pure baseline, rerun
with `--no-fallbacks` if that count is not zero.

## Invalid answers

A row whose answer is missing or unusable (no JSON, an unknown intent or
route, a refusal, a cut-off answer, an API error after retries) is routed
`--on-invalid` (default `REFUSE`: a gate that cannot decide fails closed),
with intent `__invalid__` and no spans. `summary.json` reports how many
and why. Report the count with the scores.

## Scoring again

Both predictions files are ordinary `autogate_eval.score` inputs:

```
uv run python -m autogate_eval.score --rows data/generated/full/rows.jsonl \
    --preds runs/claude-opus-5-5/predictions_rulebook.jsonl --split test
```
