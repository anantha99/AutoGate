# AutoGate

Router and benchmark that decides, per in-car utterance, whether an assistant
answers LOCAL, CLOUD, CLOUD_MASKED, DEFER or REFUSE.

## Read first

- `docs/experiment-log.md` is the project's running memory: current state,
  approaches, every run's results, decisions and open questions. Read it
  before starting work.
- Update it in the same commit whenever a run finishes, a decision is made,
  or an approach is added or dropped. Add a runs-table row even for failed
  runs.

## Working rules

- Labels come from the rulebook. Changing intents, tags, seeds, paraphrases
  or policy tables changes labels: regenerate the pilot and say so if its
  sha256 in `docs/pilot.md` moves.
- Paraphrase files: fix lines in place; never reorder or delete (line
  position is part of the row id).
- Generated data and checkpoints are gitignored; never commit them.
- Checks before pushing: `uv run ruff format src tests && uv run ruff check src tests && uv run pytest -q`.
