"""Generate the dataset.

    python -m autogate_bench.generate --out data/generated/pilot --rng-seed 0
    python -m autogate_bench.generate --out data/generated/full --paraphrases data/paraphrases

writes ``rows.jsonl``, ``rows.parquet`` and ``manifest.json`` to ``--out``.
Without ``--paraphrases`` the dataset is seed-only (the pilot). With it, every
paraphrase and ``--asr-per-seed`` ASR copies per seed are added. The same
arguments always give a byte-identical ``rows.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autogate_bench.dataset import build_rows, write_jsonl, write_manifest, write_parquet
from autogate_bench.paraphrases import load_paraphrases
from autogate_bench.policy import DEFAULT_POLICY, Policy
from autogate_bench.seeds import DEFAULT_SEEDS_PATH, load_seeds


def _rel(path: Path) -> str:
    cwd = Path.cwd().resolve()
    return str(path.resolve().relative_to(cwd)) if path.resolve().is_relative_to(cwd) else str(path)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--contexts-per-utterance", type=int, default=3)
    p.add_argument("--n-sensitive", type=int, default=2, help="sensitive fills per seed")
    p.add_argument("--n-generic", type=int, default=1, help="relation-word fills per contact seed")
    p.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS_PATH)
    p.add_argument("--policy", type=Path, default=None, help="policy JSON (default: built-in)")
    p.add_argument(
        "--paraphrases",
        type=Path,
        default=None,
        help="paraphrase directory, e.g. data/paraphrases (default: none, seed-only)",
    )
    p.add_argument(
        "--asr-per-seed",
        type=int,
        default=4,
        help="ASR copies per seed, made from its plain paraphrases (needs --paraphrases)",
    )
    a = p.parse_args(argv)
    if a.paraphrases is not None and not a.paraphrases.is_dir():
        p.error(f"--paraphrases: no such directory {a.paraphrases}")

    policy = Policy.load(a.policy) if a.policy else DEFAULT_POLICY
    rows = build_rows(
        load_seeds(a.seeds),
        rng_seed=a.rng_seed,
        contexts_per_utterance=a.contexts_per_utterance,
        n_sensitive=a.n_sensitive,
        n_generic=a.n_generic,
        policy=policy,
        paraphrases=load_paraphrases(a.paraphrases) if a.paraphrases else None,
        asr_per_seed=a.asr_per_seed,
    )
    a.out.mkdir(parents=True, exist_ok=True)
    jsonl = a.out / "rows.jsonl"
    write_jsonl(rows, jsonl)
    write_parquet(rows, a.out / "rows.parquet")
    args = {
        "rng_seed": a.rng_seed,
        "contexts_per_utterance": a.contexts_per_utterance,
        "n_sensitive": a.n_sensitive,
        "n_generic": a.n_generic,
        "seeds": _rel(a.seeds),
        "policy": str(a.policy) if a.policy else "default",
        "paraphrases": _rel(a.paraphrases) if a.paraphrases else None,
        "asr_per_seed": a.asr_per_seed if a.paraphrases else 0,
    }
    manifest = write_manifest(rows, a.out / "manifest.json", args, jsonl)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("n_rows", "n_utterances", "split", "route", "slice", "source", "variant")
            },
            indent=2,
        )
    )
    return manifest


if __name__ == "__main__":
    main()
