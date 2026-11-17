r"""The embedding cache: one vector per unique utterance text.

    python -m autogate_decide.embed --encoder embeddinggemma-2 \
        --rows data/generated/full/rows.jsonl --out data/generated/decide/embeddinggemma-2

Reads every ``--rows`` file, embeds each unique ``utterance`` text once (in
first-seen order) and writes to ``--out``:

* ``vectors.npy``: ``[n_texts, dim]`` float32, L2-normalised;
* ``texts.json``: the texts, in the order of the vectors;
* ``manifest.json``: the encoder spec (model id, pinned revision, prefix),
  parameter count, each rows file with its sha256 and row count, the git
  commit, wall time and a determinism check (the first 10 texts embedded a
  second time; the maximum absolute difference must be 0).

Several rows files may share one cache (the full dataset and the
augmented one used only for the minimal-pair probe); a text that appears in
both is embedded once. Generated data: ``data/generated/`` is gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autogate_decide.encoders import ENCODERS, Encoder, load_encoder

VECTORS_FILE = "vectors.npy"
TEXTS_FILE = "texts.json"
MANIFEST_FILE = "manifest.json"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unique_texts(rows: Iterable[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for r in rows:
        seen.setdefault(r["utterance"], None)
    return list(seen)


@dataclass
class EmbeddingCache:
    texts: list[str]
    vectors: np.ndarray
    manifest: dict

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.vectors):
            raise ValueError(f"{len(self.texts)} texts but {len(self.vectors)} vectors")
        self.index = {t: i for i, t in enumerate(self.texts)}

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def lookup(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self.index]
        if missing:
            raise KeyError(f"{len(missing)} texts not in the cache, e.g. {missing[:3]}")
        return self.vectors[[self.index[t] for t in texts]]

    def save(self, out: str | Path) -> Path:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / VECTORS_FILE, self.vectors.astype(np.float32))
        (out / TEXTS_FILE).write_text(json.dumps(self.texts, ensure_ascii=False), encoding="utf-8")
        (out / MANIFEST_FILE).write_text(json.dumps(self.manifest, indent=2) + "\n", "utf-8")
        return out

    @classmethod
    def load(cls, directory: str | Path) -> EmbeddingCache:
        d = Path(directory)
        texts = json.loads((d / TEXTS_FILE).read_text(encoding="utf-8"))
        manifest = json.loads((d / MANIFEST_FILE).read_text(encoding="utf-8"))
        return cls(texts, np.load(d / VECTORS_FILE), manifest)


def build_cache(
    encoder: Encoder, rows_paths: Sequence[str | Path], log_every: int = 2000
) -> EmbeddingCache:
    from autogate_bench.dataset import git_commit
    from autogate_router.data import load_rows

    sources, texts = [], []
    seen: set[str] = set()
    for path in rows_paths:
        rows = load_rows(path)
        sources.append({"path": str(path), "sha256": sha256_file(path), "n_rows": len(rows)})
        for t in unique_texts(rows):
            if t not in seen:
                seen.add(t)
                texts.append(t)
    t0 = time.time()
    chunks = []
    for i in range(0, len(texts), log_every):
        chunks.append(encoder.encode(texts[i : i + log_every]))
        print(f"  embedded {min(i + log_every, len(texts))}/{len(texts)} ({time.time() - t0:.0f}s)")
    vectors = np.concatenate(chunks) if chunks else np.zeros((0, encoder.spec.dim), np.float32)
    seconds = time.time() - t0
    probe = texts[:10]
    again = encoder.encode(probe) if probe else vectors[:0]
    max_diff = float(np.abs(again - vectors[: len(probe)]).max()) if probe else 0.0
    manifest = {
        "encoder": encoder.spec.to_dict(),
        "n_params": getattr(encoder, "n_params", None),
        "sources": sources,
        "n_texts": len(texts),
        "dim": int(vectors.shape[1]),
        "seconds": round(seconds, 1),
        "determinism_max_abs_diff": max_diff,
        "all_finite": bool(np.isfinite(vectors).all()),
        "git": git_commit(),
    }
    return EmbeddingCache(texts, vectors, manifest)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    p.add_argument("--rows", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args(argv)
    encoder = load_encoder(a.encoder, a.device, a.batch_size)
    cache = build_cache(encoder, a.rows)
    cache.save(a.out)
    m = cache.manifest
    print(
        f"wrote {m['n_texts']} x {m['dim']} to {a.out} in {m['seconds']}s; "
        f"determinism diff {m['determinism_max_abs_diff']}, finite {m['all_finite']}"
    )
    return m


if __name__ == "__main__":
    main()
