r"""A tiny random Qwen3 backbone with its own BPE tokenizer, for CPU smoke runs and tests.

    python -m autogate_router.tiny --out checkpoints/tiny-qwen3 \
        --rows data/generated/pilot/rows.jsonl

Nothing is downloaded: the byte-level BPE tokenizer is trained on the rows'
utterances and context prefixes, and the model is a randomly initialised
Qwen3 (Qwen2 if the installed ``transformers`` has no Qwen3) with hidden
size 32, 2 layers and 2 heads. The directory it writes loads with
``AutoModel.from_pretrained`` and ``AutoTokenizer.from_pretrained`` like any
hub checkpoint, so ``--backbone <dir>`` works everywhere. It only shows that
the pipeline runs; its numbers mean nothing.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SPECIAL_TOKENS = ("<|endoftext|>", "<|pad|>", "<|unk|>")


def train_tokenizer(texts: Iterable[str], vocab_size: int = 1000) -> Any:
    """A byte-level BPE fast tokenizer (the same pre-tokenization family as Qwen)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(list(texts), trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        padding_side="right",
    )


def tiny_config(vocab_size: int, pad_token_id: int, eos_token_id: int) -> Any:
    kw = dict(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=eos_token_id,
        tie_word_embeddings=True,
    )
    try:
        from transformers import Qwen3Config

        return Qwen3Config(head_dim=16, **kw)
    except ImportError:  # older transformers
        from transformers import Qwen2Config

        return Qwen2Config(**kw)


def build_tiny_backbone(
    out: str | Path, rows: Iterable[Mapping[str, Any]], vocab_size: int = 1000, seed: int = 0
) -> Path:
    """Train the tokenizer on ``rows``, initialise the model, save both to ``out``."""
    import torch
    from transformers import AutoModel

    rows = list(rows)
    texts = [r["utterance"] for r in rows] + sorted({r["context_prefix"] for r in rows})
    tokenizer = train_tokenizer(texts, vocab_size)
    torch.manual_seed(seed)
    config = tiny_config(len(tokenizer), tokenizer.pad_token_id, tokenizer.eos_token_id)
    model = AutoModel.from_config(config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    return out


def main(argv: list[str] | None = None) -> Path:
    from autogate_router.data import load_rows

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True, help="rows.jsonl to train the tokenizer on")
    p.add_argument("--vocab-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    out = build_tiny_backbone(a.out, load_rows(a.rows), a.vocab_size, a.seed)
    print(f"wrote tiny backbone to {out}")
    return out


if __name__ == "__main__":
    main()
