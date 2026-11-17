"""Text encoders: one fixed-size vector per utterance.

Every encoder is described by an :class:`EncoderSpec` (model id, pinned
revision, task prefix, pooling) so a run's manifest can record exactly what
produced its vectors. :func:`load_encoder` builds one from a spec name in
:data:`ENCODERS`.

* ``embeddinggemma-2``: the A7 candidate. Text tower only (the vision and
  audio configs are dropped before loading, 271M parameters), fp32, the
  model card's classification prefix ``task: classification | query: ``,
  mean pooling over every non-padding token including the prefix (the
  card's ``1_Pooling`` has ``include_prompt: true``), then L2 normalisation.
  The model returns per-token states already projected to 768 dimensions,
  so pooling after the projection equals the card's pipeline.
* ``e5-small``: the control encoder (``intfloat/multilingual-e5-small``,
  ungated, multilingual, 118M), prefix ``query: ``, mean pooling, L2.
* ``fake``: :class:`FakeEncoder`, a deterministic hashed bag of words and
  character trigrams. No download; used by the unit tests. Its numbers
  mean nothing.

Precision is fp32 throughout: the EmbeddingGemma 2 card warns that float16
overflows to NaN silently.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    model_id: str
    revision: str | None
    prefix: str
    dim: int
    text_only: bool = False
    max_length: int = 128

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ENCODERS: dict[str, EncoderSpec] = {
    "embeddinggemma-2": EncoderSpec(
        name="embeddinggemma-2",
        model_id="google/embeddinggemma-2",
        revision="f9972ac22e696a105c482202646979e33e4b5a1e",
        prefix="task: classification | query: ",
        dim=768,
        text_only=True,
    ),
    "e5-small": EncoderSpec(
        name="e5-small",
        model_id="intfloat/multilingual-e5-small",
        revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        prefix="query: ",
        dim=384,
    ),
    "fake": EncoderSpec(name="fake", model_id="fake", revision=None, prefix="", dim=64),
}


class Encoder(Protocol):
    spec: EncoderSpec

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """``[len(texts), dim]`` float32, L2-normalised rows."""
        ...

    def token_states(
        self, texts: Sequence[str], layers: Sequence[int] | None = None
    ) -> list[np.ndarray]:
        """Per text, ``[n_tokens, dim]`` float32 states (no padding; prefix included).

        ``layers`` picks entries of the model's ``hidden_states`` (0 = embeddings) and
        concatenates them along the last axis; ``None`` is the encoder's output."""
        ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.clip(n, 1e-12, None)).astype(np.float32)


_TOKEN = re.compile(r"\w+", re.UNICODE)


class FakeEncoder:
    """Deterministic hashed features: words and character trigrams, signed, L2-normalised."""

    def __init__(self, spec: EncoderSpec = ENCODERS["fake"]):
        self.spec = spec

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        v = int.from_bytes(h, "little")
        return v % self.spec.dim, 1.0 if (v >> 63) & 1 else -1.0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.spec.dim), dtype=np.float64)
        for i, t in enumerate(texts):
            words = _TOKEN.findall(t.lower())
            feats = [f"w:{w}" for w in words]
            joined = " ".join(words)
            feats += [f"c:{joined[j : j + 3]}" for j in range(max(len(joined) - 2, 0))]
            for f in feats:
                k, s = self._bucket(f)
                out[i, k] += s
        return _l2(out)

    def token_states(
        self, texts: Sequence[str], layers: Sequence[int] | None = None
    ) -> list[np.ndarray]:
        out = []
        for t in texts:
            words = _TOKEN.findall(t.lower()) or [""]
            m = np.zeros((len(words), self.spec.dim))
            for i, w in enumerate(words):
                for f in (f"w:{w}", f"c:{w[:3]}"):
                    k, s = self._bucket(f)
                    m[i, k] += s
            m = m.astype(np.float32)
            if layers:  # fake "layers": scaled copies, enough to exercise the layer mix
                m = np.concatenate([m * (k + 1) for k in range(len(layers))], axis=1)
            out.append(m)
        return out


class HFEncoder:
    """A Hugging Face encoder with mean pooling over the attention mask, fp32, CPU or GPU."""

    def __init__(self, spec: EncoderSpec, device: str = "cpu", batch_size: int = 64):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.spec = spec
        self.device = device
        self.batch_size = batch_size
        kw: dict[str, Any] = {"revision": spec.revision}
        config = AutoConfig.from_pretrained(spec.model_id, **kw)
        if spec.text_only:
            config.vision_config = None
            config.audio_config = None
        self.model = (
            AutoModel.from_pretrained(spec.model_id, config=config, dtype=torch.float32, **kw)
            .to(device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, **kw)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def _states(self, batch: dict, layers: Sequence[int] | None = None) -> Any:
        import torch

        if not layers:
            return self.model(**batch).last_hidden_state
        hs = self.model(**batch, output_hidden_states=True).hidden_states
        return torch.cat([hs[i] for i in layers], dim=-1)

    def _batches(self, texts: Sequence[str], layers: Sequence[int] | None = None):
        """Yield ``(indices, states [b, L, D], attention_mask [b, L])`` in length order."""
        import torch

        texts = [self.spec.prefix + t for t in texts]
        order = np.argsort([len(t) for t in texts], kind="stable")
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                idx = order[i : i + self.batch_size]
                batch = self.tokenizer(
                    [texts[j] for j in idx],
                    padding=True,
                    truncation=True,
                    max_length=self.spec.max_length,
                    return_tensors="pt",
                ).to(self.device)
                states = self._states(batch, layers).float()
                if not torch.isfinite(states).all():
                    raise FloatingPointError(f"{self.spec.name}: non-finite states")
                yield idx, states, batch["attention_mask"]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.spec.dim), dtype=np.float32)
        for idx, states, attn in self._batches(texts):
            mask = attn.unsqueeze(-1).float()
            pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            out[idx] = pooled.cpu().numpy()
        return _l2(out)

    def token_states(
        self, texts: Sequence[str], layers: Sequence[int] | None = None
    ) -> list[np.ndarray]:
        out: list[np.ndarray] = [np.zeros((0, self.spec.dim), np.float32)] * len(texts)
        for idx, states, attn in self._batches(texts, layers):
            s, m = states.cpu().numpy(), attn.cpu().numpy().astype(bool)
            for k, j in enumerate(idx):
                out[j] = s[k][m[k]].astype(np.float32)
        return out


def load_encoder(name: str, device: str = "cpu", batch_size: int = 64) -> Encoder:
    spec = ENCODERS[name]
    if name == "fake":
        return FakeEncoder(spec)
    return HFEncoder(spec, device=device, batch_size=batch_size)
