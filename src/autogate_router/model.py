"""The AutoGate router: one backbone forward pass, a route head and its helpers.

* ``route``: 5-way over the pooled utterance representation, cross-entropy
  with per-class weights from the cost matrix
  (:func:`autogate_router.data.class_weights_from_cost_matrix`).
* ``intent``: 65-way over the same pooled representation, plain
  cross-entropy (auxiliary).
* ``span``: per-token BIO over the 11 sensitive-span tags, cross-entropy
  ignoring ``-100`` (prefix, special and padding tokens).
* tag heads (``tag_heads=True``): ``capability`` (4), ``distraction`` (3),
  ``actuation`` (4) and ``consequence`` (3), each a linear layer over the
  pooled vector (after the head dropout), cross-entropy with per-class
  weights from :func:`autogate_router.data.tag_class_weights` stored in the
  config. These are the intent tags the rulebook reads, so a route can also
  be computed from them (``autogate_router.tag_routing``).

``loss = route + lambda_intent * intent + lambda_span * span
+ sum_k lambda_tags[k] * tag_k``. ``label_smoothing`` applies to the route,
intent and tag cross-entropies, not to the span loss.

Pooling is ``mean`` over utterance tokens (``utterance_mask``; the context
prefix and padding are masked out) or ``last``, the hidden state of the last
utterance token, which is the natural summary for a causal backbone such as
Qwen3 since it has attended to everything before it (inputs are
right-padded).

A checkpoint directory holds ``router_config.json``, ``labels.json``,
``heads.safetensors``, the tokenizer in ``tokenizer/``, and either the LoRA
adapter in ``adapter/`` (the base backbone is loaded by name) or the full
backbone in ``backbone/`` when trained with ``--no-lora``. A checkpoint
saved before the tag heads existed (no ``tag_heads`` key in its config, or a
``labels.json`` without tag vocabularies) loads without them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from autogate_router.labels import (
    DEFAULT_LABELS,
    IGNORE_INDEX,
    TAG_NAMES,
    Labels,
    load_labels,
    save_labels,
)

CONFIG_FILE = "router_config.json"
HEADS_FILE = "heads.safetensors"
ADAPTER_DIR = "adapter"
BACKBONE_DIR = "backbone"
TOKENIZER_DIR = "tokenizer"

DEFAULT_BACKBONE = "Qwen/Qwen3-0.6B"
# Attention and MLP projections of Qwen2/Qwen3 decoder layers.
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
POOLINGS = ("mean", "last")
DEFAULT_LAMBDA_TAGS: dict[str, float] = {
    "actuation": 1.0,
    "distraction": 1.0,
    "capability": 0.5,
    "consequence": 0.3,
}


@dataclass
class RouterConfig:
    """Everything needed to rebuild the model around saved weights."""

    backbone: str = DEFAULT_BACKBONE
    pooling: str = "mean"
    use_context: bool = True
    max_length: int = 96
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    head_dropout: float = 0.1
    lambda_intent: float = 0.5
    lambda_span: float = 1.0
    route_class_weights: list[float] | None = None
    n_routes: int = len(DEFAULT_LABELS.routes)
    n_intents: int = len(DEFAULT_LABELS.intents)
    n_bio: int = len(DEFAULT_LABELS.bio_tags)
    tag_heads: bool = True
    lambda_tags: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LAMBDA_TAGS))
    label_smoothing: float = 0.0
    tag_class_weights: dict[str, list[float]] | None = None
    """Per tag, loss weights in ``labels.json`` order (from the train rows)."""

    def __post_init__(self) -> None:
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {self.pooling!r}")
        unknown = set(self.lambda_tags) - set(TAG_NAMES)
        if unknown:
            raise ValueError(f"lambda_tags has unknown tags {sorted(unknown)}")
        self.lambda_tags = {k: float(self.lambda_tags.get(k, 0.0)) for k in TAG_NAMES}
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RouterConfig:
        """Rebuild from ``router_config.json``; a config without ``tag_heads`` predates them."""
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        kw.setdefault("tag_heads", False)
        return cls(**kw)


def pool(hidden: torch.Tensor, mask: torch.Tensor, how: str = "mean") -> torch.Tensor:
    """Pool ``hidden`` [B, T, H] over positions where ``mask`` [B, T] is 1.

    ``mean`` averages them; ``last`` takes the last such position. A row
    with an empty mask (the utterance was truncated away) pools to zeros
    for ``mean`` and to position 0 for ``last``.
    """
    m = mask.to(hidden.dtype)
    if how == "mean":
        return (hidden * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)
    if how == "last":
        positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        last = (positions * mask.long()).amax(dim=1)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    raise ValueError(f"unknown pooling {how!r}")


def _dtype_kwarg() -> str:
    """``dtype`` on transformers >= 4.56, ``torch_dtype`` before (older ones ignore ``dtype``)."""
    import transformers
    from packaging.version import Version

    return "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"


def _load_backbone(name_or_path: str, attn_implementation: str | None = None) -> nn.Module:
    """The bare backbone (no LM head) in fp32; autocast handles mixed precision."""
    from transformers import AutoModel

    kw: dict[str, Any] = {_dtype_kwarg(): torch.float32}
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    return AutoModel.from_pretrained(name_or_path, **kw)


class AutoGateRouter(nn.Module):
    def __init__(
        self,
        config: RouterConfig,
        backbone: nn.Module | None = None,
        labels: Labels = DEFAULT_LABELS,
        attn_implementation: str | None = None,
    ):
        super().__init__()
        self.config = config
        self.labels = labels
        if (config.n_routes, config.n_intents, config.n_bio) != (
            len(labels.routes),
            len(labels.intents),
            len(labels.bio_tags),
        ):
            raise ValueError("config head sizes do not match the labels")
        if config.tag_heads and not labels.has_tags:
            raise ValueError("tag_heads=True needs labels with tag vocabularies")
        self.tag_names: tuple[str, ...] = tuple(labels.tags) if config.tag_heads else ()
        if backbone is None:
            backbone = _load_backbone(config.backbone, attn_implementation)
            if config.use_lora:
                backbone = self._wrap_lora(backbone, config)
        self.backbone = backbone
        hidden = self._hidden_size(backbone)
        self.dropout = nn.Dropout(config.head_dropout)
        self.route_head = nn.Linear(hidden, config.n_routes)
        self.intent_head = nn.Linear(hidden, config.n_intents)
        self.span_head = nn.Linear(hidden, config.n_bio)
        weights = config.route_class_weights
        self.register_buffer(
            "route_weights",
            torch.tensor(weights, dtype=torch.float32) if weights else torch.ones(config.n_routes),
            persistent=False,
        )
        tag_weights = config.tag_class_weights or {}
        for name in self.tag_names:
            n = len(labels.tags[name])
            setattr(self, f"{name}_head", nn.Linear(hidden, n))
            w = tag_weights.get(name)
            if w is not None and len(w) != n:
                raise ValueError(f"{name}: {len(w)} class weights for {n} classes")
            self.register_buffer(
                f"{name}_weights",
                torch.tensor(w, dtype=torch.float32) if w else torch.ones(n),
                persistent=False,
            )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _hidden_size(backbone: nn.Module) -> int:
        cfg = backbone.config
        return int(getattr(cfg, "hidden_size", None) or cfg.get_text_config().hidden_size)

    @staticmethod
    def _wrap_lora(backbone: nn.Module, config: RouterConfig) -> nn.Module:
        from peft import LoraConfig, TaskType, get_peft_model

        lora = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
        )
        return get_peft_model(backbone, lora)

    def head_modules(self) -> dict[str, nn.Module]:
        heads = {"route": self.route_head, "intent": self.intent_head, "span": self.span_head}
        heads.update({name: getattr(self, f"{name}_head") for name in self.tag_names})
        return heads

    @property
    def has_tag_heads(self) -> bool:
        return bool(self.tag_names)

    @property
    def output_names(self) -> tuple[str, ...]:
        """Logit outputs in a fixed order: route, intent, span, then the tags."""
        return (
            "route_logits",
            "intent_logits",
            "span_logits",
            *(f"{n}_logits" for n in self.tag_names),
        )

    def head_parameters(self) -> list[nn.Parameter]:
        return [p for m in self.head_modules().values() for p in m.parameters()]

    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        utterance_mask: torch.Tensor,
        route_labels: torch.Tensor | None = None,
        intent_labels: torch.Tensor | None = None,
        span_labels: torch.Tensor | None = None,
        capability_labels: torch.Tensor | None = None,
        distraction_labels: torch.Tensor | None = None,
        actuation_labels: torch.Tensor | None = None,
        consequence_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hidden = out.last_hidden_state.float()
        pooled = self.dropout(pool(hidden, utterance_mask, self.config.pooling))
        result: dict[str, torch.Tensor] = {
            "route_logits": self.route_head(pooled),
            "intent_logits": self.intent_head(pooled),
            "span_logits": self.span_head(self.dropout(hidden)),
        }
        for name in self.tag_names:
            result[f"{name}_logits"] = getattr(self, f"{name}_head")(pooled)
        given_tags = {
            "capability": capability_labels,
            "distraction": distraction_labels,
            "actuation": actuation_labels,
            "consequence": consequence_labels,
        }
        if (
            route_labels is None
            and intent_labels is None
            and span_labels is None
            and all(v is None for v in given_tags.values())
        ):
            return result
        smooth = self.config.label_smoothing
        zero = result["route_logits"].new_zeros(())
        route_loss = intent_loss = span_loss = zero
        if route_labels is not None:
            route_loss = F.cross_entropy(
                result["route_logits"],
                route_labels,
                weight=self.route_weights,
                label_smoothing=smooth,
            )
        if intent_labels is not None and (intent_labels != IGNORE_INDEX).any():
            intent_loss = F.cross_entropy(
                result["intent_logits"],
                intent_labels,
                ignore_index=IGNORE_INDEX,
                label_smoothing=smooth,
            )
        tags_loss = zero
        for name in self.tag_names:
            y = given_tags[name]
            loss_k = zero
            if y is not None and (y != IGNORE_INDEX).any():
                loss_k = F.cross_entropy(
                    result[f"{name}_logits"],
                    y,
                    weight=getattr(self, f"{name}_weights"),
                    ignore_index=IGNORE_INDEX,
                    label_smoothing=smooth,
                )
            result[f"{name}_loss"] = loss_k
            tags_loss = tags_loss + self.config.lambda_tags[name] * loss_k
        if span_labels is not None and (span_labels != IGNORE_INDEX).any():
            span_loss = F.cross_entropy(
                result["span_logits"].reshape(-1, self.config.n_bio),
                span_labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        result["route_loss"] = route_loss
        result["intent_loss"] = intent_loss
        result["span_loss"] = span_loss
        result["tags_loss"] = tags_loss
        result["loss"] = (
            route_loss
            + self.config.lambda_intent * intent_loss
            + self.config.lambda_span * span_loss
            + tags_loss
        )
        return result

    # ------------------------------------------------------------------ #

    def trainable_summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"trainable params {train:,} / {total:,} ({100 * train / max(total, 1):.2f}%)"

    def save_pretrained(self, directory: str | Path, tokenizer: Any | None = None) -> Path:
        from safetensors.torch import save_file

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / CONFIG_FILE).write_text(
            json.dumps(self.config.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        save_labels(d, self.labels)
        heads = {
            f"{name}.{k}": v.detach().cpu().contiguous()
            for name, m in self.head_modules().items()
            for k, v in m.state_dict().items()
        }
        save_file(heads, str(d / HEADS_FILE))
        if self.config.use_lora:
            self.backbone.save_pretrained(str(d / ADAPTER_DIR))
        else:
            self.backbone.save_pretrained(str(d / BACKBONE_DIR))
        if tokenizer is not None:
            tokenizer.save_pretrained(str(d / TOKENIZER_DIR))
        return d

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        attn_implementation: str | None = None,
        merge_lora: bool = False,
    ) -> AutoGateRouter:
        """Rebuild a saved router. ``merge_lora`` folds the adapter into the backbone."""
        from safetensors.torch import load_file

        d = Path(directory)
        config = RouterConfig.from_dict(json.loads((d / CONFIG_FILE).read_text(encoding="utf-8")))
        labels = load_labels(d)
        if not labels.has_tags:
            config.tag_heads = False
        if config.use_lora:
            from peft import PeftModel

            base = _load_backbone(config.backbone, attn_implementation)
            backbone = PeftModel.from_pretrained(base, str(d / ADAPTER_DIR))
            if merge_lora:
                backbone = backbone.merge_and_unload()
        else:
            backbone = _load_backbone(str(d / BACKBONE_DIR), attn_implementation)
        model = cls(config, backbone=backbone, labels=labels)
        state = load_file(str(d / HEADS_FILE))
        for name, m in model.head_modules().items():
            prefix = f"{name}."
            m.load_state_dict(
                {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
            )
        return model.eval()


def load_tokenizer(name_or_path: str | Path) -> Any:
    """A fast tokenizer with a pad token (EOS when the tokenizer has none)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(name_or_path), use_fast=True)
    if not tok.is_fast:
        raise ValueError(f"{name_or_path}: a fast tokenizer is required for offset mapping")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok


def checkpoint_tokenizer(directory: str | Path) -> Any:
    """The tokenizer saved with a checkpoint, or the backbone's if none was saved."""
    d = Path(directory)
    if (d / TOKENIZER_DIR).is_dir():
        return load_tokenizer(d / TOKENIZER_DIR)
    config = RouterConfig.from_dict(json.loads((d / CONFIG_FILE).read_text(encoding="utf-8")))
    return load_tokenizer(config.backbone)
