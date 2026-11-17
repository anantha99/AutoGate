"""Small classification heads on cached encoder vectors.

A :class:`Head` is a linear softmax layer, or one hidden layer (ReLU) then
softmax, trained full-batch with AdamW on fp32 vectors. Training is
deterministic for a given seed (CPU, fixed initialisation). Inference is
plain numpy, so a fitted head needs no torch at serving time.

Two kinds of heads are trained per question (see ``docs/plan-a7-decision-router.md``):

* **weighted**: class weights from ``autogate_router.data.tag_class_weights``
  (danger x 1/sqrt(freq), as R4). They feed the conservative decoder, so
  the comparison with R4's decoding stays like for like.
* **plain** (unweighted): their probabilities feed the expected-cost decoder
  and the abstention score. Class weights shift the prior, which a scalar
  temperature cannot undo, so a decoder that multiplies probabilities by
  costs needs unweighted heads.

Temperatures are fitted per question with ``autogate_router.calibrate``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class HeadConfig:
    hidden: int = 0
    weight_decay: float = 1e-3
    lr: float = 1e-2
    steps: int = 300
    dropout: float = 0.0


@dataclass
class Head:
    n_classes: int
    config: HeadConfig
    params: dict[str, np.ndarray] = field(default_factory=dict)
    temperature: float = 1.0

    def logits(self, x: np.ndarray) -> np.ndarray:
        h = x
        if "w1" in self.params:
            h = np.maximum(h @ self.params["w1"] + self.params["b1"], 0.0)
        return h @ self.params["w"] + self.params["b"]

    def probs(self, x: np.ndarray, calibrated: bool = True) -> np.ndarray:
        return softmax(self.logits(x), self.temperature if calibrated else 1.0)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def train_head(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    config: HeadConfig | None = None,
    class_weights: Sequence[float] | None = None,
    seed: int = 0,
) -> Head:
    """Fit one head on rows of ``x`` with labels ``y`` (rows with ``y < 0`` are ignored)."""
    import torch

    config = config or HeadConfig()
    keep = y >= 0
    xt = torch.tensor(x[keep], dtype=torch.float32)
    yt = torch.tensor(y[keep], dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    d = xt.shape[1]
    layers: list[torch.nn.Module] = []
    if config.hidden:
        layers += [torch.nn.Linear(d, config.hidden), torch.nn.ReLU()]
        if config.dropout:
            layers.append(torch.nn.Dropout(config.dropout))
        d = config.hidden
    out = torch.nn.Linear(d, n_classes)
    layers.append(out)
    net = torch.nn.Sequential(*layers)
    with torch.no_grad():
        for m in net:
            if isinstance(m, torch.nn.Linear):
                bound = 1.0 / np.sqrt(m.in_features)
                m.weight.copy_(torch.empty_like(m.weight).uniform_(-bound, bound, generator=g))
                m.bias.zero_()
    w = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    net.train()
    torch.manual_seed(seed)
    for _ in range(config.steps):
        opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward()
        opt.step()
    net.eval()
    params: dict[str, np.ndarray] = {}
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    if config.hidden:
        params["w1"] = linears[0].weight.detach().numpy().T.copy()
        params["b1"] = linears[0].bias.detach().numpy().copy()
    params["w"] = linears[-1].weight.detach().numpy().T.copy()
    params["b"] = linears[-1].bias.detach().numpy().copy()
    return Head(n_classes, config, params)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    keep = y >= 0
    p = probs[keep, y[keep]]
    return float(-np.log(np.clip(p, 1e-12, None)).mean())


def fit_temperature(head: Head, x: np.ndarray, y: np.ndarray) -> float:
    from autogate_router.calibrate import fit_temperature as _fit

    keep = y >= 0
    head.temperature = float(_fit(head.logits(x[keep]), y[keep]))
    return head.temperature


def ece(probs: np.ndarray, y: np.ndarray) -> float:
    from autogate_router.calibrate import ece as _ece

    keep = y >= 0
    return float(_ece(probs[keep], y[keep]))
