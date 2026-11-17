"""Attention-pooling heads on cached per-token states (Phase 2b, run R6a2).

The encoder stays frozen; only the pooling is learned. For a text with token
states ``h_1..h_L`` (each L2-normalised first, so the encoder's scale does
not matter):

    a_t = softmax_t(h_t . q)            one learned query per question
    z   = [sum_t a_t h_t ; mean_t h_t]  attention-pooled and mean-pooled
    logits = W z + b                    (or one hidden ReLU layer first)

so each question can look at the words that decide it ("how do I" against
"open") instead of a single average. Trained with AdamW on minibatches,
deterministic per seed. Inference is numpy.

``train_head`` in :mod:`autogate_decide.heads` dispatches here when its
input is a :class:`autogate_decide.embed.TokenStates`, so the runner needs
no other change.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from autogate_decide.embed import TokenStates
    from autogate_decide.heads import Head, HeadConfig

BATCH = 256


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def features(params: dict[str, np.ndarray], x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``[b, 2 * dim]`` pooled features (numpy forward)."""
    h = _norm(x)
    s = h @ params["q"]
    s = np.where(mask, s, -np.inf)
    s = s - s.max(1, keepdims=True)
    a = np.exp(s) * mask
    a = a / a.sum(1, keepdims=True)
    att = (a[..., None] * h).sum(1)
    mean = (h * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
    return np.concatenate([att, mean], axis=1)


def token_logits(params: dict[str, np.ndarray], x: TokenStates) -> np.ndarray:
    out = []
    for i in range(0, len(x), BATCH):
        xb, mb = x.padded(np.arange(i, min(i + BATCH, len(x))))
        z = features(params, xb, mb)
        if "w1" in params:
            z = np.maximum(z @ params["w1"] + params["b1"], 0.0)
        out.append(z @ params["w"] + params["b"])
    k = params["b"].shape[0]
    return np.concatenate(out) if out else np.zeros((0, k))


def train_token_head(
    x: TokenStates,
    y: np.ndarray,
    n_classes: int,
    config: HeadConfig,
    class_weights: Sequence[float] | None = None,
    seed: int = 0,
) -> Head:
    import torch

    from autogate_decide.heads import Head

    keep = np.flatnonzero(y >= 0)
    xs, ys = x[keep], y[keep]
    d = x.dim
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.Parameter(torch.zeros(d))
    layers: list[torch.nn.Module] = []
    width = 2 * d
    if config.hidden:
        layers += [torch.nn.Linear(width, config.hidden), torch.nn.ReLU()]
        width = config.hidden
    layers.append(torch.nn.Linear(width, n_classes))
    net = torch.nn.Sequential(*layers)
    with torch.no_grad():
        for m in net:
            if isinstance(m, torch.nn.Linear):
                bound = 1.0 / np.sqrt(m.in_features)
                m.weight.copy_(torch.empty_like(m.weight).uniform_(-bound, bound, generator=g))
                m.bias.zero_()
    w = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(
        [q, *net.parameters()], lr=config.lr * 0.3, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(seed)
    n = len(xs)
    for _ in range(config.epochs):
        order = rng.permutation(n)
        for i in range(0, n, BATCH):
            idx = order[i : i + BATCH]
            xb, mb = xs.padded(idx)
            h = torch.nn.functional.normalize(torch.from_numpy(xb), dim=-1)
            m = torch.from_numpy(mb)
            s = (h @ q).masked_fill(~m, float("-inf"))
            a = torch.softmax(s, dim=1)
            att = (a.unsqueeze(-1) * h).sum(1)
            mean = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
            logits = net(torch.cat([att, mean], dim=1))
            loss = loss_fn(logits, torch.from_numpy(ys[idx]).long())
            opt.zero_grad()
            loss.backward()
            opt.step()
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    params = {"q": q.detach().numpy().copy()}
    if config.hidden:
        params["w1"] = linears[0].weight.detach().numpy().T.copy()
        params["b1"] = linears[0].bias.detach().numpy().copy()
    params["w"] = linears[-1].weight.detach().numpy().T.copy()
    params["b"] = linears[-1].bias.detach().numpy().copy()
    return Head(n_classes, config, params, kind="tokens")
