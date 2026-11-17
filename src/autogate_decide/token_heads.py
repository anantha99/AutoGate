"""Attention-pooling heads on cached per-token states (Phase 2b, run R6a2).

The encoder stays frozen; only the pooling is learned. Token states may hold
several encoder layers side by side (``TokenStates.n_layers``, from
``autogate_decide.embed --layers``). For one text with tokens ``t``:

    h_t = sum_l softmax(m)_l * LN(x_{t,l})    learned mix of layer-normalised layers
    a_{k,t} = softmax_t(h_t . q_k)            K learned queries (default 2)
    z = [sum_t a_{1,t} h_t ; ... ; mean_t h_t]
    logits = W z + b

``LN`` is a parameter-free layer norm per layer, so the layers' different
scales do not matter. The layer mix lets a question read intermediate
layers, where look-alike requests ("open the trunk" against "how do I open
the trunk") may still be apart; the final layer of an encoder trained for
mean pooling is homogenised. The mix goes beyond the plan's wording
(logged in ``docs/experiment-log.md``) and means Phase 5 needs a runtime
that exposes those layers.

Trained with AdamW on minibatches, deterministic per seed. Inference is
numpy. ``train_head`` in :mod:`autogate_decide.heads` dispatches here when its
input is a :class:`autogate_decide.embed.TokenStates`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from autogate_decide.embed import TokenStates
    from autogate_decide.heads import Head, HeadConfig

BATCH = 256
N_QUERIES = 2


def _ln(x: np.ndarray, n_layers: int) -> np.ndarray:
    """``[..., n_layers * d]`` -> ``[..., n_layers, d]``, each layer normalised."""
    x = x.reshape(*x.shape[:-1], n_layers, -1)
    mu = x.mean(-1, keepdims=True)
    sd = x.std(-1, keepdims=True)
    return (x - mu) / (sd + 1e-5)


def features(
    params: dict[str, np.ndarray], x: np.ndarray, mask: np.ndarray, n_layers: int
) -> np.ndarray:
    """``[b, (K + 1) * d]`` pooled features (numpy forward)."""
    w = np.exp(params["mix"] - params["mix"].max())
    w = w / w.sum()
    h = np.einsum("bLld,l->bLd", _ln(x, n_layers), w)
    s = np.einsum("bLd,kd->bkL", h, params["q"])
    s = np.where(mask[:, None, :], s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    a = np.exp(s) * mask[:, None, :]
    a = a / a.sum(-1, keepdims=True)
    att = np.einsum("bkL,bLd->bkd", a, h).reshape(len(x), -1)
    mean = (h * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
    return np.concatenate([att, mean], axis=1)


def token_logits(params: dict[str, np.ndarray], x: TokenStates) -> np.ndarray:
    out = []
    for i in range(0, len(x), BATCH):
        xb, mb = x.padded(np.arange(i, min(i + BATCH, len(x))))
        z = features(params, xb, mb, x.n_layers)
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
    n_layers = x.n_layers
    d = x.dim // n_layers
    g = torch.Generator().manual_seed(seed)
    mix = torch.nn.Parameter(torch.zeros(n_layers))
    q = torch.nn.Parameter(torch.empty(N_QUERIES, d).normal_(0.0, 0.02, generator=g))
    layers: list[torch.nn.Module] = []
    width = (N_QUERIES + 1) * d
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
        [mix, q, *net.parameters()], lr=config.lr * 0.3, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(seed)
    n = len(xs)
    for _ in range(config.epochs):
        order = rng.permutation(n)
        for i in range(0, n, BATCH):
            idx = order[i : i + BATCH]
            xb, mb = xs.padded(idx)
            xt = torch.from_numpy(xb).view(len(idx), xb.shape[1], n_layers, d)
            xt = torch.nn.functional.layer_norm(xt, (d,), eps=1e-5)
            h = torch.einsum("bLld,l->bLd", xt, torch.softmax(mix, 0))
            m = torch.from_numpy(mb)
            s = torch.einsum("bLd,kd->bkL", h, q).masked_fill(~m[:, None, :], float("-inf"))
            a = torch.softmax(s, dim=-1)
            att = torch.einsum("bkL,bLd->bkd", a, h).reshape(len(idx), -1)
            mean = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
            logits = net(torch.cat([att, mean], dim=1))
            loss = loss_fn(logits, torch.from_numpy(ys[idx]).long())
            opt.zero_grad()
            loss.backward()
            opt.step()
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    params = {"mix": mix.detach().numpy().copy(), "q": q.detach().numpy().copy()}
    if config.hidden:
        params["w1"] = linears[0].weight.detach().numpy().T.copy()
        params["b1"] = linears[0].bias.detach().numpy().copy()
    params["w"] = linears[-1].weight.detach().numpy().T.copy()
    params["b"] = linears[-1].bias.detach().numpy().copy()
    return Head(n_classes, config, params, kind="tokens")
