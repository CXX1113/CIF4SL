"""Bliss-guided three-arm response predictor (paper Section 5.1)."""

from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _dropout_probability(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric and lie in [0,1)")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric and lie in [0,1)") from exc
    if not np.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f"{name} must be finite and lie in [0,1)")
    return value


def validate_pairs(pairs, n, allow_self=True):
    raw = np.asarray(pairs)
    if raw.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("pairs must have shape [N,2]")
    raw = raw.astype(np.float64)
    if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError("Gene IDs must be finite integers")
    if (raw < 0).any() or (raw >= n).any():
        raise ValueError("Gene IDs must lie in [0,n_genes)")
    ids = raw.astype(np.int64)
    if not allow_self and (ids[:, 0] == ids[:, 1]).any():
        raise ValueError("Training pairs must not contain self loops")
    return ids


class ResponseEncoder(nn.Module):
    def __init__(self, n_genes, dim=128, hidden_dim=128, node_features=None, dropout=0.0):
        super().__init__()
        self.dropout = _dropout_probability(dropout, "dropout")
        self._dropout_masks = {}
        self.n = int(n_genes)
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding = nn.Embedding(self.n, self.dim)
        nn.init.normal_(self.embedding.weight, std=0.5)
        if node_features is None:
            features = torch.empty(self.n, 0)
        else:
            features = torch.as_tensor(node_features, dtype=torch.float32)
            if (
                features.ndim != 2
                or len(features) != self.n
                or (not torch.isfinite(features).all())
            ):
                raise ValueError("node_features must be finite, two-dimensional and match n_genes")
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-06)
        self.register_buffer("node_features", features)
        self.feature_projection = (
            nn.Linear(features.shape[1], self.dim, bias=False) if features.shape[1] else None
        )
        self.shared_mlp = nn.Sequential(
            nn.Linear(4 * self.dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.arm_heads = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in range(3)])
        self.arm_skips = nn.ModuleList([nn.Linear(self.dim, 1, bias=False) for _ in range(3)])
        for head, skip, bias in zip(self.arm_heads, self.arm_skips, (0.0, -1.6, -1.6)):
            nn.init.normal_(head.weight, std=0.02)
            nn.init.constant_(head.bias, bias)
            nn.init.normal_(skip.weight, std=1 / np.sqrt(self.dim))

    def train(self, mode=True):
        self._dropout_masks = {}
        return super().train(mode)

    def _apply(self, fn, *args, **kwargs):
        self._dropout_masks = {}
        return super()._apply(fn, *args, **kwargs)

    def _drop_hidden(self, hidden, branch):
        if not self.training or self.dropout == 0:
            return hidden
        # Share the dropout mask across chunks of one training pass.
        mask = self._dropout_masks.get(branch)
        if mask is None:
            mask = F.dropout(hidden.new_ones(1, hidden.shape[-1]), self.dropout, True)
            self._dropout_masks[branch] = mask
        return hidden * mask

    def probabilities(self, pairs):
        return self(pairs).sigmoid()

    def raw_difference(self, pairs):
        p = self.probabilities(pairs)
        return p[:, 0] - (p[:, 1] + p[:, 2])

    def lower_bound(self, pairs):
        lower = self.raw_difference(pairs).clamp_min(0.0)
        return lower.masked_fill(pairs[:, 0] == pairs[:, 1], 0.0)


class CIF4SL(ResponseEncoder):
    def __init__(self, n_genes, contexts, config, node_features=None):
        super().__init__(
            n_genes,
            config["embedding_dim"],
            config["hidden_dim"],
            node_features,
            dropout=config["dropout"],
        )
        self.context_names = tuple(contexts)
        if not self.context_names:
            raise ValueError("At least one context is required")
        context_dim = config["context_dim"]
        self.context_embedding = nn.Embedding(len(contexts), context_dim)
        self.context_projection = nn.Linear(self.hidden_dim, 3 * context_dim, bias=False)
        nn.init.normal_(self.context_embedding.weight, std=0.1)
        nn.init.normal_(self.context_projection.weight, std=0.02)

    def _components(self, pairs):
        canonical = pairs.sort(dim=1).values
        u, v = canonical.unbind(1)
        hu, hv = (self.embedding(u), self.embedding(v))
        if self.feature_projection is not None:
            hu = hu + self.feature_projection(self.node_features[u])
            hv = hv + self.feature_projection(self.node_features[v])
        product = hu * hv
        hidden = self._drop_hidden(
            self.shared_mlp(torch.cat([hu, hv, product, (hu - hv).abs()], dim=1)), "pair"
        )
        base = torch.cat(
            [head(hidden) + skip(product) for head, skip in zip(self.arm_heads, self.arm_skips)],
            dim=1,
        )
        modulation = self.context_projection(hidden).reshape(len(pairs), 3, -1)
        return (base, modulation)

    def _raw_context_logits(self, pairs, contexts=None):
        base, modulation = self._components(pairs)
        if contexts is None:
            effect = torch.einsum("nkd,cd->nck", modulation, self.context_embedding.weight)
            return base[:, None, :] + effect
        if contexts.shape != (len(pairs),):
            raise ValueError("contexts and pairs must have the same length")
        effect = (modulation * self.context_embedding(contexts)[:, None, :]).sum(-1)
        return base + effect

    def _arm_logits(self, raw):
        a, b = (raw[..., 1], raw[..., 2])
        # Exact Bliss logit without clipping the single-arm probabilities.
        baseline_logit = torch.logsumexp(torch.stack([a, b, a + b], dim=-1), dim=-1)
        return torch.stack([baseline_logit + raw[..., 0], a, b], dim=-1)

    def forward(self, pairs, contexts=None):
        logits = self._arm_logits(self._raw_context_logits(pairs, contexts))
        if contexts is None:
            probabilities = logits.sigmoid()
            margin = probabilities[..., 0] - (probabilities[..., 1] + probabilities[..., 2])
            selected = margin.argmax(dim=1)
            logits = logits[torch.arange(len(pairs), device=pairs.device), selected]
        reverse = pairs[:, 0] > pairs[:, 1]
        return torch.where(reverse[:, None], logits[:, [0, 2, 1]], logits)


@torch.no_grad()
def predict_in_batches(model, pairs, chunk, kind="lower_bound"):
    values = []
    for start in range(0, len(pairs), chunk):
        batch = pairs[start : start + chunk]
        if kind == "probabilities":
            val = model.probabilities(batch)
        elif kind == "raw_difference":
            val = model.raw_difference(batch)
        else:
            val = model.lower_bound(batch)
        values.append(val)
    if values:
        return torch.cat(values, dim=0)
    shape = (0, 3) if kind == "probabilities" else (0,)
    return torch.empty(shape, device=model.embedding.weight.device)


def load_response_model(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != "cif4sl_response_v4":
        raise ValueError("Expected a CIF4SL v4 response checkpoint")
    state, settings = saved["model_state"], saved["settings"]
    features = state["node_features"]
    model = CIF4SL(
        saved["n_genes"], saved["contexts"], settings, features if features.shape[1] else None
    )
    if "evidence_support" in state:
        model.register_buffer("evidence_support", state["evidence_support"].clone())
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), settings, saved["training"]
