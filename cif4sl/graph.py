"""Sparse functional evidence and propagation over the training SL graph."""

import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch


class _Progress:
    def __init__(self, phase, total, unit="items"):
        self.phase = str(phase)
        self.total = int(total)
        self.unit = str(unit)
        self.started = self.last = time.monotonic()
        print(f"[CIF4SL] {self.phase}: 0/{self.total:,} {self.unit}, starting", flush=True)

    def update(self, done, detail="", force=False):
        now = time.monotonic()
        done = int(done)
        if not force and done < self.total and (now - self.last < 5):
            return
        elapsed = now - self.started
        eta = elapsed * (self.total - done) / done if done > 0 else 0.0
        pct = 100.0 * done / max(self.total, 1)
        print(
            f"[CIF4SL] {self.phase}: {done:,}/{self.total:,} {self.unit} ({pct:.1f}%) | elapsed={elapsed:.1f}s | ETA≈{eta:.1f}s {detail}",
            flush=True,
        )
        self.last = now


def sl_adjacency(train, n_genes):
    """Known positive SL edges from the current training fold."""
    positive = train.loc[train.label.eq(1), ["u", "v"]].to_numpy(dtype=np.int64)
    if not len(positive):
        raise ValueError("The training fold has no positive SL edges")
    positive = positive[positive[:, 0] != positive[:, 1]]
    u, v = positive.T
    adjacency = sp.csr_matrix(
        (np.ones(2 * len(positive), dtype=np.float32), (np.r_[u, v], np.r_[v, u])),
        shape=(int(n_genes), int(n_genes)),
    )
    adjacency.data[:] = 1.0
    adjacency.eliminate_zeros()
    return adjacency


def two_hop_operator(adjacency_matrix):
    a = adjacency_matrix.tocsr().astype(np.float32)
    degree = np.asarray(a.sum(axis=1)).ravel().astype(np.float32)
    inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    return (a @ sp.diags(inv_degree) @ a).tocsr().astype(np.float32)


def sparsemax_row(lower, method, temperature):
    if method != "sparsemax":
        raise ValueError("CIF4SL requires sparsemax projection")
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Sparsemax temperature must be finite and positive")
    lower = np.asarray(lower, dtype=np.float64)
    if lower.ndim != 1 or not np.isfinite(lower).all() or (lower < 0).any():
        raise ValueError("L must be a finite, nonnegative one-dimensional array")
    output = np.zeros_like(lower)
    indices = np.flatnonzero(lower > 0)
    if not len(indices):
        return output
    values = lower[indices]
    gap = values.max() - values
    possible = gap < temperature
    indices = indices[possible]
    z = -gap[possible] / temperature
    ordered = np.sort(z)[::-1]
    count = np.arange(1, len(ordered) + 1, dtype=np.float64)
    cumulative = np.cumsum(ordered)
    active = 1 + count * ordered > cumulative
    size = int(np.flatnonzero(active)[-1]) + 1
    threshold = (cumulative[size - 1] - 1) / size
    weights = np.maximum(z - threshold, 0)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Sparsemax produced invalid weights")
    output[indices] = weights / total
    return output


def sparsemax_batch(lower, temperature):
    values = lower.double()
    positive = values > 0
    nonempty = positive.any(dim=1, keepdim=True)
    maximum = values.masked_fill(~positive, -torch.inf).amax(dim=1, keepdim=True)
    maximum = torch.where(nonempty, maximum, torch.zeros_like(maximum))
    gap = maximum - values
    possible = positive & (gap < temperature)
    z = torch.where(possible, -gap / temperature, torch.full_like(values, -2.0))
    ordered = z.sort(dim=1, descending=True).values
    count = torch.arange(1, lower.shape[1] + 1, dtype=z.dtype, device=z.device)[None, :]
    cumulative = ordered.cumsum(dim=1)
    support = (1 + count * ordered > cumulative) & (ordered > -2)
    size = support.sum(dim=1, keepdim=True).clamp_min(1)
    threshold = (cumulative.gather(1, size - 1) - 1) / size
    weights = torch.where(possible, (z - threshold).clamp_min(0), torch.zeros_like(z))
    total = weights.sum(dim=1, keepdim=True)
    total = torch.where(nonempty, total, torch.ones_like(total))
    return (weights / total).to(lower.dtype)


def attach_support(model, support):
    support = support.tocsr().copy()
    support.eliminate_zeros()
    support.sort_indices()
    if support.shape != (model.n, model.n) or np.any(support.diagonal()):
        raise ValueError("Fixed support must match the gene catalog and exclude self loops")
    coo = support.tocoo()
    indices = torch.as_tensor(
        np.stack([coo.row, coo.col]), dtype=torch.long, device=model.embedding.weight.device
    )
    model.register_buffer("evidence_support", indices)


def support_lower_bounds(model, pair_batch_size):
    pairs = model.evidence_support.T
    if not len(pairs):
        return model.embedding.weight.reshape(-1)[:0]
    pieces = []
    for start in range(0, len(pairs), pair_batch_size):
        batch = pairs[start : start + pair_batch_size]
        pieces.append(model.lower_bound(batch))
    return torch.cat(pieces)


def _support_layout(model, row_block_size):
    key = (row_block_size, str(model.evidence_support.device))
    cache = getattr(model, "_support_layout_cache", {})
    if key not in cache:
        indices = model.evidence_support.detach().cpu().numpy()
        counts = np.bincount(indices[0], minlength=model.n)
        indptr = np.r_[0, np.cumsum(counts)]
        layout = []
        for start in range(0, model.n, row_block_size):
            stop = min(start + row_block_size, model.n)
            left, right = (int(indptr[start]), int(indptr[stop]))
            if left == right:
                continue
            width = int(counts[start:stop].max())
            rows = indices[0, left:right]
            offset = np.arange(left, right) - indptr[rows]
            slots = torch.as_tensor(
                (rows - start) * width + offset,
                dtype=torch.long,
                device=model.evidence_support.device,
            )
            layout.append((left, right, stop - start, width, slots))
        cache[key] = layout
        model._support_layout_cache = cache
    return cache[key]


def project_support(model, lower, config):
    if not lower.numel():
        return lower * 0
    pieces = []
    for left, right, rows, width, slots in _support_layout(model, config["graph_row_block_size"]):
        padded = lower.new_zeros(rows * width).scatter(0, slots, lower[left:right])
        weights = sparsemax_batch(padded.reshape(rows, width), config["sparsemax_temperature"])
        pieces.append(weights.reshape(-1)[slots])
    return torch.cat(pieces)


def normalize_evidence(graph):
    b = graph.tocsr().astype(np.float64)
    b.eliminate_zeros()
    if (
        (b != b.T).nnz
        or np.any(b.diagonal() != 0)
        or not np.isfinite(b.data).all()
        or np.any(b.data < 0)
    ):
        raise ValueError("Functional evidence must be symmetric, finite, nonnegative and loop-free")
    degree = np.asarray(b.sum(axis=1)).ravel()
    weight = np.zeros_like(degree)
    occupied = degree > 0
    weight[occupied] = degree[occupied] ** (-1.0)
    return (b @ sp.diags(weight)).T.tocsr().astype(np.float32)


def propagate(topology, graph):
    topology = topology.tocsr().astype(np.float32)
    if topology.shape != graph.shape:
        raise ValueError("Topology and functional evidence must have the same shape")
    evidence = normalize_evidence(graph)
    return (topology @ evidence).tocsr().astype(np.float32)


@torch.no_grad()
def export_evidence(model, settings, out, capture_support=False):
    model.eval()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = model.n
    chunk = settings["pair_batch_size"]
    temperature = settings["sparsemax_temperature"]
    device = model.embedding.weight.device
    if hasattr(model, "evidence_support"):
        lower = support_lower_bounds(model, settings["graph_pair_batch_size"])
        weights = project_support(model, lower.double(), settings)
        indices = model.evidence_support.detach().cpu().numpy()
        directed = sp.csr_matrix((weights.cpu().numpy(), (indices[0], indices[1])), shape=(n, n))
        directed.eliminate_zeros()
    else:
        columns, values, indptr = [], [], [0]
        progress = _Progress("Construct functional evidence", n, "rows")
        for u in range(n):
            lower = np.zeros(n, dtype=np.float64)
            for start in range(0, n, chunk):
                v = np.arange(start, min(start + chunk, n), dtype=np.int64)
                pairs = torch.as_tensor(np.column_stack((np.full(len(v), u), v)), device=device)
                batch = model.lower_bound(pairs)
                lower[start : start + len(v)] = batch.cpu().numpy()
            lower[u] = 0.0
            weights = sparsemax_row(lower, "sparsemax", temperature)
            selected = np.flatnonzero(weights > 0)
            columns.append(selected.astype(np.int32))
            values.append(weights[selected])
            indptr.append(indptr[-1] + len(selected))
            progress.update(u + 1, f"directed entries={indptr[-1]:,}")
        directed = sp.csr_matrix(
            (np.concatenate(values), np.concatenate(columns), np.asarray(indptr)), shape=(n, n)
        )
    if capture_support:
        sp.save_npz(out / "support.npz", (directed > 0).astype(np.uint8))
    graph = ((directed + directed.T) * 0.5).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    if not np.isfinite(graph.data).all() or (graph.data < 0).any():
        raise FloatingPointError("Functional evidence contains invalid weights")
    sp.save_npz(out / "L_sym.npz", graph)
    return graph, {
        "genes": n,
        "directed_entries": int(directed.nnz),
        "undirected_edges": int(graph.nnz // 2),
        "fixed_support": hasattr(model, "evidence_support"),
    }
