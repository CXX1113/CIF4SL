"""Symmetric ranking scores and full-catalog validation."""

from pathlib import Path
import numpy as np
import torch


class CIF4SLScorer:
    def __init__(self, representations):
        self.representations = np.asarray(representations, dtype=np.float32)
        self.n = int(self.representations.shape[0])

    def predict(self, pairs):
        pair_array = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        u, v = pair_array.T
        return np.log1p(
            np.maximum(0.5 * (self.representations[u, v] + self.representations[v, u]), 0.0)
        ).astype(np.float32)

    def rows(self, queries):
        q = np.asarray(queries, dtype=np.int64).reshape(-1)
        sym = 0.5 * (self.representations[q, :] + self.representations[:, q].T)
        return np.log1p(np.maximum(sym, 0.0)).astype(np.float32)


def validation_ndcg(scorer, train, valid):
    partners = {}
    for u, v in valid.loc[valid.label.eq(1), ["u", "v"]].to_numpy(dtype=np.int64):
        partners.setdefault(int(u), set()).add(int(v))
        partners.setdefault(int(v), set()).add(int(u))
    banned = {}
    for u, v in train[["u", "v"]].to_numpy(dtype=np.int64):
        banned.setdefault(int(u), set()).add(int(v))
        banned.setdefault(int(v), set()).add(int(u))
    query_ids = np.array(sorted(partners), dtype=np.int64)
    if not len(query_ids):
        raise ValueError("Validation requires at least one positive query for NDCG@10")
    score_matrix = scorer.rows(query_ids)
    all_nodes = np.arange(scorer.n, dtype=np.int64)
    values = []
    for row_index, query in enumerate(query_ids):
        excluded = banned.get(int(query), set()) | {int(query)}
        candidates = np.setdiff1d(all_nodes, list(excluded), assume_unique=False)
        scores = score_matrix[row_index]
        order = np.lexsort((candidates, -scores[candidates]))
        top_ids = candidates[order[:10]]
        hit = np.isin(top_ids, list(partners[int(query)])).astype(np.float64)
        discount = 1.0 / np.log2(np.arange(2, len(hit) + 2))
        ideal_len = min(10, len(partners[int(query)]))
        idcg = (1.0 / np.log2(np.arange(2, ideal_len + 2))).sum()
        values.append(float((hit * discount).sum() / idcg))
    return float(np.mean(values))


def load_scorer(path):
    path = Path(path)
    checkpoint = path / "model.pt" if path.is_dir() else path
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") != "cif4sl_scorer_v4":
        raise ValueError("Expected a CIF4SL v4 scorer checkpoint")
    return CIF4SLScorer(np.load(checkpoint.parent / saved["representation_file"]))
