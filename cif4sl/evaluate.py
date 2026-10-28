"""Exact full-catalog evaluation with deterministic gene-ID tie breaking."""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import topk_metrics, classification, validation_threshold


def _query_subset(test, fraction, seed):
    positives = test.loc[test.label.eq(1), ["u", "v"]].to_numpy(int)
    degree = {}
    for u, v in positives:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    table = pd.DataFrame({"query": sorted(degree)})
    table["degree"] = table["query"].map(degree)
    if fraction == 1:
        return table
    if not 0 < fraction <= 1:
        raise ValueError("ranking_query_fraction must be in (0,1]")
    q = min(4, len(table))
    table["stratum"] = pd.qcut(
        table.degree.rank(method="first"), q=q, labels=False, duplicates="drop"
    )
    rng = np.random.default_rng(seed)
    picked = []
    for _, g in table.groupby("stratum", sort=True):
        count = max(1, int(round(len(g) * fraction)))
        ids = np.sort(rng.choice(g.index.to_numpy(), size=min(count, len(g)), replace=False))
        picked.extend(ids.tolist())
    return table.loc[sorted(picked), ["query", "degree"]].reset_index(drop=True)


DISPLAY_METRICS = [
    "NDCG@10",
    "NDCG@50",
    "Recall@10",
    "Recall@50",
    "F1",
    "ACC",
]


def evaluate(
    scorer, data, train, valid, test, cfg, out, query_fraction=1.0, query_seed=723, selection=False
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if selection:
        truth = valid
    else:
        truth = test
    if truth is None:
        raise ValueError("Missing evaluation labels")
    vscore = scorer.predict(valid[["u", "v"]].to_numpy(int))
    threshold = validation_threshold(valid.label, vscore)
    if selection:
        score = vscore
    else:
        score = scorer.predict(truth[["u", "v"]].to_numpy(int))
    metrics = classification(truth.label, score, threshold)
    pd.DataFrame({"u": truth.u, "v": truth.v, "label": truth.label, "score": score}).to_csv(
        out / "predictions.csv", index=False
    )
    if selection:
        blocked = train
    else:
        blocked = pd.concat([train, valid], ignore_index=True)
    partners, banned = ({}, {})
    for u, v in truth.loc[truth.label.eq(1), ["u", "v"]].to_numpy(int):
        partners.setdefault(u, set()).add(v)
        partners.setdefault(v, set()).add(u)
    for u, v in blocked[["u", "v"]].to_numpy(int):
        banned.setdefault(u, set()).add(v)
        banned.setdefault(v, set()).add(u)
    queries = _query_subset(truth, query_fraction, query_seed)["query"].to_numpy(int)
    digest, rows = (hashlib.sha256(), [])
    batch = cfg["compute"]["query_batch_size"]
    ks = cfg["ranking_ks"]
    for start in range(0, len(queries), batch):
        qs = queries[start : start + batch]
        scores = scorer.rows(qs)
        for q, values in zip(qs, scores):
            candidates = np.setdiff1d(np.arange(data.n_genes), list(banned.get(q, set()) | {q}))
            if not np.isfinite(values[candidates]).all():
                raise ValueError("Nonfinite catalog score")
            labels = np.isin(candidates, list(partners[q])).astype(int)
            digest.update(
                np.column_stack([np.full(len(candidates), q), candidates, labels])
                .astype("<i8")
                .tobytes()
            )
            ids = candidates[np.lexsort((candidates, -values[candidates]))[: max(ks)]]
            rows.append(
                {
                    "query": int(q),
                    **topk_metrics(ids, partners[q], len(candidates), ks),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No held-out positive queries")
    metrics.update({k: float(frame[k].mean()) for k in frame if "@" in k})
    frame.to_csv(out / "ranking_queries.csv", index=False)
    return (
        metrics,
        {
            "candidate_sha256": digest.hexdigest(),
            "n_queries": len(queries),
            "n_genes": data.n_genes,
            "query_fraction": query_fraction,
            "split": "validation" if selection else "test",
            "full_candidate_catalog": True,
            "filter": "train" if selection else "train_valid",
        },
    )
