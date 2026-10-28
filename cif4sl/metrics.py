"""Classification and full-catalog ranking metrics."""

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    f1_score,
)


def validation_threshold(y, score):
    y, score = (np.asarray(y), np.asarray(score))
    if len(np.unique(y)) != 2:
        raise ValueError("Validation set must contain both classes")
    p, r, t = precision_recall_curve(y, score)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=p + r > 0)[:-1]
    return float(t[np.flatnonzero(f == f.max())[-1]])


def classification(y, score, threshold):
    y, score = (np.asarray(y, dtype=int), np.asarray(score, dtype=float))
    if len(np.unique(y)) != 2:
        raise ValueError("Test set must contain both classes")
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite prediction")
    return dict(
        ACC=float(accuracy_score(y, score >= threshold)),
        F1=float(f1_score(y, score >= threshold, zero_division=0)),
        threshold=float(threshold),
    )


def topk_metrics(ids, positive, n_candidates, ks):
    y = np.isin(ids, list(positive)).astype(float)
    relevant = len(positive)
    row = {"n_candidates": int(n_candidates), "n_relevant": relevant}
    for k in ks:
        hits = y[:k]
        count = float(hits.sum())
        ideal = (1 / np.log2(np.arange(2, min(k, relevant) + 2))).sum()
        row[f"NDCG@{k}"] = float((hits / np.log2(np.arange(2, len(hits) + 2))).sum() / ideal)
        row[f"Recall@{k}"] = count / relevant
    return row
