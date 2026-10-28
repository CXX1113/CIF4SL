"""Fold-safe measured responses and their supervision weights."""

import json
import numpy as np
from .model import validate_pairs


def fold_safe_responses(n_genes, measured, parts):
    required = {"u", "v", "y11", "y10", "y01", "selection_scope"}
    missing = required.difference(measured.columns)
    if missing:
        raise ValueError("Missing response columns: " + ", ".join(sorted(missing)))
    allowed_scopes = {"sldb_positive_subset", "independent_screen"}
    observed_scopes = set(measured.selection_scope.astype(str).unique())
    if not observed_scopes.issubset(allowed_scopes):
        raise ValueError("Unknown selection_scope in measured responses")
    train_positive = (
        parts["train"].loc[parts["train"].label.eq(1), ["u", "v"]].to_numpy(dtype=np.int64)
    )
    train_keys = set(map(tuple, np.sort(train_positive, axis=1)))
    ids = measured[["u", "v"]].to_numpy(dtype=np.int64)
    if ((ids < 0) | (ids >= int(n_genes))).any():
        raise ValueError("Measured response IDs are outside the gene catalog")
    canonical = np.sort(ids, axis=1)
    admissible = np.fromiter(
        (tuple(pair) in train_keys for pair in canonical), dtype=bool, count=len(canonical)
    )
    conditional = measured.selection_scope.astype(str).to_numpy() == "sldb_positive_subset"
    chosen = measured.loc[~conditional | admissible].copy()
    heldout = set()
    for split_name in ("valid", "test"):
        if split_name in parts:
            heldout.update(
                map(tuple, np.sort(parts[split_name][["u", "v"]].to_numpy(dtype=np.int64), axis=1))
            )
    chosen_keys = np.sort(chosen[["u", "v"]].to_numpy(dtype=np.int64), axis=1)
    keep = np.fromiter(
        (tuple(pair) not in heldout for pair in chosen_keys), dtype=bool, count=len(chosen_keys)
    )
    removed_heldout = int((~keep).sum())
    chosen = chosen.loc[keep].copy()
    audit = {
        "source_rows_before_fold_filter": int(len(measured)),
        "source_rows_after_fold_filter": int(len(chosen)),
        "removed_rows_selected_by_heldout_SL_labels": int((conditional & ~admissible).sum()),
        "heldout_response_rows_removed": removed_heldout,
        "included_rows_selected_by_train_SL_labels": int((conditional & admissible).sum()),
        "included_rows_independent_screen": int((~conditional).sum()),
        "conditional_source_includes_validation_or_test_pairs": False,
    }
    return (chosen, audit)


def training_pairs(train, n):
    if n < 2 or not {"u", "v"}.issubset(train.columns):
        raise ValueError("Expected n_genes >= 2 and training columns u,v")
    pairs = np.sort(validate_pairs(train[["u", "v"]], n, False), axis=1)
    labels = train["label"].to_numpy() if "label" in train else np.ones(len(train))
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("SL training labels must be 0 or 1")
    pos = np.unique(pairs[labels == 1], axis=0)
    neg = np.unique(pairs[labels == 0], axis=0)
    if len(pos) == 0:
        raise ValueError("The training graph has no positive edges")
    if np.intersect1d(pos[:, 0] * n + pos[:, 1], neg[:, 0] * n + neg[:, 1]).size:
        raise ValueError("Conflicting labels for the same unordered pair")
    return (pos, neg)


def prepare_responses(measured, parts, n_genes, settings):
    rows, audit = fold_safe_responses(n_genes, measured, parts)
    context_column = str(settings["context_column"])
    if context_column not in rows or rows[context_column].isna().any():
        raise ValueError(f"Measured responses require valid {context_column} values")
    for column in ("evidence_type", "arm_source11", "arm_source10", "arm_source01"):
        if column in rows:
            bad = rows[column].dropna().astype(str).str.lower().ne("measured")
            if bad.any():
                raise ValueError(
                    f"Only measured responses are allowed; {column} contains non-measured records"
                )
    ids = rows[["u", "v"]].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(ids).all()
        or not np.equal(ids, np.floor(ids)).all()
        or (ids[:, 0] == ids[:, 1]).any()
    ):
        raise ValueError("Measured pairs must contain distinct integer gene IDs")
    values = rows[["y11", "y10", "y01"]].to_numpy(dtype=np.float64)
    present = ~np.isnan(values)
    if (
        not np.isfinite(values[present]).all()
        or ((values[present] < 0) | (values[present] > 1)).any()
    ):
        raise ValueError("Measured y11/y10/y01 must be fractions in [0,1]")
    counts = np.column_stack(
        [
            rows["n" + arm].to_numpy(dtype=float) if "n" + arm in rows else np.ones(len(rows))
            for arm in ("11", "10", "01")
        ]
    )
    if not np.isfinite(counts[present]).all() or (counts[present] <= 0).any():
        raise ValueError("Observed responses require finite positive measurement counts")
    counts[~present] = 0
    reverse = ids[:, 0] > ids[:, 1]
    values[reverse] = values[reverse][:, [0, 2, 1]]
    counts[reverse] = counts[reverse][:, [0, 2, 1]]
    rows = rows.copy()
    rows[["u", "v"]] = np.sort(ids.astype(np.int64), axis=1)
    rows[["y11", "y10", "y01"]] = values
    rows[["n11", "n10", "n01"]] = counts
    rows["study"] = (
        rows["study"].fillna("unspecified").astype(str) if "study" in rows else "unspecified"
    )
    rows["context"] = [
        json.dumps([study, str(context)], ensure_ascii=False)
        for study, context in zip(rows.study, rows[context_column])
    ]
    rows = rows.dropna(subset=["y11", "y10", "y01"], how="all")
    kept_counts = rows.selection_scope.value_counts().to_dict()
    rows = rows.groupby(["u", "v", "study", "context"], as_index=False).agg(
        {
            **{"y" + arm: "mean" for arm in ("11", "10", "01")},
            **{"n" + arm: "max" for arm in ("11", "10", "01")},
        }
    )
    if rows.empty or rows[["y11", "y10", "y01"]].notna().sum().eq(0).any():
        raise ValueError("The fold lacks measured supervision for all three arms")
    weights = np.zeros((len(rows), 3), dtype=np.float32)
    for arm_index, arm in enumerate(("11", "10", "01")):
        observed = rows["y" + arm].notna()
        studies = rows.loc[observed, "study"].unique()
        for study in studies:
            contexts = rows.loc[observed & rows.study.eq(study), "context"].unique()
            for context in contexts:
                indices = rows.index[observed & rows.study.eq(study) & rows.context.eq(context)]
                w = np.sqrt(rows.loc[indices, "n" + arm].to_numpy(dtype=float))
                weights[indices, arm_index] = w / w.sum() / len(studies) / len(contexts) / 3
    context_names = sorted(rows.context.unique().tolist())
    context_to_id = {name: index for index, name in enumerate(context_names)}
    audit.update(
        source_rows_after_pair_context_aggregation=len(rows),
        response_unique_pairs=int(rows[["u", "v"]].drop_duplicates().shape[0]),
        response_context_count=len(context_names),
        response_study_count=int(rows.study.nunique()),
        included_rows_by_scope_after_heldout_filter=kept_counts,
        response_targets="measured threshold fractions, no additional binarization",
        response_weights="equal study / context / arm mass; within-context sqrt(n) reliability",
        response_duplicate_aggregation="mean y and max n, not summed independent trial counts",
        heldout_pair_response_results_used=False,
    )
    return (rows, context_names, context_to_id, weights, audit)
