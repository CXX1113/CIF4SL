"""Select complete measured three-arm records for split audits."""

import numpy as np
import pandas as pd


def _keys(frame):
    a = frame[["u", "v"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = (
        np.isfinite(a).all(1) & (a >= 0).all(1) & (a == np.floor(a)).all(1) & (a[:, 0] != a[:, 1])
    )
    keys = [tuple(sorted(map(int, p))) if ok else None for p, ok in zip(a, valid)]
    return pd.Series(keys, index=frame.index, dtype=object)


def complete_measured_mask(rows, source):
    sl_keys = set(_keys(source).dropna())
    required = ["u", "v", "context", "evidence_type"] + [
        c + a for a in ("11", "10", "01") for c in ("y", "n", "arm_source")
    ]
    absent = set(required) - set(rows)
    if absent:
        raise ValueError(f"Measured response table missing columns: {sorted(absent)}")
    values = rows[["y11", "y10", "y01"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    counts = rows[["n11", "n10", "n01"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    complete = np.isfinite(values).all(1) & (values >= 0).all(1) & (values <= 1).all(1)
    complete &= np.isfinite(counts).all(1) & (counts > 0).all(1)
    measured = rows.evidence_type.eq("measured")
    for arm in ("11", "10", "01"):
        measured &= rows["arm_source" + arm].eq("measured")
    row_keys = _keys(rows)
    valid = (
        complete
        & measured
        & row_keys.isin(sl_keys)
        & rows.context.notna()
        & rows.context.astype(str).str.strip().ne("")
    )
    return valid
