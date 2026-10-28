"""Pair splits and negative sampling; catalog IDs retain their original order."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def canonical_pairs(df):
    df = df.copy()
    u, v = (df.u.to_numpy(), df.v.to_numpy())
    swap = u > v
    df["u"], df["v"] = (np.minimum(u, v), np.maximum(u, v))
    for a, b in [("y10", "y01"), ("n10", "n01"), ("mean10", "mean01")]:
        if a in df and b in df:
            old = df[a].copy()
            df.loc[swap, a] = df.loc[swap, b]
            df.loc[swap, b] = old[swap]
    return df[df.u != df.v].reset_index(drop=True)


def keys(frame):
    return set(map(tuple, frame[["u", "v"]].to_numpy(int)))


def sample_negatives(genes, blocked, count, rng):
    genes = np.sort(np.unique(genes)).astype(int)
    gene_set = set(genes)
    occupied = sum((u in gene_set and v in gene_set for u, v in blocked))
    capacity = len(genes) * (len(genes) - 1) // 2 - occupied
    if count > capacity:
        raise ValueError(
            f"Negative sampling requests {count} distinct pairs but pool capacity is {capacity}; change source/ratio"
        )
    if count == 0:
        return pd.DataFrame(columns=["u", "v"], dtype=int)
    if capacity < 4 * count or len(genes) < 200:
        available = [
            (int(u), int(v))
            for i, u in enumerate(genes)
            for v in genes[i + 1 :]
            if (int(u), int(v)) not in blocked
        ]
        chosen = [available[i] for i in rng.choice(len(available), count, replace=False)]
    else:
        chosen = set()
        while len(chosen) < count:
            a = rng.choice(genes, size=(max(1024, 2 * (count - len(chosen))), 2))
            a.sort(axis=1)
            for u, v in a:
                pair = (int(u), int(v))
                if u != v and pair not in blocked:
                    chosen.add(pair)
                if len(chosen) == count:
                    break
    return pd.DataFrame(sorted(chosen), columns=["u", "v"])


def validate_config(cfg):
    s = cfg.get("split", {})
    if s.get("cv") not in ("CV1", "CV2", "CV3"):
        raise ValueError("split.cv: CV1/CV2/CV3 required")
    if s.get("dataset_variant") == "independent" and s.get("cv") != "CV1":
        raise ValueError(
            "Independent source holdout is not CV2/CV3; use CV1 as the config placeholder and read external_test_repeats_inner_holdout"
        )
    if s.get("negative_strategy") not in ("Rand", "Exp", "Dep"):
        raise ValueError("negative_strategy: Rand/Exp/Dep required")
    if s.get("negative_ratio") not in (1, 5, 20, 50):
        raise ValueError("SLB negative_ratio: 1/5/20/50")
    if s.get("dataset_variant", "full") not in (
        "full",
        "without_computational",
        "independent",
        "context_specific",
    ):
        raise ValueError("Unknown SLB dataset_variant")
    if type(cfg.get("folds", 5)) is not int or cfg.get("folds", 5) < 2:
        raise ValueError("SLB requires folds>=2")
    if not 0 < float(s.get("validation_fraction", 0.1)) < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    if s.get("validation_mode", "pair_holdout") not in ("pair_holdout", "gene_holdout"):
        raise ValueError("Unknown validation_mode")
    if cfg.get("subset_genes_for_smoke", False):
        raise ValueError("SLB preserves the fixed gene catalog; do not subset genes")
    if type(s.get("ranking_block_size", 1024)) is not int or s.get("ranking_block_size", 1024) < 1:
        raise ValueError("ranking_block_size must be positive")
    if s.get("catalog_filter", "train_valid") not in ("train_valid", "all_known_non_test"):
        raise ValueError("Unknown catalog_filter")
    if s.get("catalog_scope", "all_genes") not in ("all_genes", "split_eligible"):
        raise ValueError("Unknown catalog_scope")
    if s.get("ranking_engine", "exact_pruned") not in ("exhaustive", "exact_pruned"):
        raise ValueError("Unknown ranking_engine")
    if s.get("split_source", "generated") not in ("generated", "imported"):
        raise ValueError("Unknown split_source")
    if cfg.get("gene_universe", "all_curated_genes") != "all_curated_genes":
        raise ValueError("SLB requires fixed catalog")


def read_pairs(path, n, labels=False):
    f = pd.read_csv(path)
    required = ["u", "v"] + (["label"] if labels else [])
    if not set(required) <= set(f):
        raise ValueError(f"{path}: require {required}, using curated numeric gene IDs")
    if (
        not np.isfinite(f[["u", "v"]].to_numpy(float)).all()
        or not np.equal(f[["u", "v"]], np.floor(f[["u", "v"]])).all().all()
    ):
        raise ValueError(f"{path}: gene IDs must be finite integers")
    f[["u", "v"]] = np.sort(f[["u", "v"]].to_numpy(int), axis=1)
    if not ((f.u >= 0) & (f.u < f.v) & (f.v < n)).all():
        raise ValueError(f"{path}: invalid pair IDs")
    if f.duplicated(["u", "v"]).any():
        raise ValueError(f"{path}: duplicate canonical pairs")
    if labels and (not f.label.isin([0, 1]).all()):
        raise ValueError(f"{path}: binary labels required")
    return f[required].sort_values(["u", "v"]).reset_index(drop=True)


def dataset_positives(data, cfg):
    s = cfg["split"]
    variant = s.get("dataset_variant", "full")
    if s.get("positive_pairs"):
        pos = read_pairs(s["positive_pairs"], data.n_genes)
    else:
        pos = data.sl[["u", "v"]].copy()
    if variant == "context_specific" and (not s.get("positive_pairs")):
        raise ValueError(
            "context_specific requires a sourced cell-specific positive_pairs table; global SLDB labels are insufficient"
        )
    excluded = pd.DataFrame(columns=["u", "v"])
    if variant == "without_computational":
        if not s.get("computational_pairs"):
            raise FileNotFoundError(
                "without_computational requires computational_pairs from source provenance, not response availability"
            )
        excluded = read_pairs(s["computational_pairs"], data.n_genes)
        excluded_keys = keys(excluded)
        pos = pos.loc[[p not in excluded_keys for p in pos.itertuples(index=False, name=None)]]
    return (pos.sort_values(["u", "v"]).reset_index(drop=True), excluded)


def select_correlation_negatives(scores, n, count, blocked, strategy):
    a = np.asarray(scores)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("negative_scores must have columns u,v,Pearson r")
    if not np.isfinite(a).all() or (np.abs(a[:, 2]) > 1 + 1e-06).any():
        raise ValueError("Invalid Pearson scores")
    uv = np.sort(a[:, :2].astype(np.int64), axis=1)
    if (
        not np.equal(a[:, :2], a[:, :2].astype(np.int64)).all()
        or (uv[:, 0] < 0).any()
        or (uv[:, 0] >= uv[:, 1]).any()
        or (uv[:, 1] >= n).any()
    ):
        raise ValueError("Invalid correlation pair IDs")
    code = uv[:, 0] * n + uv[:, 1]
    if len(np.unique(code)) != len(code):
        raise ValueError("Duplicate correlation pairs")
    eligible = ~np.isin(code, np.array([u * n + v for u, v in blocked], dtype=np.int64))
    idx = np.flatnonzero(eligible)
    if strategy == "Exp":
        priority = a[idx, 2]
    else:
        priority = -np.abs(a[idx, 2])
    idx = idx[np.lexsort((uv[idx, 1], uv[idx, 0], priority))]
    seen = np.zeros(n, bool)
    initial_pairs = []
    taken = np.zeros(len(a), bool)
    for i in idx:
        u, v = uv[i]
        if not (seen[u] and seen[v]):
            initial_pairs.append(i)
            taken[i] = True
            seen[u] = seen[v] = True
        if seen.all():
            break
    if not seen.all():
        raise ValueError(
            f"Correlation resource cannot cover {int((~seen).sum())} catalog genes; supply complete source or an explicitly remapped catalog"
        )
    if len(initial_pairs) > count or len(idx) < count:
        raise ValueError("The negative pool cannot meet the requested size and include every gene")
    fill = idx[~taken[idx]][: count - len(initial_pairs)]
    return pd.DataFrame(uv[np.r_[initial_pairs, fill]], columns=["u", "v"])


def negative_pool(data, cfg, pos, seed, extra_blocked=()):
    s = cfg["split"]
    n = data.n_genes
    count = len(pos) * int(s["negative_ratio"])
    blocked = keys(data.sl) | keys(pos) | set(extra_blocked)
    if s.get("negative_pool"):
        pool = read_pairs(s["negative_pool"], n)
        if keys(pool) & blocked:
            raise ValueError("Imported negative_pool includes known positives/external holdout")
        if len(pool) != count:
            raise ValueError(f"negative_pool requires exactly {count} pairs; supplied {len(pool)}")
        return pool
    if s["negative_strategy"] == "Rand":
        arr = sample_negatives(np.arange(n), blocked, count, np.random.default_rng(seed + 1000003))
        return pd.DataFrame(arr, columns=["u", "v"])
    path = s.get("negative_scores")
    if not path:
        raise FileNotFoundError(
            f"{s['negative_strategy']} needs negative_scores (.npy raw Pearson u,v,r) or negative_pool. The selected CCLE response-context matrix is not SLB DepMap 22Q4."
        )
    return select_correlation_negatives(
        np.load(path, mmap_mode="r", allow_pickle=False), n, count, blocked, s["negative_strategy"]
    )


def _eligible(f, held, mode):
    a = f.u.isin(held).to_numpy()
    b = f.v.isin(held).to_numpy()
    return (~(a | b), a ^ b if mode == "CV2" else a & b)


def _inner(frame, seed, fraction, mode, cv, seen):
    if mode == "pair_holdout" or cv == "CV1":
        tr = []
        va = []
        for label in (0, 1):
            ids = np.flatnonzero(frame.label.to_numpy() == label)
            a, b = train_test_split(ids, test_size=fraction, random_state=seed + label)
            tr.extend(a)
            va.extend(b)
        return (frame.iloc[tr], frame.iloc[va], [])
    rng = np.random.default_rng(seed)
    ids = rng.permutation(seen)
    held = ids[: max(1, int(round(len(ids) * fraction)))]
    tr, va = _eligible(frame, held, cv)
    return (frame.loc[tr], frame.loc[va], held.tolist())


def build_split(data, cfg, seed, fold):
    validate_config(cfg)
    s = cfg["split"]
    n = data.n_genes
    cache_key = json.dumps({"s": s, "seed": seed, "folds": cfg["folds"]}, sort_keys=True)
    cache = getattr(data, "_slb_cache", {})
    if cache_key not in cache:
        pos, excluded = dataset_positives(data, cfg)
        external = None
        if s.get("dataset_variant") == "independent":
            if not s.get("external_test"):
                raise FileNotFoundError(
                    "independent requires an experimentally sourced external_test u,v,label table"
                )
            external = read_pairs(s["external_test"], n, True)
            extkeys = keys(external)
            pos = pos.loc[[p not in extkeys for p in pos.itertuples(index=False, name=None)]]
        if s.get("split_source") == "imported":
            neg = None
        else:
            neg = negative_pool(
                data, cfg, pos, seed, keys(external) if external is not None else ()
            )
        cache[cache_key] = (pos, excluded, neg, external)
        data._slb_cache = cache
    pos, excluded, neg, external = cache[cache_key]
    held = []
    inner_held = []
    if s.get("split_source") == "imported":
        root = Path(s["split_directory"]) / f"seed_{seed}" / f"fold_{fold}"
        parts = {k: read_pairs(root / f"{k}.csv", n, True) for k in ("train", "valid", "test")}
        if s["cv"] != "CV1":
            seen = set(parts["train"][["u", "v"]].to_numpy().ravel())
            held = sorted(set(range(n)) - seen)
    else:
        all_rows = pd.concat([pos.assign(label=1), neg.assign(label=0)], ignore_index=True)
        if external is not None:
            outer = all_rows
            test = external
        elif s["cv"] == "CV1":
            train_idx = []
            test_idx = []
            for label in (1, 0):
                idx = np.flatnonzero(all_rows.label.to_numpy() == label)
                tr, te = list(KFold(cfg["folds"], shuffle=True, random_state=seed).split(idx))[fold]
                train_idx.extend(idx[tr])
                test_idx.extend(idx[te])
            outer = all_rows.iloc[train_idx]
            test = all_rows.iloc[test_idx]
        else:
            seen_idx, held_idx = list(
                KFold(cfg["folds"], shuffle=True, random_state=seed).split(np.arange(n))
            )[fold]
            held = held_idx.tolist()
            tr, te = _eligible(all_rows, held, s["cv"])
            outer = all_rows.loc[tr]
            test = all_rows.loc[te]
        seen = np.setdiff1d(np.arange(n), held)
        tr, va, inner_held = _inner(
            outer,
            seed + fold + 31337,
            float(s.get("validation_fraction", 0.1)),
            s.get("validation_mode", "pair_holdout"),
            s["cv"],
            seen,
        )
        parts = {"train": tr, "valid": va, "test": test}
    parts = {
        k: f[["u", "v", "label"]].sort_values(["u", "v"]).reset_index(drop=True)
        for k, f in parts.items()
    }
    allowed_pos = keys(pos) | (
        keys(external.loc[external.label.eq(1)]) if external is not None else set()
    )
    for name, f in parts.items():
        if f.empty or set(f.label) != {0, 1}:
            raise ValueError(f"SLB {name} has empty/one-class labels")
        if not keys(f.loc[f.label.eq(1)]) <= allowed_pos:
            raise ValueError("Imported positive labels absent from declared source")
        if keys(f.loc[f.label.eq(0)]) & (keys(data.sl) | allowed_pos):
            raise ValueError("Known positive assigned a negative label")
    sets = [keys(parts[k]) for k in ("train", "valid", "test")]
    if any((sets[i] & sets[j] for i in range(3) for j in range(i))):
        raise ValueError("SLB pair leakage")
    if s["cv"] != "CV1" and external is None:
        for name in ("train", "valid"):
            tr, _ = _eligible(parts[name], held, s["cv"])
            if not tr.all():
                raise ValueError("Cold test genes used for supervision/validation")
        _, te = _eligible(parts["test"], held, s["cv"])
        if not te.all():
            raise ValueError("Imported test violates CV gene partition")
    responses = {
        k: data.observations.loc[
            [p in sets[i] for p in data.observations[["u", "v"]].itertuples(index=False, name=None)]
        ].copy()
        for i, k in enumerate(("train", "valid"))
    }
    if any((f.empty for f in responses.values())):
        raise ValueError(
            "No measured response train/validation intersection in this split. Change a declared seed/fraction; do not fill responses."
        )
    meta = {
        "protocol": "SLB",
        "seed": seed,
        "fold": fold,
        "cv": s["cv"],
        "negative_strategy": s["negative_strategy"],
        "negative_ratio_requested_global_pool": s["negative_ratio"],
        "dataset_variant": s.get("dataset_variant", "full"),
        "actual_negative_ratios": {
            k: float((f.label == 0).sum() / (f.label == 1).sum()) for k, f in parts.items()
        },
        "heldout_gene_ids": held,
        "inner_validation_gene_ids": inner_held,
        "fixed_background_nodes_visible": True,
        "strict_node_inductive": False,
        "cv2_cross_edges_can_be_test_in_two_folds": s["cv"] == "CV2",
        "external_test_repeats_inner_holdout": external is not None,
        "test_used_for_selection": False,
        "response_join": "SL train/valid intersection with measured complete triples",
        "validation_mode": s.get("validation_mode", "pair_holdout"),
    }
    return (parts, responses, meta, excluded)
