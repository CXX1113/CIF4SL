"""Prepare benchmark features and measured responses without changing gene IDs."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import statistics
from .features import features
from .io import save_json, sha256
from .splits import canonical_pairs


def _output_dirs(root):
    root = Path(root)
    directories = [root / name / "processed" for name in ("slbenchmark", "perturbation_responses")]
    for path in directories:
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"The processed output directory must be empty: {path}")
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)
        (path.parent / "raw").mkdir(exist_ok=True)
    return directories


def write_manifests(root):
    root = Path(root)
    summary = statistics(root)
    response_dir = root / "perturbation_responses" / "processed"
    rows = pd.read_csv(response_dir / "responses.csv.gz")
    response_summary = {
        "measured_response_records": len(rows),
        "measured_response_pairs": len(rows[["u", "v"]].drop_duplicates()),
        "measured_response_genes": int(np.unique(rows[["u", "v"]].to_numpy()).size),
        "studies": int(rows.study.nunique()) if "study" in rows else 1,
        "study_contexts": len(rows[["study", "context"]].drop_duplicates())
        if "study" in rows
        else int(rows.context.nunique()),
        "complete_measured_pairs": summary["complete_measured_pairs"],
        "complete_measured_records": summary["complete_measured_records"],
    }
    for name, counts, filenames in (
        (
            "slbenchmark",
            {k: summary[k] for k in ("genes", "known_positive_pairs", "node_feature_dimension")},
            ("genes.csv", "sl_pairs.csv", "features.npz"),
        ),
        ("perturbation_responses", response_summary, ("responses.csv.gz", "response_records.csv")),
    ):
        directory = root / name / "processed"
        files = [
            {
                "path": file,
                "bytes": (directory / file).stat().st_size,
                "sha256": sha256(directory / file),
            }
            for file in filenames
        ]
        save_json(
            {"schema_version": 1, "summary": counts, "files": files}, directory / "manifest.json"
        )
    return {**summary, **response_summary}


def _copy_responses(responses, records, destination):
    shutil.copy2(records, destination / "response_records.csv")
    target = destination / "responses.csv.gz"
    if str(responses).endswith(".gz"):
        shutil.copy2(responses, target)
    else:
        pd.read_csv(responses).to_csv(target, index=False, compression="gzip")


def copy_curated(source, out):
    source = Path(source).resolve()
    if (source / "curated").is_dir():
        source = source / "curated"
    records = [source / name / "response_records.csv" for name in ("protocol_a", "protocal_a")]
    records = [p for p in records if p.is_file()]
    if len(records) != 1:
        raise ValueError("Expected one response_records.csv under protocol_a or protocal_a")
    responses = source / "intervention_sources_v28_measured.csv.gz"
    for path in [source / name for name in ("genes.csv", "sl_pairs.csv", "features.npz")] + [
        responses
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if Path(out).resolve() == source:
        raise ValueError("Choose a separate output directory")
    benchmark, response_dir = _output_dirs(out)
    for name in ("genes.csv", "sl_pairs.csv"):
        shutil.copy2(source / name, benchmark / name)
    with np.load(source / "features.npz", allow_pickle=False) as archive:
        np.savez_compressed(benchmark / "features.npz", stable_nodes=archive["stable_nodes"])
    _copy_responses(responses, records[0], response_dir)
    return write_manifests(out)


def build(slb, responses, response_records, out):
    slb = Path(slb)
    names = (
        "meta_table_9845.csv",
        "human_sl_9845.csv",
        "ppi_sparse_upper_matrix_without_sl_relation_9845.npz",
        "final_gosim_bp_from_r_9845.npy",
        "final_gosim_cc_from_r_9845.npy",
    )
    for path in [slb / name for name in names] + [Path(responses), Path(response_records)]:
        if not path.is_file():
            raise FileNotFoundError(path)
    benchmark, response_dir = _output_dirs(out)
    genes = pd.read_csv(slb / "meta_table_9845.csv").sort_values("unified_id")
    genes.to_csv(benchmark / "genes.csv", index=False)
    sl = pd.read_csv(slb / "human_sl_9845.csv").rename(
        columns={"unified_id_A": "u", "unified_id_B": "v"}
    )
    sl = canonical_pairs(sl)[["u", "v"]].drop_duplicates().sort_values(["u", "v"])
    sl["label"] = 1
    sl.to_csv(benchmark / "sl_pairs.csv", index=False)
    # Intermediate matrices are needed for PCA, not for training.
    with tempfile.TemporaryDirectory(prefix="cif4sl_features_") as temporary:
        work = Path(temporary)
        ppi = sp.load_npz(slb / "ppi_sparse_upper_matrix_without_sl_relation_9845.npz").tocsr()
        ppi = ppi.maximum(ppi.T).astype(np.float32)
        ppi.eliminate_zeros()
        sp.save_npz(work / "ppi.npz", ppi)
        for name in ("bp", "cc"):
            values = np.load(slb / f"final_gosim_{name}_from_r_9845.npy", mmap_mode="r")
            if np.isinf(values).any():
                raise ValueError("Infinite GO similarity")
            np.save(work / f"go_{name}.npy", np.nan_to_num(values, nan=0.0).astype(np.float32))
        features(work)
        shutil.copy2(work / "features.npz", benchmark / "features.npz")
    _copy_responses(responses, response_records, response_dir)
    return write_manifests(out)
