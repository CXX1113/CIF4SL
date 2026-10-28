"""Load the gene catalog, SL pairs and features used by CIF4SL."""

from pathlib import Path
import numpy as np
import pandas as pd
from ._measured import complete_measured_mask
from .io import verify_manifest
from .splits import canonical_pairs, keys


class Dataset:
    def __init__(self, root, verify=True):
        self.root = Path(root)
        benchmark = self.root / "slbenchmark" / "processed"
        responses = self.root / "perturbation_responses" / "processed"
        if verify:
            self.manifest = {
                "slbenchmark": verify_manifest(benchmark),
                "perturbation_responses": verify_manifest(responses),
            }
        self.genes = pd.read_csv(benchmark / "genes.csv")
        self.n_genes = len(self.genes)
        sl = canonical_pairs(pd.read_csv(benchmark / "sl_pairs.csv"))
        self.sl = (
            sl.loc[sl.label.eq(1), ["u", "v"]]
            .drop_duplicates()
            .sort_values(["u", "v"])
            .reset_index(drop=True)
        )
        self.sl["label"] = 1
        rows = pd.read_csv(responses / "response_records.csv")
        rows = canonical_pairs(rows.loc[complete_measured_mask(rows, self.sl)])
        self.observations = rows.sort_values(["u", "v", "context"]).reset_index(drop=True)
        if not keys(self.observations) <= keys(self.sl):
            raise ValueError("Measured pairs must be known SLDB positives")
        with np.load(benchmark / "features.npz", allow_pickle=False) as features:
            self.stable_nodes = features["stable_nodes"]
        if self.stable_nodes.ndim != 2 or len(self.stable_nodes) != self.n_genes:
            raise ValueError("stable_nodes must have one row per catalog gene")
        if not np.isfinite(self.stable_nodes).all():
            raise ValueError("stable_nodes contains nonfinite values")
        if "unified_id" in self.genes and (
            not np.array_equal(self.genes.unified_id.to_numpy(), np.arange(self.n_genes))
        ):
            raise ValueError("genes.csv must retain contiguous, ordered catalog IDs")


def statistics(root):
    data = Dataset(root, verify=False)
    return {
        "genes": data.n_genes,
        "known_positive_pairs": len(data.sl),
        "unknown_pairs": data.n_genes * (data.n_genes - 1) // 2 - len(data.sl),
        "complete_measured_pairs": len(data.observations[["u", "v"]].drop_duplicates()),
        "complete_measured_records": len(data.observations),
        "node_feature_dimension": data.stable_nodes.shape[1],
    }
