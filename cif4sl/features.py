"""PPI and GO feature construction used by CIF4SL."""

from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def features(root):
    root = Path(root)
    matrices = [sp.load_npz(root / "ppi.npz").toarray()] + [
        np.load(root / f"go_{name}.npy", mmap_mode="r") for name in ["bp", "cc"]
    ]
    pcas = []
    for name, a in zip(["ppi", "go_bp", "go_cc"], matrices):
        print(f"PCA {name} {a.shape}", flush=True)
        z = (
            PCA(n_components=128, svd_solver="randomized", iterated_power=3, random_state=42)
            .fit_transform(a)
            .astype(np.float32)
        )
        pcas.append(StandardScaler().fit_transform(z).astype(np.float32))
    np.savez_compressed(
        root / "features.npz",
        stable_nodes=np.concatenate(pcas, -1),
    )
