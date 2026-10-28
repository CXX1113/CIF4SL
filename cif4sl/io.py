"""JSON output and dataset checksums."""

import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def verify_manifest(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest["files"]:
        p = (root / entry["path"]).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError("Unsafe manifest path")
        if not p.exists() or sha256(p) != entry["sha256"]:
            raise IOError(f"Missing/corrupt dataset file: {p}")
    return manifest
