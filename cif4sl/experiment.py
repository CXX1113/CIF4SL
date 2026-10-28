"""Run CIF4SL across the configured seeds and folds."""

import copy
import gc
import hashlib
import json
import platform
from pathlib import Path
import pandas as pd
import torch
from .data import Dataset
from .evaluate import DISPLAY_METRICS, evaluate
from .io import save_json, sha256
from .splits import build_split
from .train import fit
from .config import resolve_settings


def select_device(cfg):
    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA is unavailable; select --device cpu")
    torch.set_num_threads(cfg["threads"])
    return device


def initialize(cfg, root):
    root.mkdir(parents=True, exist_ok=True)
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        source.update(str(path.relative_to(Path(__file__).parent)).encode() + path.read_bytes())
    normalized = copy.deepcopy(cfg)
    for key in ("output_dir", "fold_ids"):
        normalized.pop(key, None)
    body = {
        "config": normalized,
        "source_sha256": source.hexdigest(),
        "data_manifest_sha256": {
            name: sha256(Path(cfg["data_dir"]) / name / "processed/manifest.json")
            for name in ("slbenchmark", "perturbation_responses")
        },
        "response_sha256": sha256(cfg["response_file"]),
    }
    sign = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    path = root / "experiment.json"
    if path.exists() and json.loads(path.read_text())["signature"] != sign:
        raise ValueError("Code, settings or data changed; select a new --output directory")
    save_json(
        {
            "signature": sign,
            **body,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "test_used_for_selection": False,
        },
        path,
    )


def summarize(rows, root):
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "fold_metrics.csv", index=False)
    columns = [name for name in DISPLAY_METRICS if name in frame]
    summary = frame[columns].agg(["mean", "std"]).T
    summary.index.name = "metric"
    summary.to_csv(root / "summary.csv")
    return summary


def run(cfg, resume=True, verify=True, stats_only=False):
    device = select_device(cfg)
    root = Path(cfg["output_dir"])
    initialize(cfg, root)
    data = Dataset(cfg["data_dir"], verify=verify)
    folds = cfg.get("fold_ids", list(range(cfg["folds"])))
    rows = []
    for seed in cfg["seeds"]:
        for fold in folds:
            fold_root = root / "runs" / f"seed_{seed}" / f"fold_{fold}"
            split_root = fold_root / "splits"
            split_root.mkdir(parents=True, exist_ok=True)
            parts, responses, metadata, _ = build_split(data, cfg, seed, fold)
            for name, frame in parts.items():
                frame.to_csv(split_root / f"{name}.csv", index=False)
            for name, frame in responses.items():
                frame.to_csv(split_root / f"response_{name}.csv", index=False)
            save_json(metadata, split_root / "split_metadata.json")
            if stats_only:
                continue
            out = fold_root / "CIF4SL"
            out.mkdir(parents=True, exist_ok=True)
            done = out / "metrics.json"
            if resume and done.is_file() and (out / "ranking_audit.json").is_file():
                metrics = json.loads(done.read_text())
                print(f"Resume CIF4SL: seed={seed}, fold={fold}", flush=True)
            else:
                print(f"Train CIF4SL: seed={seed}, fold={fold}", flush=True)
                settings = resolve_settings(cfg)
                scorer = fit(
                    data=data,
                    parts=parts,
                    settings=settings,
                    seed=seed,
                    fold=fold,
                    out=out,
                    device=device,
                )
                metrics, audit = evaluate(
                    scorer, data, parts["train"], parts["valid"], parts["test"], cfg, out
                )
                save_json(metrics, done)
                save_json(audit, out / "ranking_audit.json")
                del scorer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            rows.append({"model": "CIF4SL", "seed": seed, "fold": fold, **metrics})
            summarize(rows, root)
    if rows:
        print(summarize(rows, root).to_string(), flush=True)
        save_json(
            {"completed_folds": folds, "seeds": cfg["seeds"], "full_candidate_catalog": True},
            root / "completion.json",
        )
    return rows
