"""Read and validate the experiment YAML."""

import math
from pathlib import Path

import yaml

from .splits import validate_config as validate_splits


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def resolve_settings(cfg):
    return {
        **cfg["model"],
        **cfg["training"],
        **cfg["compute"],
        "response_file": cfg["response_file"],
    }


def validate_config(cfg):
    required = {
        "data_dir",
        "response_file",
        "output_dir",
        "device",
        "threads",
        "seeds",
        "folds",
        "split",
        "model",
        "training",
        "compute",
        "ranking_ks",
    }
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Missing settings: {sorted(missing)}")
    for key in ("threads", "folds"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (
        not isinstance(cfg["seeds"], list)
        or not cfg["seeds"]
        or any(type(seed) is not int or seed < 0 for seed in cfg["seeds"])
    ):
        raise ValueError("seeds must be a nonempty list of nonnegative integers")
    if list(cfg["ranking_ks"]) != [10, 50]:
        raise ValueError("The reproduction setting uses ranking_ks: [10, 50]")
    for key, expected in {
        "cv": "CV1",
        "negative_strategy": "Rand",
        "catalog_scope": "all_genes",
        "catalog_filter": "train_valid",
    }.items():
        if cfg["split"].get(key) != expected:
            raise ValueError(f"The reproduction setting requires split.{key}: {expected}")
    validate_splits(cfg)
    if not isinstance(cfg["response_file"], str) or not cfg["response_file"]:
        raise ValueError("response_file must name the measured response table")
    for section in ("model", "training", "compute"):
        if not isinstance(cfg[section], dict):
            raise ValueError(f"{section} must be a mapping")
    names = [key for section in ("model", "training", "compute") for key in cfg[section]]
    if len(names) != len(set(names)):
        raise ValueError("A setting must appear in only one YAML section")
    settings = resolve_settings(cfg)
    for key in (
        "embedding_dim",
        "hidden_dim",
        "context_dim",
        "warmup_epochs",
        "joint_epochs",
        "warmup_eval_every",
        "joint_eval_every",
        "bliss_pairs",
        "batch_size",
        "pair_batch_size",
        "graph_row_block_size",
        "graph_pair_batch_size",
        "query_batch_size",
    ):
        if key not in settings:
            raise ValueError(f"Missing YAML setting: {key}")
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "warmup_lr",
        "joint_lr",
        "grad_clip",
        "lambda_margin",
        "lambda_bliss",
        "weight_decay",
        "m_pos",
        "m_neg",
        "dropout",
        "sparsemax_temperature",
    ):
        if key not in settings:
            raise ValueError(f"Missing YAML setting: {key}")
        value = settings[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be finite and numeric")
    if min(settings["warmup_lr"], settings["joint_lr"], settings["grad_clip"]) <= 0:
        raise ValueError("Learning rates and gradient clipping must be positive")
    if min(settings["lambda_margin"], settings["lambda_bliss"], settings["weight_decay"]) < 0:
        raise ValueError("Loss weights and weight decay must be nonnegative")
    if not 0 <= settings["dropout"] < 1:
        raise ValueError("dropout must lie in [0,1)")
    if not -1 < settings["m_neg"] < settings["m_pos"] < 1:
        raise ValueError("Expected -1 < m_neg < m_pos < 1")
    if settings["sparsemax_temperature"] <= 0:
        raise ValueError("sparsemax_temperature must be positive")
    if not isinstance(settings.get("context_column"), str) or not settings["context_column"]:
        raise ValueError("context_column must name the measured context column")
    return cfg


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        cfg = yaml.load(handle, Loader=UniqueLoader)
    if not isinstance(cfg, dict):
        raise ValueError("The configuration must be a YAML mapping")
    return validate_config(cfg)
