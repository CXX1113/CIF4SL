"""Command-line interface for training, prediction and data preparation."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="cif4sl", description="CIF4SL training and prediction")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("run", help="Train and evaluate the configured folds")
    train.add_argument("--config", default="configs/cif4sl.yaml")
    train.add_argument("--device")
    train.add_argument("--fold-ids", type=int, nargs="+")
    train.add_argument("--output")
    train.add_argument("--data")
    train.add_argument("--responses")
    train.add_argument("--no-resume", action="store_true")
    train.add_argument(
        "--stats-only", action="store_true", help="Export the splits without training"
    )
    train.add_argument("--skip-hash-check", action="store_true")
    for name in ("stats", "audit"):
        utility = commands.add_parser(name)
        utility.add_argument("--data", default="data")
    prepare = commands.add_parser(
        "prepare-data", help="Copy required files from an existing curated dataset"
    )
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--out", default="data")
    build = commands.add_parser(
        "build-data", help="Build features from source matrices and supplied measured tables"
    )
    build.add_argument("--slb", required=True)
    build.add_argument("--responses", required=True)
    build.add_argument("--response-records", required=True)
    build.add_argument("--out", default="data")
    predict = commands.add_parser("predict", help="Score a CSV with numeric u,v gene IDs")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--pairs", required=True)
    predict.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        from .config import load_config, validate_config
        from .experiment import run

        cfg = load_config(args.config)
        if args.data:
            old_root = Path(cfg["data_dir"])
            response = Path(cfg["response_file"])
            if response.is_relative_to(old_root):
                cfg["response_file"] = str(Path(args.data) / response.relative_to(old_root))
            cfg["data_dir"] = args.data
        if args.responses:
            cfg["response_file"] = args.responses
        if args.output:
            cfg["output_dir"] = args.output
        if args.device:
            cfg["device"] = args.device
        if args.fold_ids is not None:
            if len(set(args.fold_ids)) != len(args.fold_ids) or any(
                (fold < 0 or fold >= cfg["folds"] for fold in args.fold_ids)
            ):
                parser.error("--fold-ids must be unique and lie within the configured folds")
            cfg["fold_ids"] = args.fold_ids
        validate_config(cfg)
        run(
            cfg,
            resume=not args.no_resume,
            verify=not args.skip_hash_check,
            stats_only=args.stats_only,
        )
    elif args.command == "stats":
        from .data import statistics

        print(json.dumps(statistics(args.data), indent=2))
    elif args.command == "audit":
        from .data import Dataset, statistics

        Dataset(args.data, verify=True)
        print(json.dumps(statistics(args.data), indent=2))
    elif args.command == "prepare-data":
        from .prepare import copy_curated

        print(json.dumps(copy_curated(args.source, args.out), indent=2))
    elif args.command == "build-data":
        from .prepare import build

        print(
            json.dumps(build(args.slb, args.responses, args.response_records, args.out), indent=2)
        )
    elif args.command == "predict":
        from .scoring import load_scorer
        from .splits import read_pairs

        scorer = load_scorer(args.checkpoint)
        pairs = read_pairs(args.pairs, scorer.n)
        pairs["score"] = scorer.predict(pairs[["u", "v"]].to_numpy())
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(path, index=False)


if __name__ == "__main__":
    main()
