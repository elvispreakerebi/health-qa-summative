"""Command line entry points for local and Colab runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from health_qa.config import DataConfig, data_config_from_mapping, load_yaml
from health_qa.data import infer_schema, load_csv, summarize_frame
from health_qa.modeling import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Health QA pipeline utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-data", help="Summarize train/val/test CSVs")
    inspect_parser.add_argument("--config", default=None, help="Optional YAML config path")
    inspect_parser.add_argument("--data-dir", default=None, help="Override raw data directory")

    train_parser = subparsers.add_parser(
        "train-generate",
        help="Fine-tune a seq2seq model and generate a submission",
    )
    train_parser.add_argument("--config", required=True, help="YAML experiment config")
    train_parser.add_argument("--output-dir", required=True, help="Run output directory")

    args = parser.parse_args()
    if args.command == "inspect-data":
        config = load_yaml(args.config) if args.config else {"data": {}}
        data_config = data_config_from_mapping(config)
        if args.data_dir:
            data_config = DataConfig(raw_dir=Path(args.data_dir))
        _inspect_data(data_config)
    elif args.command == "train-generate":
        artifacts = run_training_pipeline(args.config, args.output_dir)
        print(f"Submission: {artifacts.submission_path}")
        print(f"Validation predictions: {artifacts.validation_predictions_path}")
        print(f"Metrics: {artifacts.metrics_path}")


def _inspect_data(config: DataConfig) -> None:
    payload = {}
    for split, path, require_answer in (
        ("train", config.train_path, True),
        ("val", config.val_path, True),
        ("test", config.test_path, False),
    ):
        df = load_csv(path)
        schema = infer_schema(df, require_answer=require_answer)
        payload[split] = {
            "path": str(path),
            "schema": schema.__dict__,
            "summary": summarize_frame(df, schema),
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
