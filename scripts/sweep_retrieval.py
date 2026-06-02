"""Run local TF-IDF retrieval sweeps against the validation split."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from health_qa.config import data_config_from_mapping, load_yaml
from health_qa.data import infer_schema, load_csv
from health_qa.metrics import score_predictions
from health_qa.retrieval import _predict_by_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep local retrieval hyperparameters")
    parser.add_argument("--config", default="configs/local_retrieval_char24.yaml")
    parser.add_argument("--output", default="outputs/retrieval_sweep/extended_results.csv")
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_config = data_config_from_mapping(config)
    train_df = load_csv(data_config.train_path)
    val_df = load_csv(data_config.val_path)
    train_schema = infer_schema(train_df, require_answer=True)
    val_schema = infer_schema(val_df, require_answer=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for candidate in _candidate_configs():
        validation = _predict_by_retrieval(train_df, val_df, train_schema, val_schema, candidate)
        metrics = score_predictions(
            val_df[val_schema.answer_col].fillna("").astype(str).tolist(),  # type: ignore[index]
            validation["prediction"].fillna("").astype(str).tolist(),
        )
        row = {
            **candidate,
            "rouge1_f1": metrics.rouge1_f1,
            "rouge_l_f1": metrics.rouge_l_f1,
            "weighted_without_llm": metrics.weighted_without_llm,
        }
        rows.append(row)
        pd.DataFrame(rows).sort_values("weighted_without_llm", ascending=False).to_csv(
            output_path,
            index=False,
        )
        print(
            f"{len(rows):03d} {metrics.weighted_without_llm:.6f} "
            f"{candidate['analyzer']} {candidate['ngram_min']}-{candidate['ngram_max']} "
            f"min_df={candidate['min_df']} max_features={candidate['max_features']} "
            f"sublinear={candidate['sublinear_tf']} binary={candidate['binary']} "
            f"use_idf={candidate['use_idf']} norm={candidate['norm']}"
        )

    best = pd.DataFrame(rows).sort_values("weighted_without_llm", ascending=False).iloc[0]
    print("\nBest candidate:")
    print(best.to_string())


def _candidate_configs() -> list[dict[str, Any]]:
    base = {
        "lowercase": True,
        "batch_size": 512,
        "include_val_for_test": True,
    }
    candidates: list[dict[str, Any]] = []
    for analyzer, ngram_range in (
        ("char", (1, 3)),
        ("char", (1, 4)),
        ("char", (2, 3)),
        ("char", (2, 4)),
        ("char", (2, 5)),
        ("char", (3, 5)),
        ("char_wb", (2, 4)),
        ("char_wb", (3, 5)),
        ("word", (1, 1)),
        ("word", (1, 2)),
    ):
        for max_features, min_df, sublinear_tf, binary, use_idf, norm in product(
            [50000, 100000, 200000],
            [1, 2, 3, 5],
            [False, True],
            [False, True],
            [False, True],
            ["l2"],
        ):
            if binary and sublinear_tf:
                continue
            candidates.append(
                {
                    **base,
                    "analyzer": analyzer,
                    "ngram_min": ngram_range[0],
                    "ngram_max": ngram_range[1],
                    "max_features": max_features,
                    "min_df": min_df,
                    "max_df": 1.0,
                    "sublinear_tf": sublinear_tf,
                    "binary": binary,
                    "use_idf": use_idf,
                    "smooth_idf": True,
                    "norm": norm,
                }
            )
    return candidates


if __name__ == "__main__":
    main()
