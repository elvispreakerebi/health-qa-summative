"""Merge prediction files by subset for validation and submission outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from health_qa.config import data_config_from_mapping, load_yaml
from health_qa.data import infer_schema, load_csv
from health_qa.metrics import score_predictions
from health_qa.submission import build_submission, save_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge predictions from two runs for selected subsets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--override-dir", required=True)
    parser.add_argument("--override-subsets", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_config = data_config_from_mapping(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    override_subsets = {subset.strip() for subset in args.override_subsets.split(",") if subset.strip()}

    val_df = load_csv(data_config.val_path)
    test_df = load_csv(data_config.test_path)
    val_schema = infer_schema(val_df, require_answer=True)
    test_schema = infer_schema(test_df, require_answer=False)

    validation = _merge_predictions(
        val_df,
        val_schema.id_col,
        "subset",
        Path(args.base_dir) / "validation_predictions.csv",
        Path(args.override_dir) / "validation_predictions.csv",
        override_subsets,
    )
    metrics = score_predictions(
        val_df[val_schema.answer_col].fillna("").astype(str).tolist(),  # type: ignore[index]
        validation["prediction"].fillna("").astype(str).tolist(),
    )
    validation["reference"] = val_df[val_schema.answer_col].to_numpy()  # type: ignore[index]
    validation.to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "rouge1_f1": metrics.rouge1_f1,
                "rouge_l_f1": metrics.rouge_l_f1,
                "weighted_without_llm": metrics.weighted_without_llm,
            }
        ]
    ).to_csv(output_dir / "metrics.csv", index=False)

    test_predictions = _merge_predictions(
        test_df,
        test_schema.id_col,
        "subset",
        Path(args.base_dir) / "submission.csv",
        Path(args.override_dir) / "submission.csv",
        override_subsets,
        submission_mode=True,
    )
    submission = build_submission(test_df[test_schema.id_col], test_predictions["prediction"].tolist())
    save_submission(submission, output_dir / "submission.csv")
    print(f"Metrics: {output_dir / 'metrics.csv'}")
    print(f"Submission: {output_dir / 'submission.csv'}")


def _merge_predictions(
    source_df: pd.DataFrame,
    id_col: str,
    group_col: str,
    base_path: Path,
    override_path: Path,
    override_subsets: set[str],
    *,
    submission_mode: bool = False,
) -> pd.DataFrame:
    base = _load_predictions(base_path, submission_mode=submission_mode)
    override = _load_predictions(override_path, submission_mode=submission_mode)
    merged = source_df[[id_col, group_col]].merge(base, left_on=id_col, right_on="ID", how="left")
    replacement = source_df[[id_col, group_col]].merge(override, left_on=id_col, right_on="ID", how="left")
    use_override = merged[group_col].isin(override_subsets)
    merged.loc[use_override, "prediction"] = replacement.loc[use_override, "prediction"].to_numpy()
    if merged["prediction"].isna().any():
        missing = merged.loc[merged["prediction"].isna(), id_col].head().tolist()
        raise ValueError(f"Missing predictions for IDs: {missing}")
    return pd.DataFrame({"ID": merged[id_col], "prediction": merged["prediction"]})


def _load_predictions(path: Path, *, submission_mode: bool) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    if submission_mode:
        return predictions.rename(columns={"TargetRLF1": "prediction"})[["ID", "prediction"]]
    return predictions[["ID", "prediction"]]


if __name__ == "__main__":
    main()
