"""Apply subset-specific postprocessing to validation predictions and submissions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from health_qa.config import data_config_from_mapping, load_yaml
from health_qa.data import infer_schema, load_csv
from health_qa.metrics import score_predictions
from health_qa.submission import build_submission, save_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess predictions by subset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-words-by-subset", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_config = data_config_from_mapping(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_words_by_subset = _parse_subset_ints(args.max_words_by_subset)

    val_df = load_csv(data_config.val_path)
    test_df = load_csv(data_config.test_path)
    val_schema = infer_schema(val_df, require_answer=True)
    test_schema = infer_schema(test_df, require_answer=False)
    input_dir = Path(args.input_dir)

    validation = _postprocess(
        val_df,
        val_schema.id_col,
        input_dir / "validation_predictions.csv",
        max_words_by_subset,
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

    test_predictions = _postprocess(
        test_df,
        test_schema.id_col,
        input_dir / "submission.csv",
        max_words_by_subset,
        submission_mode=True,
    )
    submission = build_submission(test_df[test_schema.id_col], test_predictions["prediction"].tolist())
    save_submission(submission, output_dir / "submission.csv")
    print(f"Metrics: {output_dir / 'metrics.csv'}")
    print(f"Submission: {output_dir / 'submission.csv'}")


def _postprocess(
    source_df: pd.DataFrame,
    id_col: str,
    prediction_path: Path,
    max_words_by_subset: dict[str, int],
    *,
    submission_mode: bool = False,
) -> pd.DataFrame:
    predictions = pd.read_csv(prediction_path)
    if submission_mode:
        predictions = predictions.rename(columns={"TargetRLF1": "prediction"})[["ID", "prediction"]]
    else:
        predictions = predictions[["ID", "prediction"]]
    merged = source_df[[id_col, "subset"]].merge(predictions, left_on=id_col, right_on="ID", how="left")
    if merged["prediction"].isna().any():
        missing = merged.loc[merged["prediction"].isna(), id_col].head().tolist()
        raise ValueError(f"Missing predictions for IDs: {missing}")
    merged["prediction"] = [
        _trim_words(prediction, max_words_by_subset.get(str(subset)))
        for prediction, subset in zip(merged["prediction"], merged["subset"], strict=True)
    ]
    return pd.DataFrame({"ID": merged[id_col], "prediction": merged["prediction"]})


def _trim_words(text: object, max_words: int | None) -> str:
    output = str(text).strip()
    if max_words is None:
        return output
    words = output.split()
    if len(words) <= max_words:
        return output
    return " ".join(words[:max_words])


def _parse_subset_ints(raw: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for item in raw.split(","):
        subset, max_words = item.split(":", maxsplit=1)
        values[subset.strip()] = int(max_words)
    return values


if __name__ == "__main__":
    main()
