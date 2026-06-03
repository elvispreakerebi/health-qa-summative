"""Blend source questions into predictions for selected subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from health_qa.config import data_config_from_mapping, load_yaml
from health_qa.data import infer_schema, load_csv
from health_qa.metrics import score_predictions
from health_qa.submission import build_submission, save_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Create subset-specific input/prediction hybrids")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rules", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_config = data_config_from_mapping(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rules = _parse_rules(args.rules)

    val_df = load_csv(data_config.val_path)
    test_df = load_csv(data_config.test_path)
    val_schema = infer_schema(val_df, require_answer=True)
    test_schema = infer_schema(test_df, require_answer=False)
    input_dir = Path(args.input_dir)

    validation = _hybrid_predictions(
        val_df,
        val_schema.id_col,
        val_schema.question_col,
        input_dir / "validation_predictions.csv",
        rules,
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

    test_predictions = _hybrid_predictions(
        test_df,
        test_schema.id_col,
        test_schema.question_col,
        input_dir / "submission.csv",
        rules,
        submission_mode=True,
    )
    submission = build_submission(test_df[test_schema.id_col], test_predictions["prediction"].tolist())
    save_submission(submission, output_dir / "submission.csv")
    print(f"Metrics: {output_dir / 'metrics.csv'}")
    print(f"Submission: {output_dir / 'submission.csv'}")


def _hybrid_predictions(
    source_df: pd.DataFrame,
    id_col: str,
    question_col: str,
    prediction_path: Path,
    rules: dict[str, tuple[str, int | None]],
    *,
    submission_mode: bool = False,
) -> pd.DataFrame:
    predictions = pd.read_csv(prediction_path)
    if submission_mode:
        predictions = predictions.rename(columns={"TargetRLF1": "prediction"})[["ID", "prediction"]]
    else:
        predictions = predictions[["ID", "prediction"]]

    merged = source_df[[id_col, question_col, "subset"]].merge(predictions, left_on=id_col, right_on="ID", how="left")
    if merged["prediction"].isna().any():
        missing = merged.loc[merged["prediction"].isna(), id_col].head().tolist()
        raise ValueError(f"Missing predictions for IDs: {missing}")

    outputs: list[str] = []
    for question, prediction, subset in zip(
        merged[question_col],
        merged["prediction"],
        merged["subset"],
        strict=True,
    ):
        rule = rules.get(str(subset))
        if rule is None:
            outputs.append(str(prediction).strip())
            continue
        mode, max_words = rule
        outputs.append(_trim_words(_combine(str(question), str(prediction), mode), max_words))
    return pd.DataFrame({"ID": merged[id_col], "prediction": outputs})


def _combine(question: str, prediction: str, mode: str) -> str:
    question = question.strip()
    prediction = prediction.strip()
    if mode == "input_pred":
        return f"{question} {prediction}".strip()
    if mode == "pred_input":
        return f"{prediction} {question}".strip()
    if mode == "input_only":
        return question
    raise ValueError(f"Unsupported hybrid mode: {mode}")


def _trim_words(text: str, max_words: int | None) -> str:
    if max_words is None:
        return text.strip()
    return " ".join(text.strip().split()[:max_words])


def _parse_rules(raw: str) -> dict[str, tuple[str, int | None]]:
    rules: dict[str, tuple[str, int | None]] = {}
    for item in raw.split(","):
        subset, mode, max_words = item.split(":", maxsplit=2)
        rules[subset.strip()] = (mode.strip(), None if max_words.strip() == "none" else int(max_words))
    return rules


if __name__ == "__main__":
    main()
