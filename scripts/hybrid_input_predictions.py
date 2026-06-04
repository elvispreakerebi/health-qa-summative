"""Blend source questions into predictions for selected subsets."""

from __future__ import annotations

import argparse
import operator
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
    rules: dict[str, HybridRule],
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
        if not rule.applies(str(question), str(prediction)):
            outputs.append(str(prediction).strip())
            continue
        outputs.append(rule.combine(str(question), str(prediction)))
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


class HybridRule:
    def __init__(self, mode: str, max_words: str, condition: str | None = None) -> None:
        self.mode = mode
        self.max_words, self.input_max_words, self.prediction_max_words = _parse_max_words(max_words)
        self.condition = _parse_condition(condition)

    def applies(self, question: str, prediction: str) -> bool:
        if self.condition is None:
            return True
        feature, compare, threshold = self.condition
        value = {
            "input_words": len(question.split()),
            "pred_words": len(prediction.split()),
        }[feature]
        return compare(value, threshold)

    def combine(self, question: str, prediction: str) -> str:
        if self.input_max_words is None and self.prediction_max_words is None:
            return _trim_words(_combine(question, prediction, self.mode), self.max_words)
        question_part = _trim_words(question, self.input_max_words)
        prediction_part = _trim_words(prediction, self.prediction_max_words)
        return _combine(question_part, prediction_part, self.mode)


def _parse_rules(raw: str) -> dict[str, HybridRule]:
    rules: dict[str, HybridRule] = {}
    for item in raw.split(","):
        parts = item.split(":")
        if len(parts) not in (3, 4):
            raise ValueError("Each rule must be subset:mode:max_words[:condition]")
        subset, mode, max_words = parts[:3]
        condition = parts[3] if len(parts) == 4 else None
        rules[subset.strip()] = HybridRule(mode.strip(), max_words.strip(), condition)
    return rules


def _parse_max_words(raw: str) -> tuple[int | None, int | None, int | None]:
    if "+" not in raw:
        return None if raw == "none" else int(raw), None, None
    input_cap, prediction_cap = raw.split("+", maxsplit=1)
    return None, _parse_optional_int(input_cap), _parse_optional_int(prediction_cap)


def _parse_optional_int(raw: str) -> int | None:
    value = raw.strip()
    return None if value == "none" else int(value)


def _parse_condition(raw: str | None):
    if not raw:
        return None
    operators = {
        "<=": operator.le,
        ">=": operator.ge,
        "<": operator.lt,
        ">": operator.gt,
    }
    for symbol, compare in operators.items():
        if symbol in raw:
            feature, threshold = raw.split(symbol, maxsplit=1)
            feature = feature.strip()
            if feature not in {"input_words", "pred_words"}:
                raise ValueError(f"Unsupported condition feature: {feature}")
            return feature, compare, float(threshold)
    raise ValueError(f"Unsupported condition: {raw}")


if __name__ == "__main__":
    main()
