"""Sweep source/prediction hybrid rules against validation ROUGE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Search conditional hybrid rules by subset")
    parser.add_argument("--val-path", default="data/raw/Val.csv")
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--subsets", nargs="+", required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    val = pd.read_csv(args.val_path)
    predictions = pd.read_csv(args.predictions_path)
    merged = val[["ID", "subset", "input", "output"]].merge(
        predictions[["ID", "prediction"]],
        on="ID",
        how="inner",
    )

    rows: list[dict[str, object]] = []
    for subset in args.subsets:
        subset_rows = merged[merged["subset"].eq(subset)].reset_index(drop=True)
        if subset_rows.empty:
            raise ValueError(f"No validation rows for subset: {subset}")
        print(f"Sweeping {subset} ({len(subset_rows)} rows)", flush=True)
        rows.extend(_sweep_subset(subset_rows, subset))

    results = pd.DataFrame(rows).sort_values(["subset", "delta"], ascending=[True, False])
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    for subset, group in results.groupby("subset", sort=False):
        print(f"\n{subset}", flush=True)
        columns = ["delta", "score", "condition", "mode", "cap", "input_cap", "prediction_cap", "n_changed"]
        print(group[columns].head(args.top).to_string(index=False), flush=True)
    print(f"\nResults: {output_path}", flush=True)


def _sweep_subset(df: pd.DataFrame, subset: str) -> list[dict[str, object]]:
    refs = df["output"].fillna("").astype(str).tolist()
    base_preds = df["prediction"].fillna("").astype(str).tolist()
    base_scores = _row_scores(refs, base_preds)
    base_mean = float(base_scores.mean())
    input_words = df["input"].fillna("").astype(str).str.split().str.len().to_numpy()
    pred_words = df["prediction"].fillna("").astype(str).str.split().str.len().to_numpy()

    rows: list[dict[str, object]] = []
    specs = _candidate_specs()
    for index, spec in enumerate(specs, start=1):
        if index % 25 == 0:
            print(f"  {subset}: scored {index}/{len(specs)} candidate forms", flush=True)
        candidate = [
            _build_candidate(question, prediction, spec)
            for question, prediction in zip(df["input"], df["prediction"], strict=True)
        ]
        candidate_scores = _row_scores(refs, candidate)
        delta = candidate_scores - base_scores
        rows.append(_result_row(subset, base_mean, float(candidate_scores.mean()), "all", spec, len(df)))

        for condition, mask in _condition_masks(input_words, pred_words):
            n_changed = int(mask.sum())
            if n_changed == 0 or n_changed == len(df):
                continue
            score = base_mean + float(delta[mask].sum()) / len(df)
            rows.append(_result_row(subset, base_mean, score, condition, spec, n_changed))
    return rows


def _candidate_specs() -> list[dict[str, int | str | None]]:
    specs: list[dict[str, int | str | None]] = []
    for cap in [40, 50, 60, 70, 80, 90, 100, 120, 140, None]:
        for mode in ["input_only", "input_pred", "pred_input"]:
            specs.append({"mode": mode, "cap": cap, "input_cap": None, "prediction_cap": None})
    for input_cap in [30, 40, 50, 60, 80, 100, 120, None]:
        for prediction_cap in [20, 30, 40, 50, 60, 80, None]:
            for mode in ["input_pred", "pred_input"]:
                specs.append(
                    {
                        "mode": mode,
                        "cap": None,
                        "input_cap": input_cap,
                        "prediction_cap": prediction_cap,
                    }
                )
    return specs


def _condition_masks(input_words: np.ndarray, pred_words: np.ndarray):
    thresholds = {
        "input_words": [20, 30, 40, 50, 60, 80, 100, 120, 160],
        "pred_words": [20, 24, 30, 40, 50, 60, 80, 100, 120],
    }
    values = {"input_words": input_words, "pred_words": pred_words}
    for feature, feature_values in values.items():
        for threshold in thresholds[feature]:
            yield f"{feature}<={threshold}", feature_values <= threshold
            yield f"{feature}>{threshold}", feature_values > threshold


def _build_candidate(question: object, prediction: object, spec: dict[str, int | str | None]) -> str:
    mode = str(spec["mode"])
    question_text = _trim_words(str(question), spec["input_cap"])
    prediction_text = _trim_words(str(prediction), spec["prediction_cap"])
    if mode == "input_only":
        output = question_text
    elif mode == "input_pred":
        output = f"{question_text} {prediction_text}".strip()
    elif mode == "pred_input":
        output = f"{prediction_text} {question_text}".strip()
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return _trim_words(output, spec["cap"])


def _trim_words(text: str, cap: int | str | None) -> str:
    text = text.strip()
    if cap is None:
        return text
    return " ".join(text.split()[: int(cap)])


def _row_scores(refs: list[str], predictions: list[str]) -> np.ndarray:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    scores = np.empty(len(refs), dtype=float)
    for index, (reference, prediction) in enumerate(zip(refs, predictions, strict=True)):
        rouge = scorer.score(reference, prediction)
        scores[index] = 0.5 * rouge["rouge1"].fmeasure + 0.5 * rouge["rougeL"].fmeasure
    return scores


def _result_row(
    subset: str,
    base_score: float,
    score: float,
    condition: str,
    spec: dict[str, int | str | None],
    n_changed: int,
) -> dict[str, object]:
    return {
        "subset": subset,
        "base_score": base_score,
        "score": score,
        "delta": score - base_score,
        "condition": condition,
        "mode": spec["mode"],
        "cap": spec["cap"],
        "input_cap": spec["input_cap"],
        "prediction_cap": spec["prediction_cap"],
        "n_changed": n_changed,
    }


if __name__ == "__main__":
    main()
