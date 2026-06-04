"""Train a lightweight row-level selector over saved candidate predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from health_qa.config import data_config_from_mapping, load_yaml
from health_qa.data import infer_schema, load_csv
from health_qa.metrics import score_predictions
from health_qa.submission import build_submission, save_submission


DEFAULT_CANDIDATE_DIRS = [
    "outputs/local_retrieval_char13",
    "outputs/local_retrieval_char24",
    "outputs/local_retrieval_subset_sweep",
    "outputs/local_retrieval_mpnet_rerank",
    "outputs/local_retrieval_e5_rerank",
    "outputs/local_retrieval_mpnet_e5_blend_trimmed",
    "outputs/local_retrieval_mpnet_union_rerank",
    "outputs/local_retrieval_mpnet_union_rerank_wide",
    "outputs/local_retrieval_mpnet_union_rerank_xwide",
    "outputs/local_retrieval_mpnet_union_rerank_xxwide",
    "outputs/local_retrieval_mpnet_e5_union_xxwide_blend_trimmed",
    "outputs/local_retrieval_mpnet_e5_union_xxwide_hybrid_parts_eng70",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn a candidate selector from validation labels")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-dirs", nargs="*", default=DEFAULT_CANDIDATE_DIRS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_config = data_config_from_mapping(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_df = load_csv(data_config.val_path)
    test_df = load_csv(data_config.test_path)
    val_schema = infer_schema(val_df, require_answer=True)
    test_schema = infer_schema(test_df, require_answer=False)

    val_candidates, test_candidates = _load_candidate_predictions(
        val_df,
        test_df,
        val_schema.id_col,
        test_schema.id_col,
        [Path(path) for path in args.candidate_dirs],
    )
    _add_synthetic_candidates(val_df, test_df, val_schema.question_col, test_schema.question_col, val_candidates, test_candidates)

    train_long = _build_long_frame(val_df, val_schema.id_col, val_schema.question_col, val_candidates)
    test_long = _build_long_frame(test_df, test_schema.id_col, test_schema.question_col, test_candidates)
    train_long["target"] = _score_candidate_rows(
        val_df[val_schema.answer_col].fillna("").astype(str).tolist(),  # type: ignore[index]
        val_candidates,
    )

    feature_cols = [
        "subset",
        "candidate",
        "question_words",
        "question_chars",
        "prediction_words",
        "prediction_chars",
        "pred_to_question_words",
        "word_jaccard",
        "char_jaccard",
        "starts_with_question",
    ]
    cv_predictions = _cross_validated_select(train_long, val_df[val_schema.id_col], feature_cols, args.folds, args.seed)
    metrics = score_predictions(
        val_df[val_schema.answer_col].fillna("").astype(str).tolist(),  # type: ignore[index]
        cv_predictions["prediction"].fillna("").astype(str).tolist(),
    )

    model = _make_model(args.seed)
    model.fit(train_long[feature_cols], train_long["target"])
    test_scored = test_long.copy()
    test_scored["selector_score"] = model.predict(test_long[feature_cols])
    test_predictions = _select_best(test_scored, ordered_ids=test_df[test_schema.id_col])

    cv_predictions["reference"] = val_df[val_schema.answer_col].to_numpy()  # type: ignore[index]
    cv_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_selected_predictions.csv", index=False)
    save_submission(
        build_submission(test_predictions["ID"], test_predictions["prediction"]),
        output_dir / "submission.csv",
    )
    pd.DataFrame(
        [
            {
                "rouge1_f1": metrics.rouge1_f1,
                "rouge_l_f1": metrics.rouge_l_f1,
                "weighted_without_llm": metrics.weighted_without_llm,
            }
        ]
    ).to_csv(output_dir / "metrics.csv", index=False)
    _write_selection_report(cv_predictions, output_dir / "selection_report.csv")
    print(f"Metrics: {output_dir / 'metrics.csv'}")
    print(f"Submission: {output_dir / 'submission.csv'}")


def _load_candidate_predictions(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_id_col: str,
    test_id_col: str,
    candidate_dirs: list[Path],
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    val_candidates: dict[str, pd.Series] = {}
    test_candidates: dict[str, pd.Series] = {}
    for directory in candidate_dirs:
        val_path = directory / "validation_predictions.csv"
        test_path = directory / "submission.csv"
        if not val_path.exists() or not test_path.exists():
            continue
        name = directory.name
        val_predictions = pd.read_csv(val_path)[["ID", "prediction"]]
        test_predictions = pd.read_csv(test_path)[["ID", "TargetRLF1"]].rename(
            columns={"TargetRLF1": "prediction"}
        )
        val_candidates[name] = (
            val_df[[val_id_col]]
            .merge(val_predictions, left_on=val_id_col, right_on="ID", how="left")["prediction"]
            .fillna("")
            .astype(str)
        )
        test_candidates[name] = (
            test_df[[test_id_col]]
            .merge(test_predictions, left_on=test_id_col, right_on="ID", how="left")["prediction"]
            .fillna("")
            .astype(str)
        )
    if not val_candidates:
        raise ValueError("No candidate predictions were loaded")
    return val_candidates, test_candidates


def _add_synthetic_candidates(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_question_col: str,
    test_question_col: str,
    val_candidates: dict[str, pd.Series],
    test_candidates: dict[str, pd.Series],
) -> None:
    base_name = "local_retrieval_mpnet_e5_union_xxwide_blend_trimmed"
    if base_name not in val_candidates:
        return
    for cap in (40, 60, 80, 120):
        name = f"input_only_{cap}"
        val_candidates[name] = val_df[val_question_col].fillna("").astype(str).map(lambda text: _trim_words(text, cap))
        test_candidates[name] = test_df[test_question_col].fillna("").astype(str).map(lambda text: _trim_words(text, cap))
    for input_cap, prediction_cap in ((80, 40), (80, 60), (120, 80), (None, 80)):
        name = f"input_pred_{input_cap}_{prediction_cap}"
        val_candidates[name] = _combine_input_prediction(
            val_df[val_question_col].fillna("").astype(str),
            val_candidates[base_name],
            input_cap,
            prediction_cap,
        )
        test_candidates[name] = _combine_input_prediction(
            test_df[test_question_col].fillna("").astype(str),
            test_candidates[base_name],
            input_cap,
            prediction_cap,
        )


def _build_long_frame(
    source_df: pd.DataFrame,
    id_col: str,
    question_col: str,
    candidates: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    questions = source_df[question_col].fillna("").astype(str).reset_index(drop=True)
    subsets = source_df["subset"].fillna("").astype(str).reset_index(drop=True)
    ids = source_df[id_col].reset_index(drop=True)
    for candidate_name, predictions in candidates.items():
        predictions = predictions.reset_index(drop=True).fillna("").astype(str)
        features = pd.DataFrame(
            {
                "ID": ids,
                "subset": subsets,
                "candidate": candidate_name,
                "prediction": predictions,
                "question_words": questions.map(lambda text: len(text.split())),
                "question_chars": questions.map(len),
                "prediction_words": predictions.map(lambda text: len(text.split())),
                "prediction_chars": predictions.map(len),
                "word_jaccard": [_word_jaccard(q, p) for q, p in zip(questions, predictions, strict=True)],
                "char_jaccard": [_char_jaccard(q, p) for q, p in zip(questions, predictions, strict=True)],
                "starts_with_question": [
                    int(str(p).startswith(str(q))) for q, p in zip(questions, predictions, strict=True)
                ],
            }
        )
        features["pred_to_question_words"] = features["prediction_words"] / features["question_words"].clip(lower=1)
        rows.append(features)
    return pd.concat(rows, ignore_index=True)


def _score_candidate_rows(references: list[str], candidates: dict[str, pd.Series]) -> list[float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    scores: list[float] = []
    for predictions in candidates.values():
        for reference, prediction in zip(references, predictions.fillna("").astype(str), strict=True):
            pair_scores = scorer.score(str(reference), str(prediction))
            scores.append(0.5 * pair_scores["rouge1"].fmeasure + 0.5 * pair_scores["rougeL"].fmeasure)
    return scores


def _cross_validated_select(
    train_long: pd.DataFrame,
    ordered_ids: pd.Series,
    feature_cols: list[str],
    folds: int,
    seed: int,
) -> pd.DataFrame:
    selected_parts = []
    row_index = train_long[["ID", "subset"]].drop_duplicates().reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_rows, holdout_rows in splitter.split(row_index, row_index["subset"]):
        train_ids = set(row_index.iloc[train_rows]["ID"])
        holdout_ids = set(row_index.iloc[holdout_rows]["ID"])
        train_part = train_long[train_long["ID"].isin(train_ids)]
        holdout_part = train_long[train_long["ID"].isin(holdout_ids)].copy()
        model = _make_model(seed)
        model.fit(train_part[feature_cols], train_part["target"])
        holdout_part["selector_score"] = model.predict(holdout_part[feature_cols])
        selected_parts.append(_select_best(holdout_part))
    selected = pd.concat(selected_parts, ignore_index=True)
    ordered = pd.DataFrame({"ID": ordered_ids})
    return ordered.merge(selected, on="ID", how="left")


def _make_model(seed: int) -> Pipeline:
    categorical = ["subset", "candidate"]
    numeric = [
        "question_words",
        "question_chars",
        "prediction_words",
        "prediction_chars",
        "pred_to_question_words",
        "word_jaccard",
        "char_jaccard",
        "starts_with_question",
    ]
    transformer = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", "passthrough", numeric),
        ]
    )
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=180,
        l2_regularization=0.02,
        random_state=seed,
    )
    return Pipeline([("features", transformer), ("regressor", regressor)])


def _select_best(scored: pd.DataFrame, ordered_ids: pd.Series | None = None) -> pd.DataFrame:
    winners = scored.sort_values(["ID", "selector_score"], ascending=[True, False]).groupby("ID", sort=False).head(1)
    selected = winners[["ID", "subset", "candidate", "prediction", "selector_score"]].reset_index(drop=True)
    if ordered_ids is None:
        return selected
    return pd.DataFrame({"ID": ordered_ids}).merge(selected, on="ID", how="left")


def _write_selection_report(predictions: pd.DataFrame, path: Path) -> None:
    report = (
        predictions.groupby(["subset", "candidate"], as_index=False)
        .size()
        .sort_values(["subset", "size"], ascending=[True, False])
    )
    report.to_csv(path, index=False)


def _combine_input_prediction(
    questions: pd.Series,
    predictions: pd.Series,
    input_cap: int | None,
    prediction_cap: int | None,
) -> pd.Series:
    return pd.Series(
        [
            f"{_trim_words(question, input_cap)} {_trim_words(prediction, prediction_cap)}".strip()
            for question, prediction in zip(questions, predictions, strict=True)
        ]
    )


def _trim_words(text: str, max_words: int | None) -> str:
    return text.strip() if max_words is None else " ".join(text.strip().split()[:max_words])


def _word_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.lower().split())
    right_tokens = set(right.lower().split())
    if not left_tokens and not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def _char_jaccard(left: str, right: str) -> float:
    left_chars = set(left.lower())
    right_chars = set(right.lower())
    if not left_chars and not right_chars:
        return 0.0
    return len(left_chars & right_chars) / max(len(left_chars | right_chars), 1)


if __name__ == "__main__":
    main()
