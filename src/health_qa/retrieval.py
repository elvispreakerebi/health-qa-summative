"""Local retrieval baseline that runs without GPU training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import pandas as pd

from health_qa.config import data_config_from_mapping, load_yaml
from health_qa.data import DatasetSchema, infer_schema, load_csv
from health_qa.metrics import score_predictions
from health_qa.submission import build_submission, save_submission


@dataclass(frozen=True)
class RetrievalArtifacts:
    output_dir: Path
    submission_path: Path
    validation_predictions_path: Path
    metrics_path: Path


def run_retrieval_pipeline(config_path: str | Path, output_dir: str | Path) -> RetrievalArtifacts:
    """Score a local TF-IDF nearest-neighbor QA baseline and write a submission."""
    config = load_yaml(config_path)
    data_config = data_config_from_mapping(config)
    retrieval_config = _retrieval_config_from_mapping(config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df = load_csv(data_config.train_path)
    val_df = load_csv(data_config.val_path)
    test_df = load_csv(data_config.test_path)

    train_schema = infer_schema(train_df, require_answer=True)
    val_schema = infer_schema(val_df, require_answer=True)
    test_schema = infer_schema(test_df, require_answer=False)

    validation = _predict_by_retrieval(train_df, val_df, train_schema, val_schema, retrieval_config)
    test_bank = _test_retrieval_bank(train_df, val_df, retrieval_config)
    test_bank_schema = infer_schema(test_bank, require_answer=True)
    test_predictions = _predict_by_retrieval(
        test_bank,
        test_df,
        test_bank_schema,
        test_schema,
        retrieval_config,
    )["prediction"].tolist()

    references = val_df[val_schema.answer_col].fillna("").astype(str).tolist()  # type: ignore[index]
    predictions = validation["prediction"].fillna("").astype(str).tolist()
    metrics = score_predictions(references, predictions)

    validation_predictions_path = output_path / "validation_predictions.csv"
    metrics_path = output_path / "metrics.csv"
    submission_path = output_path / "submission.csv"

    validation.to_csv(validation_predictions_path, index=False)
    pd.DataFrame(
        [
            {
                "rouge1_f1": metrics.rouge1_f1,
                "rouge_l_f1": metrics.rouge_l_f1,
                "weighted_without_llm": metrics.weighted_without_llm,
            }
        ]
    ).to_csv(metrics_path, index=False)

    submission = build_submission(test_df[test_schema.id_col], test_predictions)
    save_submission(submission, submission_path)

    return RetrievalArtifacts(
        output_dir=output_path,
        submission_path=submission_path,
        validation_predictions_path=validation_predictions_path,
        metrics_path=metrics_path,
    )


def _retrieval_config_from_mapping(config: dict[str, Any]) -> dict[str, Any]:
    retrieval_config = config.get("retrieval", {})
    ensemble_config = config.get("retrieval_ensemble")
    if ensemble_config:
        return {"ensemble": ensemble_config}
    return retrieval_config


def _predict_by_retrieval(
    bank_df: pd.DataFrame,
    query_df: pd.DataFrame,
    bank_schema: DatasetSchema,
    query_schema: DatasetSchema,
    config: dict[str, Any],
) -> pd.DataFrame:
    if "ensemble" in config:
        return _predict_by_ensemble(bank_df, query_df, bank_schema, query_schema, config["ensemble"])
    if "group_configs" in config:
        return _predict_with_group_configs(bank_df, query_df, bank_schema, query_schema, config)
    group_col = config.get("group_col")
    if group_col and group_col in bank_df.columns and group_col in query_df.columns:
        return _predict_grouped_by_retrieval(
            bank_df,
            query_df,
            bank_schema,
            query_schema,
            config,
            group_col=str(group_col),
        )
    return _predict_single_bank(bank_df, query_df, bank_schema, query_schema, config)


def _predict_with_group_configs(
    bank_df: pd.DataFrame,
    query_df: pd.DataFrame,
    bank_schema: DatasetSchema,
    query_schema: DatasetSchema,
    config: dict[str, Any],
) -> pd.DataFrame:
    group_col = str(config.get("group_col", "subset"))
    if group_col not in query_df.columns:
        raise ValueError(f"group_configs requires query column '{group_col}'")

    default_config = dict(config.get("default", {}))
    overrides = config.get("group_configs", {})
    if not isinstance(overrides, dict):
        raise ValueError("group_configs must be a mapping from group value to retrieval config")

    outputs: list[pd.DataFrame] = []
    for group_value, group_queries in query_df.groupby(group_col, sort=False):
        group_config = dict(default_config)
        group_config.update(overrides.get(group_value, {}))
        outputs.append(_predict_single_bank(bank_df, group_queries, bank_schema, query_schema, group_config))
    if not outputs:
        return _empty_prediction_frame(query_df, query_schema)
    return pd.concat(outputs, ignore_index=True)


def _predict_by_ensemble(
    bank_df: pd.DataFrame,
    query_df: pd.DataFrame,
    bank_schema: DatasetSchema,
    query_schema: DatasetSchema,
    config: dict[str, Any],
) -> pd.DataFrame:
    members = config.get("members", [])
    if not members:
        raise ValueError("retrieval_ensemble.members must contain at least one member")

    member_outputs: list[pd.DataFrame] = []
    for member in members:
        member_config = dict(member)
        weight = float(member_config.pop("weight", 1.0))
        output = _predict_by_retrieval(bank_df, query_df, bank_schema, query_schema, member_config)
        output = output.rename(
            columns={
                "matched_id": "member_matched_id",
                "similarity": "member_similarity",
                "prediction": "member_prediction",
            }
        )
        output["member_weight"] = weight
        member_outputs.append(output)

    rows: list[dict[str, object]] = []
    ids = query_df[query_schema.id_col].reset_index(drop=True)
    for row_index, query_id in enumerate(ids):
        candidates: dict[str, dict[str, object]] = {}
        for output in member_outputs:
            member_row = output.iloc[row_index]
            prediction = str(member_row["member_prediction"])
            similarity = float(member_row["member_similarity"])
            weight = float(member_row["member_weight"])
            candidate = candidates.setdefault(
                prediction,
                {
                    "prediction": prediction,
                    "matched_id": member_row["member_matched_id"],
                    "vote_weight": 0.0,
                    "similarity_weight": 0.0,
                    "best_similarity": 0.0,
                },
            )
            candidate["vote_weight"] = float(candidate["vote_weight"]) + weight
            candidate["similarity_weight"] = float(candidate["similarity_weight"]) + (similarity * weight)
            candidate["best_similarity"] = max(float(candidate["best_similarity"]), similarity)

        best = max(
            candidates.values(),
            key=lambda item: (
                float(item["vote_weight"]),
                float(item["similarity_weight"]),
                float(item["best_similarity"]),
            ),
        )
        rows.append(
            {
                "ID": query_id,
                "matched_id": best["matched_id"],
                "similarity": best["best_similarity"],
                "prediction": best["prediction"],
            }
        )
    output = pd.DataFrame(rows)
    if query_schema.answer_col:
        output["reference"] = query_df[query_schema.answer_col].to_numpy()
    return output


def _predict_grouped_by_retrieval(
    bank_df: pd.DataFrame,
    query_df: pd.DataFrame,
    bank_schema: DatasetSchema,
    query_schema: DatasetSchema,
    config: dict[str, Any],
    *,
    group_col: str,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    fallback_bank = bank_df
    for group_value, group_queries in query_df.groupby(group_col, sort=False):
        group_bank = bank_df[bank_df[group_col] == group_value]
        if group_bank.empty:
            group_bank = fallback_bank
        outputs.append(_predict_single_bank(group_bank, group_queries, bank_schema, query_schema, config))
    if not outputs:
        return _empty_prediction_frame(query_df, query_schema)
    return pd.concat(outputs, ignore_index=True)


def _predict_single_bank(
    bank_df: pd.DataFrame,
    query_df: pd.DataFrame,
    bank_schema: DatasetSchema,
    query_schema: DatasetSchema,
    config: dict[str, Any],
) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if bank_schema.answer_col is None:
        raise ValueError("Retrieval bank requires answer labels")
    if query_df.empty:
        return _empty_prediction_frame(query_df, query_schema)

    vectorizer = TfidfVectorizer(
        analyzer=str(config.get("analyzer", "char_wb")),
        ngram_range=(int(config.get("ngram_min", 3)), int(config.get("ngram_max", 5))),
        max_features=int(config.get("max_features", 250000)),
        min_df=int(config.get("min_df", 1)),
        max_df=float(config.get("max_df", 1.0)),
        lowercase=bool(config.get("lowercase", True)),
        sublinear_tf=bool(config.get("sublinear_tf", False)),
        binary=bool(config.get("binary", False)),
        use_idf=bool(config.get("use_idf", True)),
        smooth_idf=bool(config.get("smooth_idf", True)),
        norm=config.get("norm", "l2"),
        strip_accents="unicode",
        preprocessor=_normalize_text,
    )
    bank_questions = bank_df[bank_schema.question_col].fillna("").astype(str)
    query_questions = query_df[query_schema.question_col].fillna("").astype(str)
    bank_matrix = vectorizer.fit_transform(bank_questions)
    query_matrix = vectorizer.transform(query_questions)

    answers = bank_df[bank_schema.answer_col].fillna("").astype(str).reset_index(drop=True)  # type: ignore[index]
    bank_ids = bank_df[bank_schema.id_col].reset_index(drop=True)
    batch_size = int(config.get("batch_size", 512))
    predictions: list[str] = []
    matched_ids: list[object] = []
    similarities: list[float] = []

    for start in range(0, query_matrix.shape[0], batch_size):
        batch = query_matrix[start : start + batch_size]
        scores = cosine_similarity(batch, bank_matrix)
        best_positions = scores.argmax(axis=1)
        best_scores = scores.max(axis=1)
        for position, score in zip(best_positions, best_scores, strict=True):
            predictions.append(answers.iloc[int(position)])
            matched_ids.append(bank_ids.iloc[int(position)])
            similarities.append(float(score))

    output = pd.DataFrame(
        {
            "ID": query_df[query_schema.id_col].to_numpy(),
            "matched_id": matched_ids,
            "similarity": similarities,
            "prediction": predictions,
        }
    )
    if query_schema.answer_col:
        output["reference"] = query_df[query_schema.answer_col].to_numpy()
    return output


def _empty_prediction_frame(query_df: pd.DataFrame, query_schema: DatasetSchema) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "ID": query_df[query_schema.id_col].to_numpy(),
            "matched_id": [],
            "similarity": [],
            "prediction": [],
        }
    )
    if query_schema.answer_col:
        output["reference"] = []
    return output


def _test_retrieval_bank(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if bool(config.get("include_val_for_test", True)):
        return pd.concat([train_df, val_df], ignore_index=True)
    return train_df


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()
