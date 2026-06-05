"""Train a candidate-level reranker over union retrieval candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold

from health_qa.config import data_config_from_mapping, load_yaml
from health_qa.data import DatasetSchema, infer_schema, load_csv
from health_qa.metrics import score_predictions
from health_qa.retrieval import _normalize_text
from health_qa.submission import build_submission, save_submission

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_union_rerank import (
    _candidate_generators,
    _generator_config,
    _load_or_encode,
    _prefix_texts,
    _vectorize,
)


SUBSET_CODES = {
    "Aka_Gha": 0,
    "Amh_Eth": 1,
    "Eng_Eth": 2,
    "Eng_Gha": 3,
    "Eng_Ken": 4,
    "Eng_Uga": 5,
    "Lug_Uga": 6,
    "Swa_Ken": 7,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned reranker over retrieval candidates")
    parser.add_argument("--config", default="configs/local_retrieval_mpnet_union_rerank.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k-per-generator", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    config = load_yaml(args.config)
    config["semantic_rerank"]["top_k_per_generator"] = args.top_k_per_generator
    config["semantic_rerank"]["max_candidates"] = args.max_candidates
    data_config = data_config_from_mapping(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_csv(data_config.train_path)
    val_df = load_csv(data_config.val_path)
    test_df = load_csv(data_config.test_path)
    train_schema = infer_schema(train_df, require_answer=True)
    val_schema = infer_schema(val_df, require_answer=True)
    test_schema = infer_schema(test_df, require_answer=False)

    val_candidates = _candidate_frame(
        train_df,
        val_df,
        train_schema,
        val_schema,
        config,
        output_dir / "cache" / "validation",
        include_targets=True,
    )
    cv_predictions = _cross_validated_predictions(
        val_candidates,
        val_df,
        val_schema,
        folds=args.folds,
    )
    _write_validation(output_dir, val_df, val_schema, cv_predictions)

    test_bank = pd.concat([train_df, val_df], ignore_index=True)
    test_bank_schema = infer_schema(test_bank, require_answer=True)
    test_candidates = _candidate_frame(
        test_bank,
        test_df,
        test_bank_schema,
        test_schema,
        config,
        output_dir / "cache" / "test",
        include_targets=False,
    )
    final_model = _fit_model(val_candidates)
    test_predictions = _select_predictions(test_candidates, final_model)
    submission = build_submission(test_df[test_schema.id_col], test_predictions["prediction"].tolist())
    save_submission(submission, output_dir / "submission.csv")
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    print(f"Metrics: {output_dir / 'metrics.csv'}")
    print(f"Validation predictions: {output_dir / 'validation_predictions.csv'}")
    print(f"Submission: {output_dir / 'submission.csv'}")


def _candidate_frame(
    bank_df: pd.DataFrame,
    query_df: pd.DataFrame,
    bank_schema: DatasetSchema,
    query_schema: DatasetSchema,
    config: dict[str, Any],
    cache_dir: Path,
    *,
    include_targets: bool,
) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    rerank_config = config["semantic_rerank"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(
        rerank_config["model_name"],
        local_files_only=bool(rerank_config.get("local_files_only", False)),
    )
    bank_embeddings = _load_or_encode(
        model,
        _prefix_texts(
            bank_df[bank_schema.question_col].fillna("").astype(str).tolist(),
            str(rerank_config.get("bank_prefix", "")),
        ),
        cache_dir / "bank_embeddings.npy",
        batch_size=int(rerank_config.get("encode_batch_size", 128)),
    )
    query_embeddings = _load_or_encode(
        model,
        _prefix_texts(
            query_df[query_schema.question_col].fillna("").astype(str).tolist(),
            str(rerank_config.get("query_prefix", "")),
        ),
        cache_dir / "query_embeddings.npy",
        batch_size=int(rerank_config.get("encode_batch_size", 128)),
    )

    group_col = str(config["retrieval"].get("group_col", "subset"))
    generators = _candidate_generators(config["retrieval"])
    top_k = int(rerank_config["top_k_per_generator"])
    max_candidates = int(rerank_config["max_candidates"])
    rows: list[pd.DataFrame] = []
    for group_value, group_queries in query_df.groupby(group_col, sort=False):
        print(f"Building candidates for {group_value} ({len(group_queries)} rows)", flush=True)
        group_positions = group_queries.index.to_numpy()
        candidate_maps = [dict() for _ in range(len(group_queries))]
        for generator_index, generator in enumerate(generators):
            vectorizer_config = _generator_config(generator, group_value)
            bank_matrix, query_matrix = _vectorize(bank_df, group_queries, bank_schema, query_schema, vectorizer_config)
            _add_tracked_tfidf_candidates(
                candidate_maps,
                bank_matrix,
                query_matrix,
                generator_index=generator_index,
                top_k=top_k,
                batch_size=int(rerank_config.get("tfidf_batch_size", 256)),
            )
        rows.append(
            _group_candidate_frame(
                bank_df,
                group_queries,
                bank_schema,
                query_schema,
                candidate_maps,
                bank_embeddings,
                query_embeddings,
                group_positions,
                str(group_value),
                max_candidates=max_candidates,
                include_targets=include_targets,
            )
        )
    return pd.concat(rows, ignore_index=True)


def _group_candidate_frame(
    bank_df: pd.DataFrame,
    group_queries: pd.DataFrame,
    bank_schema: DatasetSchema,
    query_schema: DatasetSchema,
    candidate_maps: list[dict[int, float]],
    bank_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    group_positions: np.ndarray,
    subset: str,
    *,
    max_candidates: int,
    include_targets: bool,
) -> pd.DataFrame:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    rows: list[dict[str, object]] = []
    for row_offset, candidate_map in enumerate(candidate_maps):
        query_position = int(group_positions[row_offset])
        query = group_queries.iloc[row_offset]
        candidates, candidate_stats = _ranked_tracked_candidates(candidate_map, max_candidates=max_candidates)
        tfidf_scores = np.asarray([stats["max_score"] for stats in candidate_stats], dtype=np.float32)
        semantic_scores = bank_embeddings[candidates] @ query_embeddings[query_position]
        query_text = str(query[query_schema.question_col])
        query_tokens = _tokens(query_text)
        query_words = len(query_text.split())
        reference = str(query[query_schema.answer_col]) if include_targets else None
        for rank, (candidate_position, stats, tfidf_score, semantic_score) in enumerate(
            zip(candidates, candidate_stats, tfidf_scores, semantic_scores, strict=True),
            start=1,
        ):
            bank_row = bank_df.iloc[int(candidate_position)]
            bank_question = str(bank_row[bank_schema.question_col])
            answer = str(bank_row[bank_schema.answer_col])  # type: ignore[index]
            bank_question_tokens = _tokens(bank_question)
            answer_tokens = _tokens(answer)
            bank_words = len(bank_question.split())
            answer_words = len(answer.split())
            row = {
                "ID": query[query_schema.id_col],
                "subset": subset,
                "subset_code": SUBSET_CODES.get(subset, -1),
                "matched_id": bank_row[bank_schema.id_col],
                "candidate_position": int(candidate_position),
                "rank": rank,
                "tfidf_score": float(tfidf_score),
                "semantic_score": float(semantic_score),
                "generator_count": int(stats["count"]),
                "mean_tfidf_score": float(stats["score_sum"] / max(stats["count"], 1)),
                "best_generator_index": int(stats["best_generator"]),
                "query_words": query_words,
                "bank_question_words": bank_words,
                "answer_words": answer_words,
                "question_word_delta": abs(query_words - bank_words),
                "answer_to_query_ratio": answer_words / max(query_words, 1),
                "question_token_overlap": _overlap_ratio(query_tokens, bank_question_tokens),
                "question_token_jaccard": _jaccard(query_tokens, bank_question_tokens),
                "answer_query_overlap": _overlap_ratio(query_tokens, answer_tokens),
                "answer_query_jaccard": _jaccard(query_tokens, answer_tokens),
                "prediction": answer,
            }
            if include_targets:
                scores = scorer.score(reference or "", answer)
                row["target"] = 0.5 * scores["rouge1"].fmeasure + 0.5 * scores["rougeL"].fmeasure
            rows.append(row)
    return pd.DataFrame(rows)


def _cross_validated_predictions(
    candidates: pd.DataFrame,
    val_df: pd.DataFrame,
    val_schema: DatasetSchema,
    *,
    folds: int,
) -> pd.DataFrame:
    row_ids = val_df[val_schema.id_col].to_numpy()
    groups = pd.Series(np.arange(len(row_ids)), index=row_ids)
    candidates = candidates.copy()
    candidates["row_group"] = candidates["ID"].map(groups).astype(int)
    selected_frames = []
    splitter = GroupKFold(n_splits=folds)
    for fold, (train_idx, holdout_idx) in enumerate(
        splitter.split(row_ids, groups=np.arange(len(row_ids))),
        start=1,
    ):
        train_ids = set(row_ids[train_idx])
        holdout_ids = set(row_ids[holdout_idx])
        print(f"Training fold {fold}/{folds}", flush=True)
        model = _fit_model(candidates[candidates["ID"].isin(train_ids)])
        selected_frames.append(_select_predictions(candidates[candidates["ID"].isin(holdout_ids)], model))
    predictions = pd.concat(selected_frames, ignore_index=True)
    return val_df[[val_schema.id_col]].merge(predictions[["ID", "prediction"]], on="ID", how="left")


def _fit_model(candidates: pd.DataFrame) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=220,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=42,
    )
    model.fit(_features(candidates), candidates["target"].to_numpy())
    return model


def _select_predictions(candidates: pd.DataFrame, model: HistGradientBoostingRegressor) -> pd.DataFrame:
    scored = candidates.copy()
    scored["rerank_score"] = model.predict(_features(scored))
    scored = scored.sort_values(["ID", "rerank_score", "semantic_score", "tfidf_score"], ascending=[True, False, False, False])
    return scored.groupby("ID", sort=False).head(1)[["ID", "matched_id", "rerank_score", "prediction"]].reset_index(drop=True)


def _features(candidates: pd.DataFrame) -> np.ndarray:
    columns = [
        "subset_code",
        "rank",
        "tfidf_score",
        "semantic_score",
        "generator_count",
        "mean_tfidf_score",
        "best_generator_index",
        "query_words",
        "bank_question_words",
        "answer_words",
        "question_word_delta",
        "answer_to_query_ratio",
        "question_token_overlap",
        "question_token_jaccard",
        "answer_query_overlap",
        "answer_query_jaccard",
    ]
    features = candidates[columns].astype(float).to_numpy()
    features[:, 1] = 1.0 / features[:, 1]
    return features


def _add_tracked_tfidf_candidates(
    candidate_maps: list[dict[int, dict[str, float]]],
    bank_matrix,
    query_matrix,
    *,
    generator_index: int,
    top_k: int,
    batch_size: int,
) -> None:
    for start in range(0, query_matrix.shape[0], batch_size):
        scores = cosine_similarity(query_matrix[start : start + batch_size], bank_matrix)
        local_top_k = min(top_k, scores.shape[1])
        local_candidate_positions = np.argpartition(-scores, kth=local_top_k - 1, axis=1)[:, :local_top_k]
        for row_offset, local_candidates in enumerate(local_candidate_positions):
            query_candidates = candidate_maps[start + row_offset]
            for candidate_position in local_candidates:
                score = float(scores[row_offset, candidate_position])
                candidate = query_candidates.setdefault(
                    int(candidate_position),
                    {
                        "max_score": -1.0,
                        "score_sum": 0.0,
                        "count": 0.0,
                        "best_generator": float(generator_index),
                    },
                )
                candidate["score_sum"] += score
                candidate["count"] += 1.0
                if score > candidate["max_score"]:
                    candidate["max_score"] = score
                    candidate["best_generator"] = float(generator_index)


def _ranked_tracked_candidates(
    candidate_map: dict[int, dict[str, float]],
    *,
    max_candidates: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    ranked = sorted(candidate_map.items(), key=lambda item: item[1]["max_score"], reverse=True)[:max_candidates]
    candidates = np.asarray([candidate for candidate, _ in ranked], dtype=np.int64)
    stats = [stats for _, stats in ranked]
    return candidates, stats


def _tokens(text: str) -> set[str]:
    return {token for token in _normalize_text(text).split() if token}


def _overlap_ratio(source: set[str], candidate: set[str]) -> float:
    if not source:
        return 0.0
    return len(source & candidate) / len(source)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _write_validation(
    output_dir: Path,
    val_df: pd.DataFrame,
    val_schema: DatasetSchema,
    predictions: pd.DataFrame,
) -> None:
    if predictions["prediction"].isna().any():
        raise ValueError("Cross-validated predictions contain missing values")
    metrics = score_predictions(
        val_df[val_schema.answer_col].fillna("").astype(str).tolist(),  # type: ignore[index]
        predictions["prediction"].fillna("").astype(str).tolist(),
    )
    validation = predictions.copy()
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


if __name__ == "__main__":
    main()
