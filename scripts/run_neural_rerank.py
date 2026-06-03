"""Generate a submission with TF-IDF retrieval reranked by multilingual embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from health_qa.config import data_config_from_mapping, load_yaml
from health_qa.data import DatasetSchema, infer_schema, load_csv
from health_qa.metrics import score_predictions
from health_qa.retrieval import _normalize_text
from health_qa.submission import build_submission, save_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Run neural reranked retrieval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_config = data_config_from_mapping(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_csv(data_config.train_path)
    val_df = load_csv(data_config.val_path)
    test_df = load_csv(data_config.test_path)

    train_schema = infer_schema(train_df, require_answer=True)
    val_schema = infer_schema(val_df, require_answer=True)
    test_schema = infer_schema(test_df, require_answer=False)

    validation = _predict_with_rerank(
        train_df,
        val_df,
        train_schema,
        val_schema,
        config,
        cache_dir=output_dir / "cache" / "validation",
    )
    metrics = score_predictions(
        val_df[val_schema.answer_col].fillna("").astype(str).tolist(),  # type: ignore[index]
        validation["prediction"].fillna("").astype(str).tolist(),
    )

    test_bank = pd.concat([train_df, val_df], ignore_index=True)
    test_bank_schema = infer_schema(test_bank, require_answer=True)
    test_predictions = _predict_with_rerank(
        test_bank,
        test_df,
        test_bank_schema,
        test_schema,
        config,
        cache_dir=output_dir / "cache" / "test",
    )

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
    submission = build_submission(test_df[test_schema.id_col], test_predictions["prediction"].tolist())
    save_submission(submission, output_dir / "submission.csv")

    print(f"Submission: {output_dir / 'submission.csv'}")
    print(f"Validation predictions: {output_dir / 'validation_predictions.csv'}")
    print(f"Metrics: {output_dir / 'metrics.csv'}")


def _predict_with_rerank(
    bank_df: pd.DataFrame,
    query_df: pd.DataFrame,
    bank_schema: DatasetSchema,
    query_schema: DatasetSchema,
    config: dict[str, Any],
    *,
    cache_dir: Path,
) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    rerank_config = config["semantic_rerank"]
    model = SentenceTransformer(
        rerank_config["model_name"],
        local_files_only=bool(rerank_config.get("local_files_only", False)),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
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

    top_k = int(rerank_config.get("top_k", 10))
    default_semantic_weight = float(rerank_config.get("semantic_weight", 0.3))
    semantic_weights_by_group = rerank_config.get("semantic_weights_by_group", {})
    predictions_by_position: dict[int, dict[str, object]] = {}
    group_col = str(config["retrieval"].get("group_col", "subset"))
    retrieval_config = config["retrieval"]
    default_config = dict(retrieval_config["default"])
    group_configs = config["retrieval"].get("group_configs", {})
    default_candidate_scope = str(retrieval_config.get("candidate_scope", "all"))
    candidate_scope_by_group = retrieval_config.get("candidate_scope_by_group", {})

    for group_value, group_queries in query_df.groupby(group_col, sort=False):
        vectorizer_config = dict(default_config)
        vectorizer_config.update(group_configs.get(group_value, {}))
        semantic_weight = float(semantic_weights_by_group.get(group_value, default_semantic_weight))
        candidate_scope = str(candidate_scope_by_group.get(group_value, default_candidate_scope))
        candidate_bank = _candidate_bank(bank_df, group_col, group_value, candidate_scope)
        candidate_positions = candidate_bank.index.to_numpy()
        group_positions = group_queries.index.to_numpy()
        bank_matrix, query_matrix = _vectorize(
            candidate_bank,
            group_queries,
            bank_schema,
            query_schema,
            vectorizer_config,
        )

        for start in range(0, query_matrix.shape[0], int(rerank_config.get("tfidf_batch_size", 256))):
            scores = cosine_similarity(query_matrix[start : start + int(rerank_config.get("tfidf_batch_size", 256))], bank_matrix)
            local_candidate_positions = np.argpartition(
                -scores,
                kth=min(top_k - 1, scores.shape[1] - 1),
                axis=1,
            )[:, :top_k]
            for row_offset, local_candidates in enumerate(local_candidate_positions):
                query_position = int(group_positions[start + row_offset])
                local_candidates = local_candidates[np.argsort(-scores[row_offset, local_candidates])]
                tfidf_scores = scores[row_offset, local_candidates]
                candidates = candidate_positions[local_candidates]
                semantic_scores = bank_embeddings[candidates] @ query_embeddings[query_position]
                combined_scores = (1 - semantic_weight) * tfidf_scores + semantic_weight * semantic_scores
                best_position = int(candidates[int(combined_scores.argmax())])
                predictions_by_position[query_position] = {
                    "ID": query_df.loc[query_position, query_schema.id_col],
                    "matched_id": bank_df.iloc[best_position][bank_schema.id_col],
                    "similarity": float(combined_scores.max()),
                    "prediction": str(bank_df.iloc[best_position][bank_schema.answer_col]),  # type: ignore[index]
                }

    rows = [predictions_by_position[position] for position in query_df.index]
    output = pd.DataFrame(rows)
    if query_schema.answer_col:
        output["reference"] = query_df[query_schema.answer_col].to_numpy()
    return output


def _candidate_bank(bank_df: pd.DataFrame, group_col: str, group_value: object, candidate_scope: str) -> pd.DataFrame:
    if candidate_scope == "all":
        return bank_df
    if candidate_scope == "same_group":
        return bank_df[bank_df[group_col] == group_value]
    raise ValueError(f"Unsupported candidate_scope: {candidate_scope}")


def _vectorize(
    bank_df: pd.DataFrame,
    query_df: pd.DataFrame,
    bank_schema: DatasetSchema,
    query_schema: DatasetSchema,
    config: dict[str, Any],
):
    vectorizer = TfidfVectorizer(
        analyzer=str(config.get("analyzer", "char")),
        ngram_range=(int(config.get("ngram_min", 2)), int(config.get("ngram_max", 4))),
        max_features=int(config.get("max_features", 100000)),
        min_df=int(config.get("min_df", 1)),
        max_df=float(config.get("max_df", 1.0)),
        lowercase=bool(config.get("lowercase", True)),
        sublinear_tf=bool(config.get("sublinear_tf", True)),
        binary=bool(config.get("binary", False)),
        use_idf=bool(config.get("use_idf", True)),
        smooth_idf=bool(config.get("smooth_idf", True)),
        norm=config.get("norm", "l2"),
        strip_accents="unicode",
        preprocessor=_normalize_text,
    )
    bank_matrix = vectorizer.fit_transform(bank_df[bank_schema.question_col].fillna("").astype(str))
    query_matrix = vectorizer.transform(query_df[query_schema.question_col].fillna("").astype(str))
    return bank_matrix, query_matrix


def _load_or_encode(model, texts: list[str], path: Path, *, batch_size: int) -> np.ndarray:
    if path.exists():
        return np.load(path)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(path, embeddings)
    return embeddings


def _prefix_texts(texts: list[str], prefix: str) -> list[str]:
    if not prefix:
        return texts
    return [f"{prefix}{text}" for text in texts]


if __name__ == "__main__":
    main()
