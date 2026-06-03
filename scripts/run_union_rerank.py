"""Generate retrieval predictions from a union of TF-IDF candidate generators."""

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
    parser = argparse.ArgumentParser(description="Run union-candidate neural reranked retrieval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k-per-generator", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.top_k_per_generator is not None:
        config["semantic_rerank"]["top_k_per_generator"] = args.top_k_per_generator
    if args.max_candidates is not None:
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

    validation = _predict_with_union_rerank(
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
    test_predictions = _predict_with_union_rerank(
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


def _predict_with_union_rerank(
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

    group_col = str(config["retrieval"].get("group_col", "subset"))
    generators = _candidate_generators(config["retrieval"])
    rerank_batch_size = int(rerank_config.get("tfidf_batch_size", 256))
    top_k_per_generator = int(rerank_config.get("top_k_per_generator", 12))
    max_candidates = int(rerank_config.get("max_candidates", 60))
    default_semantic_weight = float(rerank_config.get("semantic_weight", 0.3))
    semantic_weights_by_group = rerank_config.get("semantic_weights_by_group", {})

    predictions_by_position: dict[int, dict[str, object]] = {}
    for group_value, group_queries in query_df.groupby(group_col, sort=False):
        group_positions = group_queries.index.to_numpy()
        candidate_maps = [dict() for _ in range(len(group_queries))]

        for generator in generators:
            vectorizer_config = _generator_config(generator, group_value)
            bank_matrix, query_matrix = _vectorize(bank_df, group_queries, bank_schema, query_schema, vectorizer_config)
            _add_tfidf_candidates(
                candidate_maps,
                bank_matrix,
                query_matrix,
                top_k=top_k_per_generator,
                batch_size=rerank_batch_size,
            )

        semantic_weight = float(semantic_weights_by_group.get(group_value, default_semantic_weight))
        for row_offset, candidate_map in enumerate(candidate_maps):
            query_position = int(group_positions[row_offset])
            candidates, tfidf_scores = _ranked_candidates(candidate_map, max_candidates=max_candidates)
            semantic_scores = bank_embeddings[candidates] @ query_embeddings[query_position]
            combined_scores = (1 - semantic_weight) * tfidf_scores + semantic_weight * semantic_scores
            best_position = int(candidates[int(combined_scores.argmax())])
            predictions_by_position[query_position] = {
                "ID": query_df.loc[query_position, query_schema.id_col],
                "matched_id": bank_df.iloc[best_position][bank_schema.id_col],
                "similarity": float(combined_scores.max()),
                "prediction": str(bank_df.iloc[best_position][bank_schema.answer_col]),  # type: ignore[index]
            }

    output = pd.DataFrame([predictions_by_position[position] for position in query_df.index])
    if query_schema.answer_col:
        output["reference"] = query_df[query_schema.answer_col].to_numpy()
    return output


def _candidate_generators(retrieval_config: dict[str, Any]) -> list[dict[str, Any]]:
    generators = retrieval_config.get("candidate_generators")
    if generators:
        return list(generators)
    return [
        {
            "name": "default",
            "default": retrieval_config["default"],
            "group_configs": retrieval_config.get("group_configs", {}),
        }
    ]


def _generator_config(generator: dict[str, Any], group_value: object) -> dict[str, Any]:
    vectorizer_config = dict(generator["default"])
    vectorizer_config.update(generator.get("group_configs", {}).get(group_value, {}))
    return vectorizer_config


def _add_tfidf_candidates(
    candidate_maps: list[dict[int, float]],
    bank_matrix,
    query_matrix,
    *,
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
                previous = query_candidates.get(int(candidate_position), -1.0)
                if score > previous:
                    query_candidates[int(candidate_position)] = score


def _ranked_candidates(candidate_map: dict[int, float], *, max_candidates: int) -> tuple[np.ndarray, np.ndarray]:
    ranked = sorted(candidate_map.items(), key=lambda item: item[1], reverse=True)[:max_candidates]
    candidates = np.asarray([candidate for candidate, _ in ranked], dtype=np.int64)
    scores = np.asarray([score for _, score in ranked], dtype=np.float32)
    return candidates, scores


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
