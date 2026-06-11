"""Run AfriE5 retrieval with optional BGE cross-encoder reranking."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from health_qa.config import data_config_from_mapping, load_yaml
from health_qa.data import DatasetSchema, infer_schema, load_csv
from health_qa.metrics import score_predictions
from health_qa.retrieval import _normalize_text
from health_qa.submission import build_submission, save_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AfriE5 retrieval plus BGE reranking")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--no-cross-encoder", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.no_cross_encoder:
        config.setdefault("cross_encoder", {})["enabled"] = False

    data_config = data_config_from_mapping(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_csv(data_config.train_path).reset_index(drop=True)
    val_df = load_csv(data_config.val_path).reset_index(drop=True)
    test_df = load_csv(data_config.test_path).reset_index(drop=True)
    if args.max_val_rows:
        val_df = val_df.head(args.max_val_rows).copy()

    train_schema = infer_schema(train_df, require_answer=True)
    val_schema = infer_schema(val_df, require_answer=True)
    test_schema = infer_schema(test_df, require_answer=False)

    validation = predict_with_afrie5_bge(
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

    if not args.skip_test:
        test_bank = pd.concat([train_df, val_df], ignore_index=True)
        test_bank_schema = infer_schema(test_bank, require_answer=True)
        test_predictions = predict_with_afrie5_bge(
            test_bank,
            test_df,
            test_bank_schema,
            test_schema,
            config,
            cache_dir=output_dir / "cache" / "test",
        )
        submission = build_submission(test_df[test_schema.id_col], test_predictions["prediction"].tolist())
        save_submission(submission, output_dir / "submission.csv")
        test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
        print(f"Submission: {output_dir / 'submission.csv'}")

    print(f"Validation predictions: {output_dir / 'validation_predictions.csv'}")
    print(f"Metrics: {output_dir / 'metrics.csv'}")


def predict_with_afrie5_bge(
    bank_df: pd.DataFrame,
    query_df: pd.DataFrame,
    bank_schema: DatasetSchema,
    query_schema: DatasetSchema,
    config: dict[str, Any],
    *,
    cache_dir: Path,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    retrieval_config = config["retrieval"]
    embedding_config = config["embedding"]
    cross_encoder_config = config.get("cross_encoder", {})
    group_col = str(retrieval_config.get("group_col", "subset"))
    candidate_scope = str(retrieval_config.get("candidate_scope", "same_group"))
    top_k_dense = int(retrieval_config.get("dense_top_k", 80))
    top_k_tfidf = int(retrieval_config.get("tfidf_top_k", 40))
    max_candidates = int(retrieval_config.get("max_candidates", 120))
    dense_weight = float(retrieval_config.get("dense_weight", 1.0))
    tfidf_weight = float(retrieval_config.get("tfidf_weight", 0.15))

    bank_embeddings, query_embeddings = _load_embeddings(
        bank_df,
        query_df,
        bank_schema,
        query_schema,
        embedding_config,
        cache_dir,
    )
    reranker = _CrossEncoderReranker(cross_encoder_config)
    predictions_by_position: dict[int, dict[str, object]] = {}

    for group_value, group_queries in query_df.groupby(group_col, sort=False):
        print(f"Predicting {group_value} ({len(group_queries)} rows)", flush=True)
        candidate_bank = _candidate_bank(bank_df, group_col, group_value, candidate_scope)
        candidate_positions = candidate_bank.index.to_numpy(dtype=np.int64)
        candidate_embeddings = bank_embeddings[candidate_positions]
        group_positions = group_queries.index.to_numpy(dtype=np.int64)
        tfidf_bank_matrix, tfidf_query_matrix = _vectorize(
            candidate_bank,
            group_queries,
            bank_schema,
            query_schema,
            retrieval_config.get("tfidf", {}),
        )
        batch_size = int(retrieval_config.get("similarity_batch_size", 128))
        for start in range(0, len(group_positions), batch_size):
            stop = min(start + batch_size, len(group_positions))
            query_positions = group_positions[start:stop]
            dense_scores = query_embeddings[query_positions] @ candidate_embeddings.T
            tfidf_scores = cosine_similarity(tfidf_query_matrix[start:stop], tfidf_bank_matrix)
            for local_offset, query_position in enumerate(query_positions):
                candidate_scores = _candidate_scores(
                    candidate_positions,
                    dense_scores[local_offset],
                    tfidf_scores[local_offset],
                    top_k_dense=top_k_dense,
                    top_k_tfidf=top_k_tfidf,
                    max_candidates=max_candidates,
                    dense_weight=dense_weight,
                    tfidf_weight=tfidf_weight,
                )
                best_position, best_score, cross_score = _select_candidate(
                    bank_df,
                    query_df.loc[int(query_position), query_schema.question_col],
                    bank_schema,
                    candidate_scores,
                    reranker,
                    cross_encoder_config,
                )
                predictions_by_position[int(query_position)] = {
                    "ID": query_df.loc[int(query_position), query_schema.id_col],
                    "matched_id": bank_df.iloc[best_position][bank_schema.id_col],
                    "similarity": best_score,
                    "cross_encoder_score": cross_score,
                    "prediction": str(bank_df.iloc[best_position][bank_schema.answer_col]),  # type: ignore[index]
                }

    output = pd.DataFrame([predictions_by_position[position] for position in query_df.index])
    if query_schema.answer_col:
        output["reference"] = query_df[query_schema.answer_col].to_numpy()
    return output


def _load_embeddings(
    bank_df: pd.DataFrame,
    query_df: pd.DataFrame,
    bank_schema: DatasetSchema,
    query_schema: DatasetSchema,
    embedding_config: dict[str, Any],
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        embedding_config["model_name"],
        local_files_only=bool(embedding_config.get("local_files_only", False)),
    )
    bank_texts = [
        format_document(row, bank_schema, embedding_config.get("document_template"))
        for _, row in bank_df.iterrows()
    ]
    query_texts = [
        format_query(str(row[query_schema.question_col]), embedding_config.get("query_prefix", ""))
        for _, row in query_df.iterrows()
    ]
    bank_embeddings = _load_or_encode(
        model,
        bank_texts,
        cache_dir / "bank_embeddings.npy",
        batch_size=int(embedding_config.get("encode_batch_size", 64)),
    )
    query_embeddings = _load_or_encode(
        model,
        query_texts,
        cache_dir / "query_embeddings.npy",
        batch_size=int(embedding_config.get("encode_batch_size", 64)),
    )
    return bank_embeddings, query_embeddings


def _candidate_scores(
    candidate_positions: np.ndarray,
    dense_scores: np.ndarray,
    tfidf_scores: np.ndarray,
    *,
    top_k_dense: int,
    top_k_tfidf: int,
    max_candidates: int,
    dense_weight: float,
    tfidf_weight: float,
) -> list[dict[str, float | int]]:
    candidate_map: dict[int, dict[str, float | int]] = {}
    _add_top_scores(candidate_map, candidate_positions, dense_scores, top_k_dense, "dense_score")
    _add_top_scores(candidate_map, candidate_positions, tfidf_scores, top_k_tfidf, "tfidf_score")
    for candidate in candidate_map.values():
        candidate["combined_score"] = (
            dense_weight * float(candidate.get("dense_score", 0.0))
            + tfidf_weight * float(candidate.get("tfidf_score", 0.0))
        )
    return sorted(
        candidate_map.values(),
        key=lambda item: float(item["combined_score"]),
        reverse=True,
    )[:max_candidates]


def _add_top_scores(
    candidate_map: dict[int, dict[str, float | int]],
    candidate_positions: np.ndarray,
    scores: np.ndarray,
    top_k: int,
    score_name: str,
) -> None:
    if top_k <= 0 or len(scores) == 0:
        return
    local_top_k = min(top_k, len(scores))
    top_indices = np.argpartition(-scores, kth=local_top_k - 1)[:local_top_k]
    for index in top_indices:
        position = int(candidate_positions[index])
        candidate = candidate_map.setdefault(
            position,
            {"candidate_position": position, "dense_score": 0.0, "tfidf_score": 0.0},
        )
        candidate[score_name] = max(float(candidate.get(score_name, 0.0)), float(scores[index]))


def _select_candidate(
    bank_df: pd.DataFrame,
    query_text: object,
    bank_schema: DatasetSchema,
    candidate_scores: list[dict[str, float | int]],
    reranker: "_CrossEncoderReranker",
    cross_encoder_config: dict[str, Any],
) -> tuple[int, float, float | None]:
    if not candidate_scores:
        raise ValueError("No candidates available for query")
    if not reranker.enabled:
        best = candidate_scores[0]
        return int(best["candidate_position"]), float(best["combined_score"]), None

    rerank_top_k = min(int(cross_encoder_config.get("top_k", len(candidate_scores))), len(candidate_scores))
    rerank_candidates = candidate_scores[:rerank_top_k]
    passages = [
        format_document(
            bank_df.iloc[int(candidate["candidate_position"])],
            bank_schema,
            cross_encoder_config.get("document_template"),
        )
        for candidate in rerank_candidates
    ]
    scores = reranker.score(str(query_text), passages)
    best_index = int(np.argmax(scores))
    best = rerank_candidates[best_index]
    return int(best["candidate_position"]), float(best["combined_score"]), float(scores[best_index])


class _CrossEncoderReranker:
    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.enabled:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = str(config["model_name"])
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=bool(config.get("local_files_only", False)),
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=bool(config.get("local_files_only", False)),
            ).to(self.device)
            self.model.eval()

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not self.enabled or self.model is None or self.tokenizer is None:
            raise RuntimeError("Cross-encoder reranker is disabled")
        batch_size = int(self.config.get("batch_size", 16))
        max_length = int(self.config.get("max_length", 512))
        scores: list[np.ndarray] = []
        for start in range(0, len(passages), batch_size):
            pairs = [[query, passage] for passage in passages[start : start + batch_size]]
            encoded = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**encoded, return_dict=True).logits.view(-1).float()
            scores.append(logits.detach().cpu().numpy())
        return np.concatenate(scores)


def _candidate_bank(
    bank_df: pd.DataFrame,
    group_col: str,
    group_value: object,
    candidate_scope: str,
) -> pd.DataFrame:
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


def format_query(question: str, query_prefix: str) -> str:
    return f"{query_prefix}{question}" if query_prefix else question


def format_document(
    row: pd.Series,
    schema: DatasetSchema,
    template: str | None,
) -> str:
    question = str(row[schema.question_col]).strip()
    answer = str(row[schema.answer_col]).strip() if schema.answer_col else ""
    subset = str(row["subset"]).strip() if "subset" in row else ""
    if template:
        return template.format(question=question, answer=answer, subset=subset)
    return f"Question: {question}\nAnswer: {answer}"


if __name__ == "__main__":
    main()
