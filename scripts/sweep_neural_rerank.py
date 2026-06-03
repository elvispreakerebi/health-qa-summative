"""Sweep neural rerank depth and blend weights by subset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from health_qa.config import data_config_from_mapping, load_yaml
from health_qa.data import infer_schema, load_csv
from health_qa.retrieval import _normalize_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep neural rerank hyperparameters")
    parser.add_argument("--config", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", default="5,10,20,30")
    parser.add_argument("--weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--subsets", default="")
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_config = data_config_from_mapping(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_csv(data_config.train_path)
    val_df = load_csv(data_config.val_path)
    train_schema = infer_schema(train_df, require_answer=True)
    val_schema = infer_schema(val_df, require_answer=True)

    cache_dir = Path(args.embedding_cache)
    bank_embeddings = np.load(cache_dir / "bank_embeddings.npy")
    query_embeddings = np.load(cache_dir / "query_embeddings.npy")

    top_k_values = [int(value) for value in args.top_k.split(",")]
    weights = [float(value) for value in args.weights.split(",")]
    subsets = {value.strip() for value in args.subsets.split(",") if value.strip()}
    max_top_k = max(top_k_values)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

    group_col = str(config["retrieval"].get("group_col", "subset"))
    default_config = dict(config["retrieval"]["default"])
    group_configs = config["retrieval"].get("group_configs", {})
    rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    for group_value, group_queries in val_df.groupby(group_col, sort=False):
        if subsets and str(group_value) not in subsets:
            continue
        vectorizer_config = dict(default_config)
        vectorizer_config.update(group_configs.get(group_value, {}))
        bank_matrix, query_matrix = _vectorize(train_df, group_queries, train_schema, val_schema, vectorizer_config)
        scores = cosine_similarity(query_matrix, bank_matrix)
        candidates = np.argpartition(
            -scores,
            kth=min(max_top_k - 1, scores.shape[1] - 1),
            axis=1,
        )[:, :max_top_k]
        ordered_candidates = np.empty_like(candidates)
        ordered_tfidf = np.empty_like(candidates, dtype=float)
        ordered_semantic = np.empty_like(candidates, dtype=float)
        query_positions = group_queries.index.to_numpy()

        for row_idx, row_candidates in enumerate(candidates):
            sorted_candidates = row_candidates[np.argsort(-scores[row_idx, row_candidates])]
            ordered_candidates[row_idx] = sorted_candidates
            ordered_tfidf[row_idx] = scores[row_idx, sorted_candidates]
            ordered_semantic[row_idx] = bank_embeddings[sorted_candidates] @ query_embeddings[int(query_positions[row_idx])]

        references = group_queries[val_schema.answer_col].fillna("").astype(str).tolist()  # type: ignore[index]
        best_for_group: dict[str, Any] | None = None
        for top_k in top_k_values:
            top_candidates = ordered_candidates[:, :top_k]
            tfidf_scores = ordered_tfidf[:, :top_k]
            semantic_scores = ordered_semantic[:, :top_k]
            for weight in weights:
                selected = ((1 - weight) * tfidf_scores + weight * semantic_scores).argmax(axis=1)
                predictions = [
                    str(train_df.iloc[int(top_candidates[row_idx, selected_idx])][train_schema.answer_col])  # type: ignore[index]
                    for row_idx, selected_idx in enumerate(selected)
                ]
                rouge1, rouge_l = _score_fast(scorer, references, predictions)
                row = {
                    "subset": group_value,
                    "n": len(group_queries),
                    "top_k": top_k,
                    "semantic_weight": weight,
                    "rouge1_f1": rouge1,
                    "rouge_l_f1": rouge_l,
                    "weighted_without_llm": 0.5 * rouge1 + 0.5 * rouge_l,
                }
                rows.append(row)
                if best_for_group is None or row["weighted_without_llm"] > best_for_group["weighted_without_llm"]:
                    best_for_group = row
        assert best_for_group is not None
        best_rows.append(best_for_group)
        print(
            f"{group_value}: {best_for_group['weighted_without_llm']:.6f} "
            f"top_k={best_for_group['top_k']} weight={best_for_group['semantic_weight']}",
            flush=True,
        )

    results = pd.DataFrame(rows).sort_values(["subset", "weighted_without_llm"], ascending=[True, False])
    best = pd.DataFrame(best_rows).sort_values("subset")
    results.to_csv(output_dir / "topk_weight_results.csv", index=False)
    best.to_csv(output_dir / "best_by_subset.csv", index=False)
    weighted = float((best["weighted_without_llm"] * best["n"]).sum() / best["n"].sum())
    pd.DataFrame([{"weighted_subset_best": weighted}]).to_csv(output_dir / "summary.csv", index=False)
    print(f"Weighted subset best: {weighted:.6f}", flush=True)


def _score_fast(scorer: rouge_scorer.RougeScorer, references: list[str], predictions: list[str]) -> tuple[float, float]:
    rouge1 = []
    rouge_l = []
    for reference, prediction in zip(references, predictions, strict=True):
        scores = scorer.score(reference, prediction)
        rouge1.append(scores["rouge1"].fmeasure)
        rouge_l.append(scores["rougeL"].fmeasure)
    return float(np.mean(rouge1)), float(np.mean(rouge_l))


def _vectorize(
    bank_df: pd.DataFrame,
    query_df: pd.DataFrame,
    bank_schema,
    query_schema,
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


if __name__ == "__main__":
    main()
