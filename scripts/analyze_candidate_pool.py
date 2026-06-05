"""Compare saved validation candidates against a baseline run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank saved validation prediction candidates")
    parser.add_argument("--val-path", default="data/raw/Val.csv")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    val = pd.read_csv(args.val_path)
    ids = val["ID"].tolist()
    refs = val.set_index("ID")["output"].fillna("").astype(str)
    subsets = sorted(val["subset"].dropna().astype(str).unique())
    baseline = _load_predictions(Path(args.baseline_dir) / "validation_predictions.csv", ids)
    baseline_scores = _row_scores(refs, baseline, ids)
    baseline_score = float(baseline_scores["score"].mean())

    rows: list[dict[str, object]] = []
    for path in sorted(Path(args.outputs_dir).glob("*/validation_predictions.csv")):
        try:
            pred = _load_predictions(path, ids)
        except ValueError:
            continue
        candidate_scores = _row_scores(refs, pred, ids)
        row: dict[str, object] = {
            "path": str(path.parent),
            "score": float(candidate_scores["score"].mean()),
        }
        row["delta"] = float(row["score"]) - baseline_score
        for subset in subsets:
            subset_ids = val.loc[val["subset"].eq(subset), "ID"].tolist()
            row[subset] = float(candidate_scores.loc[subset_ids, "score"].mean())
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("score", ascending=False)
    print(f"Baseline: {args.baseline_dir} ({baseline_score:.6f})")
    print(results[["path", "score", "delta"]].head(args.top).to_string(index=False))
    print("\nSubset leaders:")
    for subset in subsets:
        print(f"\n{subset}")
        print(results.sort_values(subset, ascending=False)[["path", subset, "score"]].head(5).to_string(index=False))


def _load_predictions(path: Path, ids: list[str]) -> pd.Series:
    if not path.exists():
        raise ValueError(f"Missing {path}")
    df = pd.read_csv(path)
    if "prediction" not in df.columns or "ID" not in df.columns:
        raise ValueError(f"Unsupported prediction format: {path}")
    pred = df.set_index("ID")["prediction"].fillna("").astype(str)
    missing = set(ids).difference(pred.index)
    if missing:
        raise ValueError(f"{path} is missing validation IDs")
    return pred


def _row_scores(refs: pd.Series, pred: pd.Series, ids: list[str]) -> pd.DataFrame:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    rows = []
    for row_id in ids:
        scores = scorer.score(str(refs.loc[row_id]), str(pred.loc[row_id]))
        weighted = 0.5 * scores["rouge1"].fmeasure + 0.5 * scores["rougeL"].fmeasure
        rows.append({"ID": row_id, "score": weighted})
    return pd.DataFrame(rows).set_index("ID")


if __name__ == "__main__":
    main()
