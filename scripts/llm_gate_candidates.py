"""Use a local Ollama judge to gate risky candidate overrides."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

from health_qa.config import data_config_from_mapping, load_yaml
from health_qa.data import infer_schema, load_csv
from health_qa.metrics import score_predictions
from health_qa.submission import build_submission, save_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend candidates only when a local LLM judge approves them")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--subsets", default="Eng_Uga,Eng_Eth")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--min-confidence", type=float, default=0.68)
    parser.add_argument("--limit-per-subset", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_config = data_config_from_mapping(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subsets = {subset.strip() for subset in args.subsets.split(",") if subset.strip()}

    val_df = load_csv(data_config.val_path)
    val_schema = infer_schema(val_df, require_answer=True)
    val_predictions = _blend_split(
        val_df,
        val_schema.id_col,
        val_schema.question_col,
        Path(args.base_dir) / "validation_predictions.csv",
        Path(args.candidate_dir) / "validation_predictions.csv",
        output_dir / "llm_gate_validation_raw.csv",
        model=args.model,
        subsets=subsets,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        min_confidence=args.min_confidence,
        limit_per_subset=args.limit_per_subset,
    )
    metrics = score_predictions(
        val_df[val_schema.answer_col].fillna("").astype(str).tolist(),  # type: ignore[index]
        val_predictions["prediction"].fillna("").astype(str).tolist(),
    )
    val_predictions["reference"] = val_df[val_schema.answer_col].to_numpy()  # type: ignore[index]
    val_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "rouge1_f1": metrics.rouge1_f1,
                "rouge_l_f1": metrics.rouge_l_f1,
                "weighted_without_llm": metrics.weighted_without_llm,
                "switched_rows": int(val_predictions["used_candidate"].sum()),
            }
        ]
    ).to_csv(output_dir / "metrics.csv", index=False)

    if not args.skip_test:
        test_df = load_csv(data_config.test_path)
        test_schema = infer_schema(test_df, require_answer=False)
        test_predictions = _blend_split(
            test_df,
            test_schema.id_col,
            test_schema.question_col,
            Path(args.base_dir) / "submission.csv",
            Path(args.candidate_dir) / "submission.csv",
            output_dir / "llm_gate_test_raw.csv",
            model=args.model,
            subsets=subsets,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            min_confidence=args.min_confidence,
            submission_mode=True,
            limit_per_subset=args.limit_per_subset,
        )
        test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
        save_submission(build_submission(test_predictions["ID"], test_predictions["prediction"]), output_dir / "submission.csv")

    print(f"Metrics: {output_dir / 'metrics.csv'}")
    if not args.skip_test:
        print(f"Submission: {output_dir / 'submission.csv'}")


def _blend_split(
    source_df: pd.DataFrame,
    id_col: str,
    question_col: str,
    base_path: Path,
    candidate_path: Path,
    raw_path: Path,
    *,
    model: str,
    subsets: set[str],
    temperature: float,
    max_tokens: int,
    min_confidence: float,
    submission_mode: bool = False,
    limit_per_subset: int | None = None,
) -> pd.DataFrame:
    base = _load_predictions(base_path, submission_mode=submission_mode)
    candidate = _load_predictions(candidate_path, submission_mode=submission_mode)
    merged = (
        source_df[[id_col, question_col, "subset"]]
        .merge(base, left_on=id_col, right_on="ID", how="left")
        .merge(candidate, left_on=id_col, right_on="ID", how="left", suffixes=("_base", "_candidate"))
    )
    if merged["prediction_base"].isna().any() or merged["prediction_candidate"].isna().any():
        raise ValueError("Base and candidate predictions must cover every source ID")

    decisions = _load_raw_decisions(raw_path)
    selected_counts = {subset: 0 for subset in subsets}
    outputs: list[dict[str, object]] = []
    new_rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        row_id = str(row[id_col])
        subset = str(row["subset"])
        base_answer = str(row["prediction_base"]).strip()
        candidate_answer = str(row["prediction_candidate"]).strip()
        use_candidate = False
        confidence = 0.0
        reason = "not_target_subset"

        if subset in subsets and candidate_answer and candidate_answer != base_answer:
            if limit_per_subset is None or selected_counts[subset] < limit_per_subset:
                selected_counts[subset] += 1
                decision = decisions.get(row_id)
                if decision is None:
                    decision = _judge_candidate(
                        model,
                        question=str(row[question_col]),
                        base_answer=base_answer,
                        candidate_answer=candidate_answer,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    decisions[row_id] = decision
                    new_rows.append({"ID": row_id, **decision})
                    _append_raw_decisions(raw_path, new_rows)
                    new_rows.clear()
                    time.sleep(0.05)
                confidence = float(decision["confidence"])
                use_candidate = decision["choice"] == "B" and confidence >= min_confidence
                reason = str(decision["reason"])
            else:
                reason = "limit_reached"

        outputs.append(
            {
                "ID": row_id,
                "subset": subset,
                "prediction": candidate_answer if use_candidate else base_answer,
                "used_candidate": use_candidate,
                "confidence": confidence,
                "decision_reason": reason,
            }
        )
    return pd.DataFrame(outputs)


def _load_predictions(path: Path, *, submission_mode: bool) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    if submission_mode:
        predictions = predictions.rename(columns={"TargetRLF1": "prediction"})
    return predictions[["ID", "prediction"]].fillna("")


def _judge_candidate(
    model: str,
    *,
    question: str,
    base_answer: str,
    candidate_answer: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, object]:
    prompt = _build_prompt(question, base_answer, candidate_answer)
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        text = str(json.loads(response.read().decode("utf-8")).get("response", ""))
    decision = parse_decision(text)
    decision["raw_response"] = text[:500]
    return decision


def _build_prompt(question: str, base_answer: str, candidate_answer: str) -> str:
    return (
        "/no_think\n"
        "You are judging two candidate answers to a health question. "
        "Choose B only if it is clearly more direct, complete, and on-topic than A. "
        "If unsure, choose A. Do not explain. Return exactly one short JSON object with keys choice, confidence, reason. "
        "choice must be A or B. confidence must be between 0 and 1.\n\n"
        f"Question:\n{question}\n\nAnswer A:\n{base_answer}\n\nAnswer B:\n{candidate_answer}\n\nJSON:"
    )


def parse_decision(text: str) -> dict[str, object]:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE).strip()
    match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            choice = str(parsed.get("choice", "A")).strip().upper()
            confidence = float(parsed.get("confidence", 0.0))
            reason = str(parsed.get("reason", "json"))
            return _normalized_decision(choice, confidence, reason)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    choice_match = re.search(r"\b([AB])\b", cleaned.upper())
    choice = choice_match.group(1) if choice_match else "A"
    confidence_match = re.search(r"(?:confidence|score)\D+([01](?:\.\d+)?)", cleaned, flags=re.IGNORECASE)
    confidence = float(confidence_match.group(1)) if confidence_match else 0.0
    return _normalized_decision(choice, confidence, "fallback_parse")


def _normalized_decision(choice: str, confidence: float, reason: str) -> dict[str, object]:
    if choice not in {"A", "B"}:
        choice = "A"
    confidence = max(0.0, min(1.0, confidence))
    return {"choice": choice, "confidence": confidence, "reason": reason[:160]}


def _load_raw_decisions(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    raw = pd.read_csv(path).fillna("")
    return {
        str(row["ID"]): {
            "choice": str(row["choice"]),
            "confidence": float(row["confidence"]),
            "reason": str(row["reason"]),
            "raw_response": str(row.get("raw_response", "")),
        }
        for _, row in raw.iterrows()
    }


def _append_raw_decisions(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists())


if __name__ == "__main__":
    main()
