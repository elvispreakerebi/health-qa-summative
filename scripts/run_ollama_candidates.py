"""Generate validation/test candidates with a local Ollama model."""

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
    parser = argparse.ArgumentParser(description="Run local Ollama generation with fallback predictions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir", required=True, help="Directory with validation_predictions.csv and submission.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subsets", required=True, help="Comma-separated subset names to generate")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--limit-per-subset", type=int, default=None)
    parser.add_argument("--mode", choices=["answer", "rewrite"], default="answer")
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_config = data_config_from_mapping(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subsets = {value.strip() for value in args.subsets.split(",") if value.strip()}

    val_df = load_csv(data_config.val_path)
    test_df = load_csv(data_config.test_path)
    val_schema = infer_schema(val_df, require_answer=True)
    test_schema = infer_schema(test_df, require_answer=False)
    base_dir = Path(args.base_dir)

    val_predictions = _blend_split(
        val_df,
        val_schema.id_col,
        val_schema.question_col,
        base_dir / "validation_predictions.csv",
        output_dir / "ollama_validation_raw.csv",
        args.model,
        subsets,
        args.temperature,
        args.max_tokens,
        args.mode,
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
            }
        ]
    ).to_csv(output_dir / "metrics.csv", index=False)

    test_predictions = _blend_split(
        test_df,
        test_schema.id_col,
        test_schema.question_col,
        base_dir / "submission.csv",
        output_dir / "ollama_test_raw.csv",
        args.model,
        subsets,
        args.temperature,
        args.max_tokens,
        args.mode,
        submission_mode=True,
        limit_per_subset=args.limit_per_subset,
    )
    submission = build_submission(test_df[test_schema.id_col], test_predictions["prediction"].tolist())
    save_submission(submission, output_dir / "submission.csv")
    print(f"Metrics: {output_dir / 'metrics.csv'}")
    print(f"Submission: {output_dir / 'submission.csv'}")


def _blend_split(
    source_df: pd.DataFrame,
    id_col: str,
    question_col: str,
    base_prediction_path: Path,
    raw_generation_path: Path,
    model: str,
    subsets: set[str],
    temperature: float,
    max_tokens: int,
    mode: str,
    *,
    submission_mode: bool = False,
    limit_per_subset: int | None = None,
) -> pd.DataFrame:
    base = pd.read_csv(base_prediction_path)
    if submission_mode:
        base = base.rename(columns={"TargetRLF1": "prediction"})[["ID", "prediction"]]
    else:
        base = base[["ID", "prediction"]]
    merged = source_df[[id_col, question_col, "subset"]].merge(base, left_on=id_col, right_on="ID", how="left")
    generated = _load_raw_generations(raw_generation_path)

    selected_counts = {subset: 0 for subset in subsets}
    outputs: list[str] = []
    raw_rows: list[dict[str, str]] = []
    for _, row in merged.iterrows():
        row_id = str(row[id_col])
        subset = str(row["subset"])
        fallback = str(row["prediction"]).strip()
        if subset not in subsets:
            outputs.append(fallback)
            continue
        if limit_per_subset is not None and selected_counts[subset] >= limit_per_subset:
            outputs.append(fallback)
            continue
        selected_counts[subset] += 1
        if row_id not in generated:
            generated[row_id] = _generate_answer(
                model,
                str(row[question_col]),
                fallback,
                temperature=temperature,
                max_tokens=max_tokens,
                mode=mode,
            )
            raw_rows.append({"ID": row_id, "prediction": generated[row_id]})
            _append_raw_generation(raw_generation_path, raw_rows)
            raw_rows.clear()
            time.sleep(0.05)
        outputs.append(generated[row_id].strip() or fallback)
    return pd.DataFrame({"ID": merged[id_col], "prediction": outputs})


def _generate_answer(
    model: str,
    question: str,
    draft_answer: str,
    *,
    temperature: float,
    max_tokens: int,
    mode: str,
) -> str:
    prompt = _build_prompt(question, draft_answer, mode)
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
        return _clean_generation(str(json.loads(response.read().decode("utf-8")).get("response", "")))


def _build_prompt(question: str, draft_answer: str, mode: str) -> str:
    if mode == "answer":
        return (
            "Answer the health question in the same language as the question. "
            "Be concise, medically relevant, and return only the answer.\n\n"
            f"Question: {question}\nAnswer:"
        )
    if mode == "rewrite":
        return (
            "Rewrite the draft answer so it directly answers the health question. "
            "Keep the same language as the question. Preserve useful medical details, "
            "remove repetition, and return only the final answer with no preface.\n\n"
            f"Question: {question}\nDraft answer: {draft_answer}\nFinal answer:"
        )
    raise ValueError(f"Unsupported generation mode: {mode}")


def _clean_generation(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = re.sub(r"^Here is[^\n]*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Here's[^\n]*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Final answer:\s*", "", cleaned, flags=re.IGNORECASE)
    parts = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    if len(parts) > 1 and re.search(r"rewritten|answer|candidate", parts[0], flags=re.IGNORECASE):
        cleaned = "\n\n".join(parts[1:])
    cleaned = re.sub(r"\n*I removed\b.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _load_raw_generations(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = pd.read_csv(path)
    return dict(zip(raw["ID"].astype(str), raw["prediction"].fillna("").astype(str), strict=False))


def _append_raw_generation(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=not path.exists())


if __name__ == "__main__":
    main()
