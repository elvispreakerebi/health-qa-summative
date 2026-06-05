"""Create translation-backed candidates for weak-language subsets."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from health_qa.data import infer_schema
from health_qa.metrics import score_predictions
from health_qa.submission import build_submission, save_submission


LANG_CODES = {
    "Aka_Gha": "aka_Latn",
    "Amh_Eth": "amh_Ethi",
}

LANG_NAMES = {
    "Aka_Gha": "Akan/Twi",
    "Amh_Eth": "Amharic",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve English answers and translate them with NLLB")
    parser.add_argument("--train-path", default="data/raw/Train.csv")
    parser.add_argument("--val-path", default="data/raw/Val.csv")
    parser.add_argument("--test-path", default="data/raw/Test.csv")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--translator", choices=["nllb", "ollama"], default="nllb")
    parser.add_argument("--ollama-model", default="aya:8b")
    parser.add_argument("--subsets", nargs="+", default=["Aka_Gha", "Amh_Eth"])
    parser.add_argument("--english-subsets", nargs="+", default=["Eng_Gha", "Eng_Eth", "Eng_Ken", "Eng_Uga"])
    parser.add_argument("--lang-code-overrides", nargs="*", default=[])
    parser.add_argument("--limit-per-subset", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--embedding-cache-dir", default="outputs/cache")
    parser.add_argument("--translate-query", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    for override in args.lang_code_overrides:
        subset, code = override.split("=", maxsplit=1)
        LANG_CODES[subset.strip()] = code.strip()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    test_df = pd.read_csv(args.test_path)
    baseline_dir = Path(args.baseline_dir)

    english_bank = train_df[train_df["subset"].isin(args.english_subsets)].reset_index(drop=True)
    if english_bank.empty:
        raise ValueError("English retrieval bank is empty")

    val_rows = _target_rows(val_df, args.subsets, args.limit_per_subset)
    val_retrieval_rows = _with_translated_queries(
        val_rows,
        args.model_name,
        enabled=args.translate_query,
        batch_size=args.batch_size,
        max_new_tokens=96,
        local_files_only=args.local_files_only,
    )
    val_candidates = _retrieve_english_answers(
        english_bank,
        val_retrieval_rows,
        local_files_only=args.local_files_only,
        cache_dir=Path(args.embedding_cache_dir),
    )
    if args.translate_query:
        val_candidates["translated_question"] = val_retrieval_rows["input"].to_numpy()
    val_candidates["prediction"] = _translate_answers(
        val_candidates["english_answer"].tolist(),
        val_candidates["subset"].tolist(),
        translator=args.translator,
        model_name=args.model_name,
        ollama_model=args.ollama_model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_files_only,
    )
    val_output = _merge_with_baseline(
        val_df,
        baseline_dir / "validation_predictions.csv",
        val_candidates[["ID", "prediction"]],
        submission_mode=False,
    )
    _write_validation(output_dir, val_df, val_output)
    val_candidates.to_csv(output_dir / "translated_validation_candidates.csv", index=False)

    if args.limit_per_subset is None:
        test_rows = _target_rows(test_df, args.subsets, None)
        test_retrieval_rows = _with_translated_queries(
            test_rows,
            args.model_name,
            enabled=args.translate_query,
            batch_size=args.batch_size,
            max_new_tokens=96,
            local_files_only=args.local_files_only,
        )
        test_candidates = _retrieve_english_answers(
            english_bank,
            test_retrieval_rows,
            local_files_only=args.local_files_only,
            cache_dir=Path(args.embedding_cache_dir),
        )
        if args.translate_query:
            test_candidates["translated_question"] = test_retrieval_rows["input"].to_numpy()
        test_candidates["prediction"] = _translate_answers(
            test_candidates["english_answer"].tolist(),
            test_candidates["subset"].tolist(),
            translator=args.translator,
            model_name=args.model_name,
            ollama_model=args.ollama_model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.local_files_only,
        )
        test_output = _merge_with_baseline(
            test_df,
            baseline_dir / "submission.csv",
            test_candidates[["ID", "prediction"]],
            submission_mode=True,
        )
        save_submission(build_submission(test_df["ID"], test_output["prediction"]), output_dir / "submission.csv")
        test_candidates.to_csv(output_dir / "translated_test_candidates.csv", index=False)

    print(f"Metrics: {output_dir / 'metrics.csv'}")
    print(f"Validation predictions: {output_dir / 'validation_predictions.csv'}")


def _target_rows(df: pd.DataFrame, subsets: list[str], limit_per_subset: int | None) -> pd.DataFrame:
    rows = df[df["subset"].isin(subsets)].copy()
    if limit_per_subset is None:
        return rows.reset_index(drop=True)
    return (
        rows.groupby("subset", sort=False, group_keys=False)
        .head(limit_per_subset)
        .reset_index(drop=True)
    )


def _with_translated_queries(
    rows: pd.DataFrame,
    model_name: str,
    *,
    enabled: bool,
    batch_size: int,
    max_new_tokens: int,
    local_files_only: bool,
) -> pd.DataFrame:
    if not enabled:
        return rows
    output = rows.copy()
    output["input"] = _translate_to_english_nllb(
        output["input"].fillna("").astype(str).tolist(),
        output["subset"].astype(str).tolist(),
        model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        local_files_only=local_files_only,
    )
    return output


def _retrieve_english_answers(
    bank_df: pd.DataFrame,
    query_df: pd.DataFrame,
    *,
    local_files_only: bool,
    cache_dir: Path,
) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        local_files_only=local_files_only,
    )
    bank_texts = bank_df["input"].fillna("").astype(str).tolist()
    bank_embeddings = _load_or_encode_bank(model, bank_texts, cache_dir)
    query_embeddings = model.encode(query_df["input"].fillna("").astype(str).tolist(), normalize_embeddings=True)
    scores = cosine_similarity(query_embeddings, bank_embeddings)
    best_positions = scores.argmax(axis=1)
    best_scores = scores.max(axis=1)
    matched = bank_df.iloc[best_positions].reset_index(drop=True)
    return pd.DataFrame(
        {
            "ID": query_df["ID"].to_numpy(),
            "subset": query_df["subset"].to_numpy(),
            "question": query_df["input"].to_numpy(),
            "matched_id": matched["ID"].to_numpy(),
            "similarity": best_scores,
            "english_answer": matched["output"].fillna("").astype(str).to_numpy(),
        }
    )


def _load_or_encode_bank(model, texts: list[str], cache_dir: Path) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
    cache_path = cache_dir / f"english_bank_mpnet_{digest}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    embeddings = model.encode(texts, normalize_embeddings=True)
    np.save(cache_path, embeddings)
    return embeddings


def _translate_answers(
    answers: list[str],
    subsets: list[str],
    *,
    translator: str,
    model_name: str,
    ollama_model: str,
    batch_size: int,
    max_new_tokens: int,
    local_files_only: bool,
) -> list[str]:
    if translator == "ollama":
        return _translate_answers_ollama(answers, subsets, ollama_model)
    return _translate_answers_nllb(
        answers,
        subsets,
        model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        local_files_only=local_files_only,
    )


def _translate_answers_nllb(
    answers: list[str],
    subsets: list[str],
    model_name: str,
    *,
    batch_size: int,
    max_new_tokens: int,
    local_files_only: bool,
) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=local_files_only)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    outputs: list[str] = []
    for subset in dict.fromkeys(subsets):
        target_lang = LANG_CODES.get(subset)
        if target_lang is None:
            raise ValueError(f"No NLLB language code configured for subset {subset}")
        indices = [index for index, item in enumerate(subsets) if item == subset]
        total_batches = (len(indices) + batch_size - 1) // batch_size
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = [answers[index] for index in batch_indices]
            batch_number = start // batch_size + 1
            print(
                f"Translating answers {subset} batch {batch_number}/{total_batches}",
                flush=True,
            )
            tokenizer.src_lang = "eng_Latn"
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend((batch_indices[index], text.strip()) for index, text in enumerate(decoded))
    return [text for _, text in sorted(outputs, key=lambda item: item[0])]


def _translate_to_english_nllb(
    texts: list[str],
    subsets: list[str],
    model_name: str,
    *,
    batch_size: int,
    max_new_tokens: int,
    local_files_only: bool,
) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=local_files_only)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    outputs: list[tuple[int, str]] = []
    for subset in dict.fromkeys(subsets):
        source_lang = LANG_CODES.get(subset)
        if source_lang is None:
            raise ValueError(f"No NLLB language code configured for subset {subset}")
        indices = [index for index, item in enumerate(subsets) if item == subset]
        total_batches = (len(indices) + batch_size - 1) // batch_size
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = [texts[index] for index in batch_indices]
            batch_number = start // batch_size + 1
            print(
                f"Translating queries {subset} batch {batch_number}/{total_batches}",
                flush=True,
            )
            tokenizer.src_lang = source_lang
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend((batch_indices[index], text.strip()) for index, text in enumerate(decoded))
    return [text for _, text in sorted(outputs, key=lambda item: item[0])]


def _translate_answers_ollama(answers: list[str], subsets: list[str], model_name: str) -> list[str]:
    import json
    import urllib.request

    outputs: list[str] = []
    for answer, subset in zip(answers, subsets, strict=True):
        language = LANG_NAMES.get(subset)
        if language is None:
            raise ValueError(f"No language name configured for subset {subset}")
        prompt = (
            f"Translate the following health answer from English into {language}.\n"
            "Return only the translation. Do not explain, summarize, add bullets, or add quotes.\n\n"
            f"English answer:\n{answer.strip()}"
        )
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 220},
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read().decode("utf-8"))
        outputs.append(_clean_ollama_translation(str(data.get("response", ""))))
    return outputs


def _clean_ollama_translation(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'").strip()
    markers = ["Translation:", "Answer:", "Here is the translation:"]
    for marker in markers:
        if cleaned.lower().startswith(marker.lower()):
            cleaned = cleaned[len(marker) :].strip()
    return cleaned


def _merge_with_baseline(
    source_df: pd.DataFrame,
    baseline_path: Path,
    candidates: pd.DataFrame,
    *,
    submission_mode: bool,
) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path)
    if submission_mode:
        baseline = baseline.rename(columns={"TargetRLF1": "prediction"})[["ID", "prediction"]]
    else:
        baseline = baseline[["ID", "prediction"]]
    replacement = candidates.set_index("ID")["prediction"].to_dict()
    baseline["prediction"] = [
        replacement.get(row_id, prediction)
        for row_id, prediction in zip(baseline["ID"], baseline["prediction"], strict=True)
    ]
    expected_ids = source_df["ID"].tolist()
    if baseline["ID"].tolist() != expected_ids:
        baseline = source_df[["ID"]].merge(baseline, on="ID", how="left")
    if baseline["prediction"].isna().any():
        raise ValueError("Merged predictions contain missing values")
    return baseline


def _write_validation(output_dir: Path, val_df: pd.DataFrame, predictions: pd.DataFrame) -> None:
    schema = infer_schema(val_df, require_answer=True)
    refs = val_df[schema.answer_col].fillna("").astype(str).tolist()  # type: ignore[index]
    pred = predictions["prediction"].fillna("").astype(str).tolist()
    metrics = score_predictions(refs, pred)
    predictions = predictions.copy()
    predictions["reference"] = val_df[schema.answer_col].to_numpy()  # type: ignore[index]
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
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
