"""Data loading, schema inference, and EDA summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

QUESTION_CANDIDATES = ("Question", "question", "Input", "input", "Source", "source")
ANSWER_CANDIDATES = ("Answer", "answer", "Target", "target", "Response", "response")
ID_CANDIDATES = ("ID", "Id", "id")
LANGUAGE_CANDIDATES = ("Language", "language", "lang", "Lang")


@dataclass(frozen=True)
class DatasetSchema:
    id_col: str
    question_col: str
    answer_col: str | None = None
    language_col: str | None = None


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a challenge CSV with strict path checking."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing data file: {csv_path}")
    return pd.read_csv(csv_path)


def infer_schema(df: pd.DataFrame, require_answer: bool) -> DatasetSchema:
    """Infer common Zindi column names without hard-coding one file version."""
    columns = set(df.columns)
    id_col = _first_present(columns, ID_CANDIDATES, required=True, label="ID")
    question_col = _first_present(columns, QUESTION_CANDIDATES, required=True, label="question")
    answer_col = _first_present(columns, ANSWER_CANDIDATES, required=require_answer, label="answer")
    language_col = _first_present(columns, LANGUAGE_CANDIDATES, required=False, label="language")
    return DatasetSchema(
        id_col=id_col,
        question_col=question_col,
        answer_col=answer_col,
        language_col=language_col,
    )


def summarize_frame(df: pd.DataFrame, schema: DatasetSchema) -> dict[str, object]:
    """Return compact EDA facts for logging and report tables."""
    question_lengths = df[schema.question_col].fillna("").astype(str).str.split().str.len()
    summary: dict[str, object] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "question_words_mean": float(question_lengths.mean()) if len(df) else 0.0,
        "question_words_p95": float(question_lengths.quantile(0.95)) if len(df) else 0.0,
    }
    if schema.answer_col:
        answer_lengths = df[schema.answer_col].fillna("").astype(str).str.split().str.len()
        summary["answer_words_mean"] = float(answer_lengths.mean()) if len(df) else 0.0
        summary["answer_words_p95"] = float(answer_lengths.quantile(0.95)) if len(df) else 0.0
    if schema.language_col:
        summary["language_counts"] = df[schema.language_col].fillna("UNKNOWN").value_counts().to_dict()
    return summary


def _first_present(
    columns: set[str],
    candidates: tuple[str, ...],
    *,
    required: bool,
    label: str,
) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise ValueError(f"Could not infer {label} column from columns: {sorted(columns)}")
    return None
