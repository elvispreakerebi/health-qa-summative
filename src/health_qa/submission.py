"""Zindi submission file helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUBMISSION_COLUMNS = ["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"]


def build_submission(ids: pd.Series, predictions: list[str] | pd.Series) -> pd.DataFrame:
    """Build the exact four-column multi-metric submission required by Zindi."""
    pred_series = pd.Series(predictions, dtype="string").fillna("")
    if len(ids) != len(pred_series):
        raise ValueError("ids and predictions must have the same length")
    return pd.DataFrame(
        {
            "ID": ids.to_numpy(),
            "TargetRLF1": pred_series.to_numpy(),
            "TargetR1F1": pred_series.to_numpy(),
            "TargetLLM": pred_series.to_numpy(),
        },
        columns=SUBMISSION_COLUMNS,
    )


def validate_submission(df: pd.DataFrame) -> None:
    """Fail fast if a submission violates the challenge format."""
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Submission columns must be {SUBMISSION_COLUMNS}")
    if df.empty:
        raise ValueError("Submission is empty")
    equal_targets = (df["TargetRLF1"] == df["TargetR1F1"]) & (
        df["TargetR1F1"] == df["TargetLLM"]
    )
    if not bool(equal_targets.all()):
        raise ValueError("TargetRLF1, TargetR1F1, and TargetLLM must match row-by-row")


def save_submission(df: pd.DataFrame, path: str | Path) -> Path:
    validate_submission(df)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
