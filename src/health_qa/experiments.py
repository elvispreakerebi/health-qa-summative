"""Adaptive experiment tracking and next-run suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    config: str
    local_score: float | None
    public_score: float | None
    notes: str = ""


def load_experiment_log(path: str | Path) -> pd.DataFrame:
    log_path = Path(path)
    if not log_path.exists():
        return pd.DataFrame()
    return pd.read_csv(log_path)


def suggest_next_config(base_config: dict[str, Any], history: pd.DataFrame) -> dict[str, Any]:
    """Create a conservative next config from previous experiment outcomes.

    The planner keeps changes small so we can attribute improvements clearly in
    the report and avoid burning submissions on uncontrolled sweeps.
    """
    config = _deep_copy(base_config)
    if history.empty or "local_score" not in history:
        return config

    scored = history.dropna(subset=["local_score"]).copy()
    if scored.empty:
        return config

    scored["local_score"] = scored["local_score"].astype(float)
    best = scored.sort_values("local_score", ascending=False).iloc[0]
    recent = scored.tail(3)
    plateaued = len(recent) >= 3 and recent["local_score"].max() - recent["local_score"].min() < 0.005

    inference = config.setdefault("inference", {})
    training = config.setdefault("training", {})
    model = config.setdefault("model", {})

    if plateaued:
        inference["num_beams"] = min(int(inference.get("num_beams", 4)) + 1, 8)
        inference["length_penalty"] = round(float(inference.get("length_penalty", 1.0)) + 0.05, 2)
    elif float(best["local_score"]) > 0.55:
        training["learning_rate"] = max(float(training.get("learning_rate", 5e-5)) * 0.7, 1e-5)
    else:
        training["epochs"] = min(int(training.get("epochs", 3)) + 1, 6)

    if "max_target_length" in model:
        model["max_target_length"] = int(model["max_target_length"])
    return config


def _deep_copy(value: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            copied[key] = _deep_copy(item)
        else:
            copied[key] = item
    return copied
