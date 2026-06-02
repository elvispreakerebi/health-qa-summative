"""Configuration loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    train_file: str = "Train.csv"
    val_file: str = "Val.csv"
    test_file: str = "Test.csv"

    @property
    def train_path(self) -> Path:
        return self.raw_dir / self.train_file

    @property
    def val_path(self) -> Path:
        return self.raw_dir / self.val_file

    @property
    def test_path(self) -> Path:
        return self.raw_dir / self.test_file


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return a plain dictionary."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping config in {path}")
    return data


def data_config_from_mapping(config: dict[str, Any]) -> DataConfig:
    data = config.get("data", {})
    return DataConfig(
        raw_dir=Path(data.get("raw_dir", "data/raw")),
        train_file=data.get("train_file", "Train.csv"),
        val_file=data.get("val_file", "Val.csv"),
        test_file=data.get("test_file", "Test.csv"),
    )
