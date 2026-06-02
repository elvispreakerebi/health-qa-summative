"""Local evaluation metrics used for experiment comparison."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rouge_score import rouge_scorer


@dataclass(frozen=True)
class RougeScores:
    rouge1_f1: float
    rouge_l_f1: float

    @property
    def weighted_without_llm(self) -> float:
        """Weighted public proxy without unavailable LLM judge signal."""
        return 0.5 * self.rouge1_f1 + 0.5 * self.rouge_l_f1


def score_predictions(references: list[str], predictions: list[str]) -> RougeScores:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    rouge1 = []
    rouge_l = []
    for reference, prediction in zip(references, predictions, strict=True):
        scores = scorer.score(str(reference), str(prediction))
        rouge1.append(scores["rouge1"].fmeasure)
        rouge_l.append(scores["rougeL"].fmeasure)
    if not rouge1:
        return RougeScores(rouge1_f1=0.0, rouge_l_f1=0.0)
    return RougeScores(
        rouge1_f1=float(pd.Series(rouge1).mean()),
        rouge_l_f1=float(pd.Series(rouge_l).mean()),
    )
