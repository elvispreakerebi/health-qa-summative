# Final Report Outline

File name target: `StudentName_FinalProject.pdf`

## 1. Project Overview

- Challenge goal: multilingual health QA for low-resource African languages.
- Supported languages and health domain context.
- Leaderboard target: beat public rank 1 score observed at `0.703047`.

## 2. Dataset Understanding and EDA

- Train/validation/test sizes.
- Column schema.
- Language/country distribution.
- Question and answer length distributions.
- Missing values, duplicates, and unusual text cases.

## 3. Preprocessing

- Text normalization decisions.
- Language-aware prompting format.
- Token length caps and truncation tradeoffs.
- Any duplicate or noisy-row handling.

## 4. Methodology

- Model family and reason for choosing it.
- Fine-tuning setup.
- Inference settings.
- Adaptive experiment strategy.
- Reproducibility setup and seed control.

## 5. Experiments

Include at least 10 meaningful experiments. For each:

- What changed.
- Why it changed.
- Local validation result.
- Public leaderboard score if submitted.
- Insight learned.

## 6. Results

- Experiment comparison table.
- Leaderboard progression table.
- Training/validation curves where available.
- Per-language performance where available.

## 7. Discussion and Critical Analysis

- Why strongest runs performed better.
- Tradeoffs between ROUGE overlap and semantic answer quality.
- Weak languages or failure modes.
- Public/private leaderboard overfitting risk.

## 8. Ethics and Responsible AI

- Health misinformation risks.
- Bias and language coverage risks.
- Limits of LLM-as-judge and lexical metrics.
- Responsible deployment considerations.
- How AI tools were used in this assignment.

## 9. Limitations and Future Work

- Compute constraints.
- Dataset constraints.
- Model limitations.
- Next experiments after the deadline.

## 10. References

Use APA or IEEE consistently.
