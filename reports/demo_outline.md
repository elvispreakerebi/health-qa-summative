# Demo Video Outline

Target length: 7-10 minutes. The demo should show that the project is
reproducible, that the experiments were intentional, and that the final
submission decision is based on evidence rather than only local validation.

## 0:00-0:45 - Project Context

- State the Zindi challenge: multilingual health question answering for
  low-resource African languages.
- State the final public leaderboard result: best public score `0.602576`.
- Explain that later experiments were kept in the report because they taught
  useful lessons, but the first official submission remained the strongest
  public result.

## 0:45-1:45 - Repository and Reproducibility

- Show the GitHub repository and branch.
- Show the main folders:
  - `src/health_qa/` for reusable pipeline code.
  - `configs/` for experiment definitions.
  - `scripts/` for retrieval, reranking, translation, and probing runs.
  - `reports/` for experiment tracking and figures.
  - `submissions/` for Zindi-ready CSV files.
- Show the Colab notebook path and explain that it is for reproducibility, while
  the heavy local experiments were run from the same repo code.

## 1:45-2:45 - Data Understanding

- Show `data/raw/Train.csv`, `Val.csv`, and `Test.csv`.
- Run or show:

```bash
python -m health_qa.cli inspect-data --config configs/baseline.yaml
```

- Explain the schema: `ID`, `input`, `output`, and `subset`.
- Mention the eight subset/language-country groups and the core difficulty:
  multilingual questions, long answers, and scarce target-language examples.

## 2:45-4:15 - Modeling Pipeline

- Show the config-driven pipeline in `src/health_qa`.
- Explain the final family of approaches:
  - retrieval baselines using TF-IDF over training answers,
  - language/subset-aware routing,
  - translation-assisted retrieval for Amharic,
  - neural reranking and dense retrieval probes,
  - LoRA fine-tuning experiments in Colab.
- Emphasize that each experiment produced a validation score before any Zindi
  submission.

## 4:15-6:15 - Experiment Progression

- Show `reports/experiment_log.csv`, `reports/leaderboard_progression.csv`, and
  `reports/figures/experiment_progression.png`.
- Explain the most important experiments:
  - Character TF-IDF improved over the first lexical baseline.
  - Conditional hybrid retrieval gave the best public score.
  - NLLB and broader Akan/English probes looked close locally but were lower on
    public leaderboard.
  - Neural reranking and LLM-gating overfit the validation split and failed to
    generalize publicly.
  - AfriTeVa LoRA generation produced low validation scores, so it was not used
    as the final answer source.

## 6:15-7:30 - Final Result and Submission Choice

- Show the Zindi submissions page screenshot.
- State the final selected file:

```text
submissions/zindi_submission_conditional_hybrid_plus_0_507600.csv
```

- Explain why this is the correct final choice:
  - local validation score: `0.507600`,
  - public score: `0.602576`,
  - later public submissions did not beat it.

## 7:30-9:00 - Critical Reflection

- Discuss the main limitation: validation ranking did not perfectly match the
  public test distribution.
- Explain the tradeoff between extractive/retrieval answers and generative
  fine-tuning.
- Discuss responsible AI risks for health QA:
  - misinformation,
  - translation loss,
  - unequal performance across languages,
  - need for human review before real-world medical use.
- Clearly disclose AI assistance: used for coding support, debugging,
  experiment planning, and report drafting support; implementation decisions and
  final analysis were reviewed by the student.
