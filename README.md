# Multilingual Health QA Summative

This repository contains the reproducible pipeline for the Zindi
Multilingual Health Question Answering in Low-Resource African Languages
Challenge.

The project is local-first and Colab-reproducible:

- reusable code lives in `src/health_qa`;
- experiment settings live in `configs`;
- local CPU-friendly baselines can generate complete submissions;
- Colab notebooks call the repo code instead of duplicating logic when GPU
  fine-tuning is needed;
- generated submissions are written to `submissions`;
- report assets and leaderboard screenshots are tracked under `reports`.

## Target

The public leaderboard benchmark is `0.36618`. The current first-place score
observed on June 2, 2026 is `0.703047`; our optimization target is to beat that
score while keeping the solution reproducible and compliant with Zindi rules.

## Data

The challenge files are not committed to git. Place the files below in
`data/raw/` when running locally, or mount the shared Google Drive folder in
Colab:

- `Train.csv`
- `Val.csv`
- `Test.csv`

Shared Drive folder:
https://drive.google.com/drive/folders/1PkgdUxwHHEtJRPViKkmHjzQS5foE4u1h?usp=sharing

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m health_qa.cli inspect-data --data-dir data/raw
```

## Local Submission Workflow

Start with the local retrieval baseline. It runs on CPU, scores against
`Val.csv`, and creates a valid Zindi submission from `Test.csv`.

```bash
python -m health_qa.cli retrieve-generate \
  --config configs/local_retrieval.yaml \
  --output-dir outputs/local_retrieval
```

Outputs:

```text
outputs/local_retrieval/metrics.csv
outputs/local_retrieval/validation_predictions.csv
outputs/local_retrieval/submission.csv
```

This is the default debugging and iteration surface. Use it to validate data,
submission format, metrics, and experiment tracking before spending Colab time.

## Colab Workflow

Open `notebooks/health_qa_summative_colab.ipynb` in Colab and run the cells in
order. The notebook clones this repo, installs dependencies, mounts Google
Drive, points the config to the shared dataset folder, and can run either the
local retrieval pipeline or neural fine-tuning.

CPU-friendly reproducibility run:

```bash
python -m health_qa.cli retrieve-generate \
  --config configs/local_retrieval.yaml \
  --output-dir outputs/local_retrieval
```

Optional GPU fine-tuning run:

```bash
python -m health_qa.cli train-generate \
  --config configs/baseline.yaml \
  --output-dir outputs/baseline_mt5
```

The final Zindi file for each run will be saved as:

```text
outputs/<run_name>/submission.csv
```

## Submission Format

Zindi requires exactly four columns:

```text
ID,TargetRLF1,TargetR1F1,TargetLLM
```

For each row, the three target columns must contain the same generated answer.

## Project Layout

```text
configs/          Experiment YAML files
data/             Local raw/processed data, ignored by git
notebooks/        Colab notebooks
reports/          Figures, tables, screenshots, report material
src/health_qa/    Reusable package code
submissions/      Generated submission CSVs, ignored by git
tests/            Lightweight tests for data and formatting logic
```

## Reproducibility Rules

- Set seeds for every experiment.
- Log every meaningful experiment in `reports/experiment_log.csv`.
- Let later experiment configs respond to previous results through
  `health_qa.experiments.suggest_next_config`.
- Keep submission generation deterministic unless a config explicitly says
  otherwise.
- Do not use paid APIs, private datasets, AutoML, or non-open-source tooling.
