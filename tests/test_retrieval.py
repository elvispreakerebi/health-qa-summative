from pathlib import Path

import pandas as pd

from health_qa.retrieval import run_retrieval_pipeline
from health_qa.submission import SUBMISSION_COLUMNS


def test_retrieval_pipeline_writes_metrics_and_submission(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "ID": ["tr1", "tr2"],
            "input": ["What treats malaria?", "How do I prevent dehydration?"],
            "output": ["Use antimalarial medicine from a health worker.", "Drink oral rehydration solution."],
            "subset": ["train", "train"],
        }
    ).to_csv(data_dir / "Train.csv", index=False)
    pd.DataFrame(
        {
            "ID": ["va1"],
            "input": ["What medicine treats malaria?"],
            "output": ["Use antimalarial medicine from a health worker."],
            "subset": ["val"],
        }
    ).to_csv(data_dir / "Val.csv", index=False)
    pd.DataFrame(
        {
            "ID": ["te1"],
            "input": ["How can dehydration be prevented?"],
            "subset": ["test"],
        }
    ).to_csv(data_dir / "Test.csv", index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  raw_dir: {data_dir}
retrieval:
  analyzer: char_wb
  ngram_min: 3
  ngram_max: 5
  max_features: 1000
  batch_size: 2
  include_val_for_test: true
""",
        encoding="utf-8",
    )

    artifacts = run_retrieval_pipeline(config_path, tmp_path / "outputs")

    metrics = pd.read_csv(artifacts.metrics_path)
    submission = pd.read_csv(artifacts.submission_path)
    validation = pd.read_csv(artifacts.validation_predictions_path)

    assert set(metrics.columns) == {"rouge1_f1", "rouge_l_f1", "weighted_without_llm"}
    assert list(submission.columns) == SUBMISSION_COLUMNS
    assert len(submission) == 1
    assert len(validation) == 1
    assert validation.loc[0, "matched_id"] == "tr1"


def test_retrieval_pipeline_can_restrict_matches_to_subset(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "ID": ["tr_aka", "tr_swa"],
            "input": ["same health question", "same health question with local wording"],
            "output": ["Akan answer", "Swahili answer"],
            "subset": ["Aka_Gha", "Swa_Ken"],
        }
    ).to_csv(data_dir / "Train.csv", index=False)
    pd.DataFrame(
        {
            "ID": ["va1"],
            "input": ["same health question"],
            "output": ["Swahili answer"],
            "subset": ["Swa_Ken"],
        }
    ).to_csv(data_dir / "Val.csv", index=False)
    pd.DataFrame(
        {
            "ID": ["te1"],
            "input": ["same health question"],
            "subset": ["Swa_Ken"],
        }
    ).to_csv(data_dir / "Test.csv", index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  raw_dir: {data_dir}
retrieval:
  analyzer: char_wb
  ngram_min: 3
  ngram_max: 5
  max_features: 1000
  batch_size: 2
  include_val_for_test: true
  group_col: subset
""",
        encoding="utf-8",
    )

    artifacts = run_retrieval_pipeline(config_path, tmp_path / "outputs")
    validation = pd.read_csv(artifacts.validation_predictions_path)

    assert validation.loc[0, "matched_id"] == "tr_swa"


def test_retrieval_pipeline_accepts_ensemble_config(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = {
        "ID": ["tr1", "tr2"],
        "input": ["malaria medicine", "dehydration prevention"],
        "output": ["Use antimalarial medicine.", "Drink oral rehydration solution."],
        "subset": ["train", "train"],
    }
    pd.DataFrame(rows).to_csv(data_dir / "Train.csv", index=False)
    pd.DataFrame(
        {
            "ID": ["va1"],
            "input": ["malaria treatment"],
            "output": ["Use antimalarial medicine."],
            "subset": ["val"],
        }
    ).to_csv(data_dir / "Val.csv", index=False)
    pd.DataFrame({"ID": ["te1"], "input": ["prevent dehydration"], "subset": ["test"]}).to_csv(
        data_dir / "Test.csv",
        index=False,
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  raw_dir: {data_dir}
retrieval_ensemble:
  members:
    - analyzer: char
      ngram_min: 2
      ngram_max: 4
      max_features: 1000
      min_df: 1
      sublinear_tf: true
      batch_size: 2
      weight: 1.0
    - analyzer: char_wb
      ngram_min: 3
      ngram_max: 5
      max_features: 1000
      min_df: 1
      sublinear_tf: false
      batch_size: 2
      weight: 0.5
""",
        encoding="utf-8",
    )

    artifacts = run_retrieval_pipeline(config_path, tmp_path / "outputs")
    submission = pd.read_csv(artifacts.submission_path)

    assert list(submission.columns) == SUBMISSION_COLUMNS
    assert len(submission) == 1


def test_retrieval_pipeline_accepts_extended_tfidf_options(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "ID": ["tr1", "tr2"],
            "input": ["malaria fever medicine", "clean water dehydration"],
            "output": ["Use malaria medicine.", "Drink safe water."],
            "subset": ["train", "train"],
        }
    ).to_csv(data_dir / "Train.csv", index=False)
    pd.DataFrame(
        {
            "ID": ["va1"],
            "input": ["malaria medicine"],
            "output": ["Use malaria medicine."],
            "subset": ["val"],
        }
    ).to_csv(data_dir / "Val.csv", index=False)
    pd.DataFrame({"ID": ["te1"], "input": ["safe water"], "subset": ["test"]}).to_csv(
        data_dir / "Test.csv",
        index=False,
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  raw_dir: {data_dir}
retrieval:
  analyzer: char
  ngram_min: 2
  ngram_max: 4
  max_features: 1000
  min_df: 1
  sublinear_tf: false
  binary: true
  use_idf: false
  smooth_idf: false
  norm: l1
  batch_size: 2
""",
        encoding="utf-8",
    )

    artifacts = run_retrieval_pipeline(config_path, tmp_path / "outputs")
    metrics = pd.read_csv(artifacts.metrics_path)

    assert metrics.loc[0, "weighted_without_llm"] >= 0
