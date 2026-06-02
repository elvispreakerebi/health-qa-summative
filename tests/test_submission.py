import pandas as pd
import pytest

from health_qa.submission import build_submission, validate_submission


def test_build_submission_repeats_prediction_columns():
    submission = build_submission(pd.Series(["id1", "id2"]), ["answer one", "answer two"])

    assert list(submission.columns) == ["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"]
    assert submission.loc[0, "TargetRLF1"] == "answer one"
    assert submission.loc[0, "TargetRLF1"] == submission.loc[0, "TargetR1F1"]
    assert submission.loc[0, "TargetR1F1"] == submission.loc[0, "TargetLLM"]


def test_validate_submission_rejects_mismatched_targets():
    bad = pd.DataFrame(
        {
            "ID": ["id1"],
            "TargetRLF1": ["a"],
            "TargetR1F1": ["b"],
            "TargetLLM": ["a"],
        }
    )

    with pytest.raises(ValueError, match="must match"):
        validate_submission(bad)
