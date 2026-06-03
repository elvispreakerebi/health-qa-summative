import pandas as pd

from health_qa.data import infer_schema
from health_qa.modeling import _filter_frame, build_prompt


def test_build_prompt_accepts_template_with_subset():
    row = pd.Series({"ID": "1", "input": "What is contraception?", "output": "Answer.", "subset": "Eng_Gha"})
    schema = infer_schema(pd.DataFrame([row]), require_answer=True)

    prompt = build_prompt(
        row,
        schema,
        {"template": "subset: {subset}\nquestion: {question}\nanswer:"},
    )

    assert prompt == "subset: Eng_Gha\nquestion: What is contraception?\nanswer:"


def test_filter_frame_can_select_subset_and_sample_rows():
    df = pd.DataFrame(
        {
            "ID": ["1", "2", "3"],
            "input": ["a", "b", "c"],
            "output": ["x", "y", "z"],
            "subset": ["A", "B", "B"],
        }
    )

    filtered = _filter_frame(
        df,
        {"train_subsets": ["B"], "max_train_rows": 1},
        split="train",
        seed=42,
    )

    assert len(filtered) == 1
    assert filtered.loc[0, "subset"] == "B"
