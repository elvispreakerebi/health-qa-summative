import pandas as pd

from health_qa.data import infer_schema, summarize_frame


def test_infer_schema_with_common_columns():
    df = pd.DataFrame(
        {
            "ID": ["1"],
            "Question": ["What is malaria?"],
            "Answer": ["A mosquito-borne disease."],
            "Language": ["English"],
        }
    )

    schema = infer_schema(df, require_answer=True)

    assert schema.id_col == "ID"
    assert schema.question_col == "Question"
    assert schema.answer_col == "Answer"
    assert schema.language_col == "Language"


def test_summarize_frame_returns_core_counts():
    df = pd.DataFrame(
        {
            "ID": ["1", "2"],
            "Question": ["What is malaria?", "How can I prevent it?"],
            "Answer": ["Disease.", "Use nets."],
        }
    )
    schema = infer_schema(df, require_answer=True)

    summary = summarize_frame(df, schema)

    assert summary["rows"] == 2
    assert summary["missing_cells"] == 0
    assert summary["duplicate_rows"] == 0
