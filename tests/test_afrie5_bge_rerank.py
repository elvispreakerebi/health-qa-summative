import numpy as np
import pandas as pd

from health_qa.data import DatasetSchema
from scripts.run_afrie5_bge_rerank import (
    _candidate_scores,
    format_document,
    format_query,
)


def test_format_query_adds_instruction_prefix():
    query = format_query("What is malaria?", "Instruct: x\nQuery: ")

    assert query == "Instruct: x\nQuery: What is malaria?"


def test_format_document_includes_question_and_answer():
    row = pd.Series({"input": "What is malaria?", "output": "A mosquito-borne illness.", "subset": "Eng_Uga"})
    schema = DatasetSchema(id_col="ID", question_col="input", answer_col="output")

    document = format_document(row, schema, "subset={subset}\nq={question}\na={answer}")

    assert "subset=Eng_Uga" in document
    assert "q=What is malaria?" in document
    assert "a=A mosquito-borne illness." in document


def test_candidate_scores_unions_dense_and_tfidf_candidates():
    candidate_positions = np.asarray([10, 11, 12, 13])
    dense_scores = np.asarray([0.9, 0.1, 0.2, 0.3])
    tfidf_scores = np.asarray([0.0, 0.8, 0.7, 0.1])

    candidates = _candidate_scores(
        candidate_positions,
        dense_scores,
        tfidf_scores,
        top_k_dense=1,
        top_k_tfidf=2,
        max_candidates=3,
        dense_weight=1.0,
        tfidf_weight=0.5,
    )

    positions = {candidate["candidate_position"] for candidate in candidates}
    assert positions == {10, 11, 12}
    assert candidates[0]["candidate_position"] == 10
