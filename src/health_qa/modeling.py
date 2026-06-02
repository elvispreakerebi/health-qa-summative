"""Seq2seq fine-tuning and deterministic generation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import inspect

import pandas as pd

from health_qa.config import data_config_from_mapping, load_yaml
from health_qa.data import DatasetSchema, infer_schema, load_csv
from health_qa.metrics import score_predictions
from health_qa.seed import set_seed
from health_qa.submission import build_submission, save_submission


@dataclass(frozen=True)
class RunArtifacts:
    output_dir: Path
    submission_path: Path
    validation_predictions_path: Path
    metrics_path: Path


def build_prompt(row: pd.Series, schema: DatasetSchema) -> str:
    """Format one row as a multilingual QA instruction."""
    question = str(row[schema.question_col]).strip()
    if schema.language_col:
        language = str(row[schema.language_col]).strip()
        return f"Answer this health question in {language}: {question}"
    return f"Answer this health question: {question}"


def run_training_pipeline(config_path: str | Path, output_dir: str | Path) -> RunArtifacts:
    """Fine-tune a seq2seq model, score validation predictions, and save test submission."""
    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_config = data_config_from_mapping(config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df = load_csv(data_config.train_path)
    val_df = load_csv(data_config.val_path)
    test_df = load_csv(data_config.test_path)

    train_schema = infer_schema(train_df, require_answer=True)
    val_schema = infer_schema(val_df, require_answer=True)
    test_schema = infer_schema(test_df, require_answer=False)

    trainer, tokenizer = _train_model(config, train_df, val_df, train_schema, val_schema, output_path)
    val_predictions = _generate_predictions(config, trainer.model, tokenizer, val_df, val_schema)
    test_predictions = _generate_predictions(config, trainer.model, tokenizer, test_df, test_schema)

    metrics = score_predictions(
        references=val_df[val_schema.answer_col].fillna("").astype(str).tolist(),  # type: ignore[index]
        predictions=val_predictions,
    )
    metrics_df = pd.DataFrame(
        [
            {
                "rouge1_f1": metrics.rouge1_f1,
                "rouge_l_f1": metrics.rouge_l_f1,
                "weighted_without_llm": metrics.weighted_without_llm,
            }
        ]
    )

    validation_predictions_path = output_path / "validation_predictions.csv"
    metrics_path = output_path / "metrics.csv"
    submission_path = output_path / "submission.csv"

    pd.DataFrame(
        {
            "ID": val_df[val_schema.id_col],
            "reference": val_df[val_schema.answer_col],  # type: ignore[index]
            "prediction": val_predictions,
        }
    ).to_csv(validation_predictions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    submission = build_submission(test_df[test_schema.id_col], test_predictions)
    save_submission(submission, submission_path)

    return RunArtifacts(
        output_dir=output_path,
        submission_path=submission_path,
        validation_predictions_path=validation_predictions_path,
        metrics_path=metrics_path,
    )


def _train_model(
    config: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_schema: DatasetSchema,
    val_schema: DatasetSchema,
    output_dir: Path,
):
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    model_name = model_config.get("name", "google/mt5-base")
    max_source_length = int(model_config.get("max_source_length", 256))
    max_target_length = int(model_config.get("max_target_length", 256))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_dataset = Dataset.from_pandas(_to_text2text_frame(train_df, train_schema))
    val_dataset = Dataset.from_pandas(_to_text2text_frame(val_df, val_schema))

    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["source_text"],
            max_length=max_source_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )
        tokenized["labels"] = labels["input_ids"]
        return tokenized

    train_tokenized = train_dataset.map(tokenize_batch, batched=True, remove_columns=train_dataset.column_names)
    val_tokenized = val_dataset.map(tokenize_batch, batched=True, remove_columns=val_dataset.column_names)

    training_args = {
        "output_dir": str(output_dir / "trainer"),
        "learning_rate": float(training_config.get("learning_rate", 5e-5)),
        "num_train_epochs": float(training_config.get("epochs", 3)),
        "per_device_train_batch_size": int(training_config.get("batch_size", 4)),
        "per_device_eval_batch_size": int(training_config.get("batch_size", 4)),
        "gradient_accumulation_steps": int(training_config.get("gradient_accumulation_steps", 4)),
        "predict_with_generate": True,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 100,
        "seed": int(config.get("seed", 42)),
        "fp16": bool(training_config.get("fp16", True)),
        "report_to": [],
    }
    # Transformers has used both names across releases.
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        training_args["eval_strategy"] = "epoch"
    else:
        training_args["evaluation_strategy"] = "epoch"

    args = Seq2SeqTrainingArguments(**training_args)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    return trainer, tokenizer


def _generate_predictions(
    config: dict[str, Any],
    model,
    tokenizer,
    df: pd.DataFrame,
    schema: DatasetSchema,
) -> list[str]:
    import torch
    from tqdm.auto import tqdm

    model_config = config.get("model", {})
    inference_config = config.get("inference", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    prompts = [build_prompt(row, schema) for _, row in df.iterrows()]
    predictions: list[str] = []
    batch_size = int(inference_config.get("batch_size", 8))
    max_source_length = int(model_config.get("max_source_length", 256))
    max_target_length = int(model_config.get("max_target_length", 256))

    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            max_length=max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_target_length,
                num_beams=int(inference_config.get("num_beams", 4)),
                no_repeat_ngram_size=int(inference_config.get("no_repeat_ngram_size", 3)),
                length_penalty=float(inference_config.get("length_penalty", 1.0)),
                early_stopping=bool(inference_config.get("early_stopping", True)),
            )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [prediction.strip() for prediction in predictions]


def _to_text2text_frame(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    if schema.answer_col is None:
        raise ValueError("Training data requires an answer column")
    return pd.DataFrame(
        {
            "source_text": [build_prompt(row, schema) for _, row in df.iterrows()],
            "target_text": df[schema.answer_col].fillna("").astype(str),
        }
    )
