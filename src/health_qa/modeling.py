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


def build_prompt(
    row: pd.Series,
    schema: DatasetSchema,
    prompt_config: dict[str, Any] | None = None,
) -> str:
    """Format one row as a multilingual QA instruction."""
    prompt_config = prompt_config or {}
    question = str(row[schema.question_col]).strip()
    subset = str(row["subset"]).strip() if "subset" in row else ""
    template = prompt_config.get("template")
    if template:
        return str(template).format(question=question, subset=subset)
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
    train_df = _filter_frame(train_df, config.get("data", {}), split="train", seed=seed)
    val_df = _filter_frame(val_df, config.get("data", {}), split="val", seed=seed)
    test_df = _filter_frame(test_df, config.get("data", {}), split="test", seed=seed)

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

    memory_guard = bool(training_config.get("memory_guard", True))
    if memory_guard:
        import torch

        if torch.cuda.is_available():
            max_source_length = min(max_source_length, int(model_config.get("safe_max_source_length", 192)))
            max_target_length = min(max_target_length, int(model_config.get("safe_max_target_length", 128)))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=bool(model_config.get("tokenizer_use_fast", True)),
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    enable_gradient_checkpointing = bool(
        training_config.get("model_gradient_checkpointing", training_config.get("gradient_checkpointing", memory_guard))
    )
    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if enable_gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
    lora_config = training_config.get("lora")
    if lora_config and bool(lora_config.get("enabled", False)):
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=int(lora_config.get("r", 8)),
                lora_alpha=int(lora_config.get("alpha", 16)),
                lora_dropout=float(lora_config.get("dropout", 0.05)),
                target_modules=list(lora_config.get("target_modules", ["q", "v"])),
            ),
        )
        model.print_trainable_parameters()

    prompt_config = model_config.get("prompt")
    train_dataset = Dataset.from_pandas(_to_text2text_frame(train_df, train_schema, prompt_config))
    val_dataset = Dataset.from_pandas(_to_text2text_frame(val_df, val_schema, prompt_config))

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

    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    training_args = {
        "output_dir": str(output_dir / "trainer"),
        "learning_rate": float(training_config.get("learning_rate", 5e-5)),
        "num_train_epochs": float(training_config.get("epochs", 3)),
        "per_device_train_batch_size": int(training_config.get("batch_size", 4)),
        "per_device_eval_batch_size": int(training_config.get("eval_batch_size", training_config.get("batch_size", 4))),
        "gradient_accumulation_steps": int(training_config.get("gradient_accumulation_steps", 4)),
        "predict_with_generate": True,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 100,
        "seed": int(config.get("seed", 42)),
        "fp16": bool(training_config.get("fp16", True)) and cuda_available,
        "report_to": [],
    }
    if memory_guard:
        training_args["per_device_train_batch_size"] = min(int(training_args["per_device_train_batch_size"]), 1)
        training_args["per_device_eval_batch_size"] = min(int(training_args["per_device_eval_batch_size"]), 1)
        training_args["gradient_accumulation_steps"] = max(int(training_args["gradient_accumulation_steps"]), 16)
    if "eval_accumulation_steps" in training_config:
        training_args["eval_accumulation_steps"] = int(training_config.get("eval_accumulation_steps", 1))
    if "save_total_limit" in training_config:
        training_args["save_total_limit"] = int(training_config.get("save_total_limit", 1))
    # Transformers has used both names across releases.
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        training_args["eval_strategy"] = "epoch"
    else:
        training_args["evaluation_strategy"] = "epoch"
    for optional_arg, value in {
        "gradient_checkpointing": enable_gradient_checkpointing,
        "auto_find_batch_size": bool(training_config.get("auto_find_batch_size", memory_guard)),
        "torch_empty_cache_steps": int(training_config.get("torch_empty_cache_steps", 50)),
        "use_cpu": bool(training_config.get("use_cpu", False)),
        "dataloader_pin_memory": bool(training_config.get("dataloader_pin_memory", False)),
        "load_best_model_at_end": bool(training_config.get("load_best_model_at_end", False)),
        "metric_for_best_model": training_config.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": bool(training_config.get("greater_is_better", False)),
    }.items():
        if optional_arg in signature.parameters:
            training_args[optional_arg] = value

    args = Seq2SeqTrainingArguments(**training_args)

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_tokenized,
        "eval_dataset": val_tokenized,
        "data_collator": DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    }
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)
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

    prompts = [build_prompt(row, schema, model_config.get("prompt")) for _, row in df.iterrows()]
    predictions: list[str] = []
    batch_size = int(inference_config.get("batch_size", 8))
    max_source_length = int(model_config.get("max_source_length", 256))
    max_target_length = int(model_config.get("max_target_length", 256))
    if torch.cuda.is_available() and bool(config.get("training", {}).get("memory_guard", True)):
        batch_size = min(batch_size, int(inference_config.get("safe_batch_size", 2)))
        max_source_length = min(max_source_length, int(model_config.get("safe_max_source_length", 192)))
        max_target_length = min(max_target_length, int(model_config.get("safe_max_target_length", 128)))

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


def _to_text2text_frame(
    df: pd.DataFrame,
    schema: DatasetSchema,
    prompt_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if schema.answer_col is None:
        raise ValueError("Training data requires an answer column")
    return pd.DataFrame(
        {
            "source_text": [
                build_prompt(row, schema, prompt_config)
                for _, row in df.iterrows()
            ],
            "target_text": df[schema.answer_col].fillna("").astype(str),
        }
    )


def _filter_frame(
    df: pd.DataFrame,
    data_config: dict[str, Any],
    *,
    split: str,
    seed: int,
) -> pd.DataFrame:
    output = df.copy()
    subsets = data_config.get(f"{split}_subsets")
    if subsets:
        output = output[output["subset"].isin(subsets)].copy()
    max_rows = data_config.get(f"max_{split}_rows")
    if max_rows:
        output = output.sample(n=min(int(max_rows), len(output)), random_state=seed)
    return output.sort_index().reset_index(drop=True)
