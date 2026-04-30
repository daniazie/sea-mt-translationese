from transformers import PreTrainedTokenizerBase, Trainer, TrainingArguments, PreTrainedModel, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from trl import SFTConfig, SFTTrainer, DPOTrainer, DPOConfig
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from evaluate import load

from typing import Callable
import numpy as np
import gc

def get_training_args_class(trainer: str) -> TrainingArguments:
    if trainer.lower() == 'sft':
        return SFTConfig
    elif trainer.lower() == 'dpo':
        return DPOConfig

def load_trainer(
    trainer: str, 
    model: PreTrainedModel,
    training_args: TrainingArguments,
    train_dataset: Dataset | IterableDataset, 
    eval_dataset: Dataset | IterableDataset | None = None,
    processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
    data_collator: DataCollatorMixin | None = None, 
    compute_metrics: Callable | None = None,
    **kwargs
) -> Trainer:
    if trainer.lower() == 'sft':
        return SFTTrainer(
            model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            **kwargs
        )
    elif trainer.lower() == "dpo":
        return DPOTrainer(
            model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            **kwargs
        )

def get_lora_modules(lora_modules: str) -> str | list[str]:
    if not "," in lora_modules:
        return lora_modules
    else:
        return [module for module in lora_modules.split(',')]


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

metric = load('sacrebleu', tokenize='spBLEU-1K')

def compute_metrics(pred_eval, tokenizer):
    gc.collect()
    preds, labels = pred_eval
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {'spBLEU': result['score']}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)