from transformers import Trainer, TrainingArguments, PreTrainedModel, TrainerCallback
from transformers.utils import is_datasets_available, is_sagemaker_mp_enabled
from transformers.trainer_utils import seed_worker
from safetensors.torch import load_model, save_model

import datasets
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import torch.nn as nn
import torch
import os

from functools import partial
from dataclasses import dataclass, asdict
from typing import Any, Callable
from collections import defaultdict

class BinaryCrossTrainer(Trainer):
    def __init__(self, model = None, args = None, data_collator = None, train_dataset = None, eval_dataset = None, processing_class = None, model_init = None, compute_loss_func = None, compute_metrics = None, callbacks = None, optimizers = (None, None), optimizer_cls_and_kwargs = None, preprocess_logits_for_metrics = None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, processing_class, model_init, compute_loss_func, compute_metrics, callbacks, optimizers, optimizer_cls_and_kwargs, preprocess_logits_for_metrics)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_ids",
                "completion_ids"
            ]

    def compute_loss(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None
    ):
        mode = 'train' if model.training else "eval"
        
        _not_in_model_keys = {"prompt_ids", "completion_ids", "completion_mask", "labels"}
        
        model_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in _not_in_model_keys
        }


        if self.model_accepts_loss_kwargs:
            if num_items_in_batch is not None:
                model_kwargs['num_items_in_batch'] = num_items_in_batch
                

        model_outputs = model(**model_kwargs)
        logits = model_outputs.logits[:, :-1]
        labels_idx = inputs['input_ids'][:, 1:]
        completion_mask = inputs['completion_mask'][:, 1:].bool()
        labels = inputs['labels']

        log_lklh = torch.log_softmax(logits, dim=-1)
        log_lklh = log_lklh.gather(-1, labels_idx.unsqueeze(-1)).squeeze(-1)
        log_lklh = log_lklh.masked_fill(~completion_mask, 0)
        log_lklh = log_lklh.sum(dim=1) / completion_mask.sum(dim=1)

        loss = - (labels * F.logsigmoid(log_lklh.exp()) + (1 - labels) * F.logsigmoid(-log_lklh.exp()))

        self._metrics[mode]['reward'] = log_lklh.detach().mean().item()

        
        loss = loss.mean()
        return (loss, model_outputs) if return_outputs else loss

    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                logits, labels = None, None
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits, labels = outputs, inputs['input_ids']
        return loss, logits, labels
    

    def log(self, logs, start_time = None):
        mode = 'train' if self.model.training else 'eval'
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        if mode == 'eval':
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
