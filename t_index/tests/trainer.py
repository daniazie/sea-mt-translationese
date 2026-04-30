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

from ensemble import TranslationeseEval


@dataclass
class TEvalTrainingArguments(TrainingArguments):
    remove_unused_columns: bool = False
    loss_type: str = 'llr_paired'
    separate_forwards: bool = False
    per_sample_loss: bool = False
    per_model_loss: bool = True

class TEvalTrainer(Trainer):
    def __init__(self, model = None, args = None, data_collator = None, train_dataset = None, eval_dataset = None, processing_class = None, model_init = None, compute_loss_func = None, compute_metrics = None, callbacks = None, optimizers = (None, None), optimizer_cls_and_kwargs = None, preprocess_logits_for_metrics = None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, processing_class, model_init, compute_loss_func, compute_metrics, callbacks, optimizers, optimizer_cls_and_kwargs, preprocess_logits_for_metrics)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_ids",
                "ht_ids",
                "lt_ids"
            ]

    def get_log_ratios(self, model_outputs, input_ids: torch.LongTensor, completion_mask: torch.LongTensor, ht_ids: torch.Tensor = None, lt_ids: torch.Tensor = None):
        shift_logits = model_outputs.logits[:, :-1]
        shift_labels = input_ids[:, 1:]
        shift_completion_mask = completion_mask[:, 1:].bool()

        log_lklh = shift_logits.log_softmax(dim=-1)
        log_lklh = log_lklh.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        log_lklh = log_lklh.masked_fill(~shift_completion_mask, 0)
        log_lklh = log_lklh.sum(dim=1) / shift_completion_mask.sum(dim=1)

        log_lklh_ht = log_lklh[:ht_ids.shape[0]]
        log_lklh_lt = log_lklh[ht_ids.shape[0]:]
        return log_lklh_ht, log_lklh_lt

    def compute_loss(
        self,
        model: TranslationeseEval | PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None
    ):
        mode = 'train' if model.training else "eval"
        
        if model.training:
            model.ht_model.train()
            model.lt_model.train()
        
        if self.args.separate_forwards:
            _not_in_model_keys = {"completion_mask", "ht_ids", "lt_ids"}
            model_kwargs = {
                k: v
                for k, v in inputs.items()
                if k not in _not_in_model_keys
            }

            _not_in_calc_llr_keys = {"attention_mask"}
            _calc_llr_inputs = {
                k: v
                for k, v in inputs.items()
                if k not in _not_in_calc_llr_keys
            }

            if self.model_accepts_loss_kwargs:
                if num_items_in_batch is not None:
                    model_kwargs['num_items_in_batch'] = num_items_in_batch

            model_kwargs['use_cache'] = False
            model_kwargs['training_mode'] = True
            ht_outputs = model.ht_model(**model_kwargs)
            log_lklhs_ht = self.get_log_ratios(ht_outputs, **_calc_llr_inputs)
            lt_outputs = model.lt_model(**model_kwargs)
            log_lklhs_lt = self.get_log_ratios(lt_outputs, **_calc_llr_inputs)
        else:
            model_kwargs = {
                k: v
                for k, v in inputs.items()
            }
            if self.model_accepts_loss_kwargs:
                if num_items_in_batch is not None:
                    model_kwargs['num_items_in_batch'] = num_items_in_batch
            log_lklhs_ht, log_lklhs_lt = model(training_mode=True, **model_kwargs)
        htm_log_lklh_ht, htm_log_lklh_lt = log_lklhs_ht
        ltm_log_lklh_ht, ltm_log_lklh_lt = log_lklhs_lt

        
        if self.args.per_sample_loss:
            llr_ht = htm_log_lklh_ht - ltm_log_lklh_ht
            llr_lt = htm_log_lklh_lt - ltm_log_lklh_lt

        elif self.args.per_model_loss:
            llr_ht = htm_log_lklh_ht - htm_log_lklh_lt
            llr_lt = ltm_log_lklh_ht - ltm_log_lklh_lt

        ht_rewards = llr_ht.mean().detach().item()
        lt_rewards = llr_lt.mean().detach().item()
        
        if self.args.loss_type == 'llr_paired':
            loss = - 2 * (llr_ht - llr_lt)
        elif self.args.loss_type == 'sigmoid':
            loss = -F.logsigmoid(2 * (llr_ht - llr_lt))
        elif self.args.loss_type == 'bco_pair':
            loss = -F.logsigmoid(2 * llr_ht) - F.logsigmoid(-2 * llr_lt)
        
        loss = loss.mean(dim=0)

        self._metrics[mode]['ht_rewards'].append(2 * ht_rewards)
        self._metrics[mode]['lt_rewards'].append(-2 * lt_rewards)

        return (loss, (llr_ht - llr_lt)) if return_outputs else loss
    
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

    def _save(self, output_dir = None, state_dict = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.ht_model.save_pretrained(f"{output_dir}/ht_model", state_dict=state_dict)
        self.model.lt_model.save_pretrained(f"{output_dir}/lt_model", state_dict=state_dict)

        return super()._save(output_dir, state_dict)