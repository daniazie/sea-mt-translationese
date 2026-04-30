from transformers import Trainer, TrainingArguments, PreTrainedModel, TrainerCallback, PreTrainedTokenizerBase, ProcessorMixin
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

from trl.data_utils import is_conversational

from functools import partial
from dataclasses import dataclass, asdict
from typing import Any, Callable, Literal
from collections import defaultdict

from ensemble_model import TranslationeseEval

@dataclass
class TEvalTrainingArguments(TrainingArguments):
    loss_type: Literal["llr_pair", "llr_pair_sigmoid", "sigmoid", "bco_pair"] = 'llr_pair'
    separate_forwards: bool = False
    per_sample_loss: bool = False
    per_model_loss: bool = True
    beta1: float = 1.0
    beta2: float = 1.0

class TEvalTrainer(Trainer):
    def __init__(self, model = None, args = None, data_collator = None, train_dataset = None, eval_dataset = None, processing_class = None, model_init = None, compute_loss_func = None, compute_metrics = None, callbacks = None, optimizers = (None, None), optimizer_cls_and_kwargs = None, preprocess_logits_for_metrics = None):
        train_dataset = self._prepare_dataset(train_dataset, processing_class=processing_class)
        eval_dataset = self._prepare_dataset(eval_dataset, processing_class=processing_class)

        print(train_dataset[0])
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.beta = self.beta1 + self.beta2
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, processing_class, model_init, compute_loss_func, compute_metrics, callbacks, optimizers, optimizer_cls_and_kwargs, preprocess_logits_for_metrics)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_ids",
                "high_translationese_ids",
                "low_translationese_ids",
            ]

    def tokenize(self, text: str | list, tokenizer: PreTrainedTokenizerBase, **kwargs):
        if isinstance(text, list):
            return tokenizer.apply_chat_template(
                text, tokenize=True, return_dict=True, **kwargs
            )
        return tokenizer(text=text)

    def _prepare_dataset(self, dataset: datasets.Dataset | datasets.IterableDataset, processing_class: PreTrainedTokenizerBase | ProcessorMixin):        
        def _tokenize_fn(example, tokenizer):
            output = {}
            if is_conversational(example):
                prompt_ids = self.tokenize(example['prompt'], tokenizer=tokenizer, add_generation_prompt=True)['input_ids']
            else:
                prompt_ids = self.tokenize(example['prompt'], tokenizer=tokenizer)
                if not example['high_translationese'].endswith(tokenizer.eos_token):
                    example['high_translationese'] += tokenizer.eos_token
                if not example['low_translationese'].endswith(tokenizer.eos_token):
                    example['low_translationese'] += tokenizer.eos_token

            prompt_high_translationese_ids = self.tokenize(example['prompt'] + example['high_translationese'], tokenizer=tokenizer)['input_ids']
            prompt_low_translationese_ids = self.tokenize(example['prompt'] + example['low_translationese'], tokenizer=tokenizer)['input_ids']

            if not prompt_high_translationese_ids[:len(prompt_ids)] == prompt_ids:
                print("Mismatch between tokenized prompt and tokenized prompt + ht")
            if not prompt_low_translationese_ids[:len(prompt_ids)] == prompt_ids:
                print("Mismatch between tokenized prompt and tokenized prompt + lt")
            
            output["prompt_ids"] = prompt_ids
            output["high_translationese_ids"] = prompt_high_translationese_ids[len(prompt_ids):]
            output["low_translationese_ids"] = prompt_low_translationese_ids[len(prompt_ids):]

            return output
        
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        else:
            tokenizer = processing_class

        return dataset.map(_tokenize_fn, fn_kwargs={"tokenizer": tokenizer})

    def get_log_ratios(self, model_outputs, input_ids: torch.LongTensor, completion_mask: torch.LongTensor, high_translationese_ids: torch.Tensor = None, low_translationese_ids: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        shift_logits: torch.Tensor = model_outputs.logits[:, :-1]
        shift_labels: torch.Tensor = input_ids[:, 1:]
        shift_completion_mask: torch.Tensor = completion_mask[:, 1:].bool()

        probs = torch.softmax(shift_logits, dim=-1)
        ht_probs, lt_probs = probs.chunk(2, dim=0)

        log_lklh = shift_logits.log_softmax(dim=-1)
        log_lklh = log_lklh.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        log_lklh = log_lklh.masked_fill(~shift_completion_mask, 0)
        log_lklh = log_lklh.sum(dim=1) / shift_completion_mask.sum(dim=1)

        log_lklh_ht = log_lklh[:high_translationese_ids.shape[0]]
        log_lklh_lt = log_lklh[high_translationese_ids.shape[0]:]
        return log_lklh_ht, log_lklh_lt, ht_probs, lt_probs

    def compute_loss(
        self,
        model: TranslationeseEval | PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None
    ):
        mode = 'train' if model.training else "eval"
        
        if model.training and not model.high_translationese_model.training and not model.low_translationese_model.training:
            model.high_translationese_model.train()
            model.low_translationese_model.train()
        
        if self.args.separate_forwards:
            _not_in_model_keys = {"completion_mask", "high_translationese_ids", "low_translationese_ids"}
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
            high_translationese_outputs = model.high_translationese_model(**model_kwargs)
            log_lklhs_ht = self.get_log_ratios(high_translationese_outputs, **_calc_llr_inputs)
            low_translationese_outputs = model.low_translationese_model(**model_kwargs)
            log_lklhs_lt = self.get_log_ratios(low_translationese_outputs, **_calc_llr_inputs)
        else:
            model_kwargs = {
                k: v
                for k, v in inputs.items()
            }
            if self.model_accepts_loss_kwargs:
                if num_items_in_batch is not None:
                    model_kwargs['num_items_in_batch'] = num_items_in_batch
            log_lklhs_ht, log_lklhs_lt = model(training_mode=True, **model_kwargs)

        htm_log_lklh_ht, htm_log_lklh_lt, htm_probs_ht, htm_probs_lt = log_lklhs_ht
        ltm_log_lklh_ht, ltm_log_lklh_lt, ltm_probs_ht, ltm_probs_lt = log_lklhs_lt

        llr_ht = htm_log_lklh_ht - htm_log_lklh_lt
        llr_lt = ltm_log_lklh_ht - ltm_log_lklh_lt



        
        if self.args.loss_type == 'llr_pair':
            loss = - self.beta * (llr_ht - llr_lt)
        elif self.args.loss_type == 'sigmoid':
            loss = -F.logsigmoid(- self.beta * (llr_ht - llr_lt))
        elif self.args.loss_type == 'llr_pair_sigmoid':
            loss = - (F.logsigmoid(- self.beta * llr_ht) - F.logsigmoid(self.beta * llr_lt))
        elif self.args.loss_type == 'bco_pair':
            loss = -F.logsigmoid(self.beta * llr_ht) - F.logsigmoid(-self.beta * llr_lt)
        elif self.args.loss_type == 'divergence':
            loss = ""
        
        loss = loss.mean(dim=0)
        high_translationese_rewards = F.sigmoid(self.beta * llr_ht.detach())
        low_translationese_rewards = F.sigmoid(self.beta * llr_lt.detach())

        self._metrics[mode]['high_translationese_rewards'].append(high_translationese_rewards.mean().item())
        self._metrics[mode]['low_translationese_rewards'].append(low_translationese_rewards.mean().item())

        return (loss, (llr_ht - llr_lt)) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
        if model.high_translationese_model.training and model.low_translationese_model.training:
            model.high_translationese_model.eval()
            model.low_translationese_model.eval()
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
        
        self.model.high_translationese_model.save_pretrained(f"{output_dir}/high_translationese_model", state_dict=state_dict)
        self.model.low_translationese_model.save_pretrained(f"{output_dir}/low_translationese_model", state_dict=state_dict)

        return super()._save(output_dir, state_dict)