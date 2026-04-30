from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.optim.lr_scheduler import LambdaLR
from typing import List, Dict, Union, Any
from typing_extensions import override, Self
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from datasets import Dataset, DatasetDict
from dataclasses import dataclass
from trl.data_utils import is_conversational
import torch.optim as optim
from functools import partial
import numpy as np

def extract_prompt(example):
    for idx in range(min(len(example["ht"]), len(example["lt"]))):
        if example["ht"][idx] != example["lt"][idx]:
            if example["ht"][idx - 1] == " ":
                idx -= 1
            break
    return {
        "prompt": example["ht"][:idx],
        "ht": example["ht"][idx:],
        "lt": example["lt"][idx:],
    }

class LRSchedulerWithWarmUp:
    def __init__(self, optimizer: optim.Optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, lr_scheduler_type: str | None = None, **kwargs):
        self.optimizer = optimizer
        self.lr_scheduler_type = lr_scheduler_type
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.last_epoch = last_epoch

        scheduler_func = self.return_scheduler_func()
        self.lr_lambda = partial(scheduler_func, **kwargs)

    def _get_scaler(self, current_step: int):
        return float(current_step) / float(max(1, self.num_warmup_steps))
    
    def _get_progress_for_cos_scheduler(self, current_step: int):
        return float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))

    def _get_linear_scheduler(self, current_step: int):
        if current_step <= self.num_warmup_steps:
            return self._get_scaler(current_step)
        lr = float(self.num_training_steps - current_step) / float(max, self.num_training_steps - self.num_warmup_steps)
        return max(0.0, lr)
    
    def _get_cosine_scheduler(self, current_step: int, *, num_cycles: float):
        if current_step <= self.num_warmup_steps:
            return self._get_scaler(current_step)
        progress = self._get_progress_for_cos_scheduler(current_step)
        lr = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return lr
    
    def _get_polynomial_scheduler(self, current_step: int, *, power: float, lr_init: int, lr_end: float):
        if current_step <= self.num_warmup_steps:
            return self._get_scaler
        elif current_step > self.num_training_steps:
            return lr_end / lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = self.num_training_steps - self.num_warmup_steps
            pct_remaining = 1 - (current_step - self.num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init
    
    def _get_inverse_sqrt_scheduler(self, current_step: int, *, timescale: int | None = None):
        if timescale is None:
            timescale = self.num_warmup_steps or 10_000
        if current_step <= self.num_warmup_steps:
            return self._get_scaler(current_step)
        
        shift = timescale - self.num_warmup_steps
        decay = 1.0 / math.sqrt((current_step + shift) / timescale)
        return decay
    
    def _get_cosine_scheduler_with_restarts(self, current_step: int, *, num_cycles: float):
        if current_step <= self.num_warmup_steps:
            return self._get_scaler(current_step)
        progress = self._get_progress_for_cos_scheduler(current_step)
        if progress >= 1.0:
            return 0.0
        lr = max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))
        return lr
    
    def _get_wsd_scheduler(self, current_step: int, *, num_decay_steps: int, warmup_type: str, decay_type: str, min_lr_ratio: float, num_cycles: float,):
        num_stable_steps = self.num_training_steps - self.num_warmup_steps - num_decay_steps
        if current_step <= self.num_warmup_steps:
            progress = self._get_scaler(current_step)
            if warmup_type == "linear":
                factor = progress
            elif warmup_type == 'cosine':
                factor = 0.5 * (1.0 - math.cos(math.pi * progress))
            elif warmup_type == "1-sqrt":
                factor = 1.0 - math.sqrt(1.0 - progress)
            
            factor = factor * (1.0 - min_lr_ratio) + min_lr_ratio
            return max(0.0, factor)
        
        if current_step < self.num_warmup_steps + num_stable_steps:
            return 1.0
        
        if current_step < self.num_warmup_steps + num_stable_steps + num_decay_steps:
            progress = float(current_step - self.num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
            if decay_type == 'linear':
                factor = 1.0 - progress
            elif decay_type == 'cosine':
                factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
            elif decay_type == '1-sqrt':
                factor = 1.0 - math.sqrt(progress)

            factor = factor * (1.0 - min_lr_ratio) + min_lr_ratio
            return max(0.0, factor)
        return min_lr_ratio
    
    def return_scheduler_func(self):
        if self.lr_scheduler_type == 'linear' or self.lr_scheduler_type is None:
            return self._get_linear_scheduler
        if self.lr_scheduler_type == 'cosine':
            return self._get_cosine_scheduler
        if self.lr_scheduler_type == 'cosine_with_restarts':
            return self._get_cosine_scheduler_with_restarts
        if self.lr_scheduler_type == 'inverse_sqrt':
            return self._get_inverse_sqrt_scheduler
        if self.lr_scheduler_type == 'wsd':
            return self._get_wsd_scheduler
        raise ValueError(f"Invalid scheduler type: {self.lr_scheduler_type}. Expected 'linear', 'cosine', 'cosine_with_restarts', 'inverse_sqrt' or 'wsd'")
    
    def __call__(self):
        return LambdaLR(self.optimizer, self.lr_lambda, last_epoch=self.last_epoch)
    
def tokenize(text: str | List, tokenizer: PreTrainedTokenizerBase, **kwargs):
    if isinstance(text, list):
        return tokenizer.apply_chat_template(
            text, tokenize=True, return_dict=True, **kwargs
        )
    return tokenizer(text=text)



def tokenize_fn(example, tokenizer):
    output = {}
    if is_conversational(example):
        prompt_ids = tokenize(example['prompt'], tokenizer=tokenizer, add_generation_prompt=True)['input_ids']
    else:
        prompt_ids = tokenize(example['prompt'], tokenizer=tokenizer)
        if not example['ht'].endswith(tokenizer.eos_token):
            example['ht'] += tokenizer.eos_token
        if not example['lt'].endswith(tokenizer.eos_token):
            example['lt'] += tokenizer.eos_token

    prompt_ht_ids = tokenize(example['prompt'] + example['ht'], tokenizer=tokenizer)['input_ids']
    prompt_lt_ids = tokenize(example['prompt'] + example['lt'], tokenizer=tokenizer)['input_ids']

    if not prompt_ht_ids[:len(prompt_ids)] == prompt_ids:
        print("Mismatch between tokenized prompt and tokenized prompt + ht")
    if not prompt_lt_ids[:len(prompt_ids)] == prompt_ids:
        print("Mismatch between tokenized prompt and tokenized prompt + lt")
    
    output["prompt_ids"] = prompt_ids
    output["ht_ids"] = prompt_ht_ids[len(prompt_ids):]
    output["lt_ids"] = prompt_lt_ids[len(prompt_ids):]

    return output

def format_dataset(examples, instruction_prompt, convert_to_chat_template=False):
    if convert_to_chat_template:
        prompts = [[{"role": "user", "content": instruction_prompt + example}] for example in examples['source']]
        hts = [[{"role": "assistant", "content": example}] for example in examples['foreignization']]
        lts = [[{"role": "assistant", "content": example}] for example in examples['domestication']]

    else:
        if examples.get("messages_foreignization") and examples.get("messages_domestication"):
            return {
                "ht": examples['messages_foreignization'],
                "lt": examples['messages_domestication']
            }
        else:
            prompts = [instruction_prompt + example for example in examples['source']]
            hts = [example for example in examples['foreignization']]
            lts = [example for example in examples['domestication']]
        
    return {
        "prompt": prompts,
        "ht": hts,
        "lt": lts
    }

def prepare_dataset(dataset: DatasetDict, instruction_prompt, tokenizer, convert_to_chat_template=False):
    format_fn = partial(format_dataset, instruction_prompt=instruction_prompt, convert_to_chat_template=convert_to_chat_template)
    dataset = dataset.map(format_fn, batched=True)
    if "prompt" not in dataset['train'].column_names:
        dataset = dataset.map(extract_prompt)

    tokenize_func = partial(tokenize_fn, tokenizer=tokenizer)
    tokenized_dataset = dataset.map(tokenize_func, remove_columns=dataset['train'].column_names)
    return tokenized_dataset

def selective_log_softmax(logits: torch.LongTensor, index: torch.LongTensor):
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(dim=-1)

    if logits.dtype in {torch.float32, torch.float64}:
        selected_logits = torch.gather(logits, dim=-1, index=index)

        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values.unsqueeze(dim=-1)
    else:
        per_token_logps = []

        for row_logits, row_labels in zip(logits, index, strict=True):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels)
            per_token_logps.append(row_per_token_logps)

    if squeeze:
        per_token_logps = per_token_logps.squeeze(dim=-1)
    return per_token_logps

def get_log_ratios(model_outputs: CausalLMOutputWithPast, input_ids: torch.LongTensor, completion_mask: torch.LongTensor):
    shift_logits = model_outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_completion_mask = completion_mask[..., 1:].contiguous()

    per_token_logps = selective_log_softmax(shift_logits, shift_labels)
    per_token_logps[shift_completion_mask == 0] = 0.0
    logps = per_token_logps.sum(dim=1)

    ht_logps, lt_logps = logps.chunk(2, dim=0)
    return ht_logps - lt_logps


def compute_joint_loss(ht_model: PreTrainedModel, lt_model: PreTrainedModel, inputs: dict, return_outputs: bool):
    mode = "train" if ht_model.training and lt_model.training else "eval"
    device = ht_model.device
    
    _non_model_keys = {"completion_mask"}
    model_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in _non_model_keys
    }

    model_kwargs['use_cache'] = False

    ht_outputs = ht_model(**model_kwargs)
    lt_outputs = lt_model(**model_kwargs)

    input_ids = inputs['input_ids']
    completion_mask = inputs['completion_mask']

    get_llr = partial(get_log_ratios, input_ids=input_ids, completion_mask=completion_mask)

    ht_llrs = get_llr(ht_outputs)
    lt_llrs = get_llr(lt_outputs)

    comb_scores = ht_llrs + lt_llrs

    loss = - F.logsigmoid(2 * comb_scores)

def format_ds_for_corda(examples, instruction_prompt, corda_mode, tokenizer: PreTrainedTokenizerBase):
    if corda_mode == 'kpm':
        return {
            "text": [instruction_prompt + example for example in examples['source']]
        }
    else:
        texts = []
        if examples.get("messages_foreignization"):
            for ht, lt in zip(examples['messages_foreignization'], examples['messages_domestication']):
                texts.append(tokenizer.apply_chat_template(ht, tokenize=False))
                texts.append(tokenizer.apply_chat_template(lt, tokenize=False))
            return {
                "text": texts
            }

        for src, ht, lt in zip(examples['source'], examples['foreignization'], examples['domestication']):
            texts.append(instruction_prompt + src + '\n\n' + ht)
            texts.append(instruction_prompt + src + '\n\n' + lt)
        return {
            "text": texts
        }