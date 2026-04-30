from datasets import Dataset, interleave_datasets
from dataclasses import dataclass, field
from functools import partial

from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing_extensions import List, Dict, Union
import torch
import json
import numpy as np


import json
import os


def clean_results(data):
    processed = []
    for item in data:
        mt = item['mt']
        if "assistant:" in mt:
            mt = mt.split("assistant:")[-1].strip()
        if "Malay:" in mt:
            mt = mt.split("Malay:")[-1].strip()
        if "\n\n" in mt:
            mt = mt.split('\n\n')[0].strip()

        processed.append({
            "src": item['src'],
            "mt": mt,
            "ref": item['ref']
        })
    return processed

@dataclass
class TranslationeseIndexArgs:
    tgt_lang: str | None = None
    batch_size: int = 32
    max_length: int = 1024
    padding: bool | str | PaddingStrategy = True
    apply_chat_template: bool = False

def prepare_dataset(examples, tgt_lang, **kwargs):
    prompts = []
    completions = [completion.strip() for completion in examples['mt']]
    prompt_template = """Translate the following text to {tgt_lang}:\n\n{src}"""
    for prompt in examples['src']:
        prompts.append(prompt_template.format(tgt_lang=tgt_lang, src=prompt.strip()).strip())

    samples = {
        "prompt": prompts,
        "completion": completions
    }

    return samples

def preprocess_dataset(dataset, tgt_lang, **kwargs) -> Dataset:
    dataset = Dataset.from_list(dataset)
    prep_dataset = partial(prepare_dataset, tgt_lang=tgt_lang, **kwargs)
    return dataset.map(prep_dataset, batched=True)

def _tokenize(inputs, tokenizer: PreTrainedTokenizerBase, apply_chat_template: bool = False, **kwargs):
    if apply_chat_template:
        return tokenizer.apply_chat_template(
            inputs,
            **kwargs
        )
    else:
        return tokenizer(inputs)

def tokenize_fn(example, tokenizer: PreTrainedTokenizerBase, max_length=1024, apply_chat_template: bool = False):
    output = {}
    if apply_chat_template:
        prompt_ids: torch.Tensor = _tokenize(
            example['prompt'],
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors='pt'
        )['input_ids'].flatten()

        prompt_completion_processed = _tokenize(
            example['prompt'] + example['completion'],
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            return_dict=True,
            padding='max_length',
            max_length=max_length,
            tokenize=True,
            return_tensors='pt'
        )

    else:
        prompt_ids = _tokenize(
            example['prompt'],
            tokenizer=tokenizer,
            return_tensors='pt'
        )['input_ids']

        prompt_completion_processed = _tokenize(
            example['prompt'] + example['completion'] + tokenizer.eos_token,
            tokenizer=tokenizer,
            return_tensors='pt'
        )
        
    prompt_completion_ids: torch.Tensor = prompt_completion_processed["input_ids"].flatten()

    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
    output['prompt'] = example['prompt']
    output['completion'] = example['completion']
    output['input_ids'] = prompt_completion_ids
    output['attention_mask'] = [1] * len(prompt_completion_ids)
    output['completion_mask'] = completion_mask
    
    return output

def encode(batch, tokenizer: PreTrainedTokenizerBase, max_length, apply_chat_template: bool = False):
    if apply_chat_template:
        prompt_ids = _tokenize(
            batch['prompt'],
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors='pt'
        )['input_ids']

        prompt_completion_processed = _tokenize(
            batch['prompt'] + batch['completion'],
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            tokenize=True,
            max_length=max_length,
            return_dict=True,
            return_tensors='pt'
        )
    else:
        prompt_ids = _tokenize(
            batch['prompt'],
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            return_tensors='pt'
        )['input_ids']

        prompt_completion_processed = _tokenize(
            batch['prompt'] + batch['completion'] + [tokenizer.eos_token] * len(batch['completion']),
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            return_tensors='pt'
        )

    prompt_completion_ids = prompt_completion_processed["input_ids"]
    if not prompt_completion_ids[:len(prompt_ids) - 1] == prompt_ids[:-1]:
        print(prompt_completion_ids[:len(prompt_ids)])

    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))

    return {
        "input_ids": prompt_completion_ids,
        "completion_mask": completion_mask,
    }

TIndexArgs = TranslationeseIndexArgs


def format_messages_for_preference(examples, is_vl, src_lang, tgt_lang):
    srcs = [example for example in examples['source']]
    refs = [example for example in examples['domestication']]
    mts = [example for example in examples['foreignization']]

    prompts = []
    completions_chosen = []
    completions_rejected = []
    sys_prompt = "Translate the given sentence from {src_lang} to {tgt_lang}.\n\n"
    user_prompt = "{src_lang}: {src}\n\n{tgt_lang}: "
    for src, ref, mt in zip(srcs, refs, mts):
        prompt = [
            {
                "role": "user", "content": [{"type": "text", "text": sys_prompt + user_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)}] if is_vl else sys_prompt + user_prompt.format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)
            }
        ]

        chosen = [
            {
            "role": "assistant", "content": [{"type": "text", "text": mt}] if is_vl else mt
            }
        ]
        
        rejected = [
            {
            "role": "assistant", "content": [{"type": "text", "text": ref}] if is_vl else ref
            }
        ]
        
        prompts.append(prompt)
        completions_chosen.append(chosen)
        completions_rejected.append(rejected)

    return {"prompt": prompts, "chosen": completions_chosen, "rejected": completions_rejected}

def format_messages(example):
    prompt = [{"role": "user", "content": example['prompt']}]
    completion = [{"role": "assistant", "content": example['completion']}]
    return {
        "prompt": prompt,
        "completion": completion,
    }

def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = 'right',
    pad_to_multiple_of: int | None = None,
    max_length: int | None = None
) -> torch.Tensor:
    output_shape = [max_length] if max_length else np.max([t.shape for t in tensors], 0).tolist()

    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == 'left':
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == 'right':
            seq_start = 0
        else:
            raise ValueError("Invalid padding_side.")
        
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output