from datasets import Dataset, interleave_datasets
from dataclasses import dataclass, field
from functools import partial

from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing_extensions import List, Dict, Union
import torch
import json
import numpy as np


@dataclass
class TranslationeseIndexArgs:
    prompt_key: str
    completion_key: str | None = None
    positive_key: str | None = None
    negative_key: str | None = None
    prompt_template: str | None = None
    batch_size: int = 32
    max_length: int = 1024
    padding: bool | str | PaddingStrategy = True
    apply_chat_template: bool = False
    device: str = 'cpu'

def prepare_dataset(examples, prompt_template, **kwargs):
    prompts = []
    completions = [completion for completion in examples['completion']]

    for prompt in examples['prompt']:
        prompts.append(prompt_template.format(input=prompt))

    samples = {
        "prompt": prompts,
        "completion": completions
    }

    if examples.get('label'):
        samples['label'] = [example for example in examples['label']]
    
    if examples.get('file_path'):
        samples['file_path'] = [example for example in examples['file_path']]

    return samples

def preprocess_dataset(dataset: Dataset, prompt_key, prompt_template, positive_key, negative_key, completion_key, **kwargs):
    if positive_key and negative_key:
        neg_dataset = dataset.select_columns(["file_path", prompt_key, negative_key])
        neg_dataset = neg_dataset.rename_columns({prompt_key: "prompt", negative_key: "completion"})
        neg_dataset = neg_dataset.add_column("label", column=[0] * neg_dataset.num_rows)
        pos_dataset = dataset.select_columns(["file_path", prompt_key, positive_key])
        pos_dataset = pos_dataset.rename_columns({prompt_key: "prompt", positive_key: "completion"})
        pos_dataset = pos_dataset.add_column("label", column=[1] * pos_dataset.num_rows)

        ds = interleave_datasets([pos_dataset, neg_dataset])
    else:
        ds = dataset.rename_columns({prompt_key: "prompt", completion_key: "completion"})

    prep_dataset = partial(prepare_dataset, prompt_template=prompt_template, **kwargs)
    return ds.map(prep_dataset, batched=True)

def _tokenize(inputs, tokenizer: PreTrainedTokenizerBase, apply_chat_template: bool = False, **kwargs):
    if apply_chat_template:
        return tokenizer.apply_chat_template(
            inputs,
            **kwargs
        )
    else:
        return tokenizer(inputs)

def tokenize_fn(example, tokenizer: PreTrainedTokenizerBase, max_length, apply_chat_template: bool = False):
    output = {}

    if apply_chat_template:
        prompt_ids = _tokenize(
        example['prompt'],
        tokenizer=tokenizer,
        apply_chat_template=apply_chat_template,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt'
    )['input_ids']

        prompt_completion_processed = _tokenize(
            example['prompt'] + example['completion'],
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            tokenize=True,
            max_length=max_length,
            return_dict=True,
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
        
    prompt_completion_ids = prompt_completion_processed["input_ids"]
    if not prompt_completion_ids[:len(prompt_ids) - 1] == prompt_ids[:-1]:
        print(prompt_completion_ids[:len(prompt_ids)])

    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
    output['file_path'] = example['file_path']
    output['prompt'] = example['prompt']
    output['completion'] = example['completion']
    if example.get('label'):
        output['label'] = example['label']
    output['input_ids'] = prompt_completion_ids
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
        "file_path": example['file_path'],
        "prompt": prompt,
        "completion": completion,
        "label": example['label']
    }