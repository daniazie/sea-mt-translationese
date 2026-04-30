from transformers.data.data_collator import DataCollatorMixin
from transformers import PreTrainedTokenizerBase, ProcessorMixin
import datasets
from torch.utils.data import Dataset
import torch
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

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

@dataclass
class DataCollatorForPairedModelPreference(DataCollatorMixin):
    pad_token_id: int
    max_length: int | None = None
    truncation_mode: str = "keep_start"
    padding_side: str = 'right'
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def prepare_data(self, examples):
        prompt_high_translationese_ids = [example["prompt_ids"] + example['high_translationese_ids'] for example in examples]
        prompt_low_translationese_ids = [example['prompt_ids'] + example['low_translationese_ids'] for example in examples]
        high_translationese_mask = [[0] * len(example['prompt_ids']) +  [1] * len(example['high_translationese_ids']) for example in examples]
        low_translationese_mask = [[0] * len(example['prompt_ids']) +  [1] * len(example['low_translationese_ids']) for example in examples]

        if self.max_length is not None:
            if self.truncation_mode == "keep_start":
                sl = slice(None, self.max_length)
            elif self.truncation_mode == 'keep_end':
                sl = slice(-self.max_length, None)
            else:
                raise ValueError(f"Unsupported truncation type.")
            
            prompt_high_translationese_ids = [ids[sl] for ids in prompt_high_translationese_ids]
            prompt_low_translationese_ids = [ids[sl] for ids in prompt_low_translationese_ids]
            high_translationese_mask = [mask[sl] for mask in high_translationese_mask]
            low_translationese_mask = [mask[sl] for mask in low_translationese_mask]

        high_translationese_attention_mask = [[1] * len(ids) for ids in prompt_high_translationese_ids]
        low_translationese_attention_mask = [[1] * len(ids) for ids in prompt_low_translationese_ids]

        return prompt_high_translationese_ids, high_translationese_attention_mask, high_translationese_mask, prompt_low_translationese_ids, low_translationese_attention_mask, low_translationese_mask
    

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_high_translationese_ids, high_translationese_attention_mask, high_translationese_mask, prompt_low_translationese_ids, low_translationese_attention_mask, low_translationese_mask = self.prepare_data(examples)
        
        prompt_high_translationese_ids = [torch.tensor(ids) for ids in prompt_high_translationese_ids]
        prompt_low_translationese_ids = [torch.tensor(ids) for ids in prompt_low_translationese_ids]
        high_translationese_attention_mask = [torch.tensor(mask) for mask in high_translationese_attention_mask]
        low_translationese_attention_mask = [torch.tensor(mask) for mask in low_translationese_attention_mask]
        high_translationese_mask = [torch.tensor(mask) for mask in high_translationese_mask]
        low_translationese_mask = [torch.tensor(mask) for mask in low_translationese_mask]

        max_length = np.max([np.max([t.shape for t in prompt_high_translationese_ids], 0).item(), np.max([t.shape for t in prompt_low_translationese_ids], 0).item()]).item()

        prompt_high_translationese_ids = pad(
            prompt_high_translationese_ids,
            padding_value=self.pad_token_id,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=max_length
        )

        prompt_low_translationese_ids = pad(
            prompt_low_translationese_ids,
            padding_value=self.pad_token_id,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=max_length
        )

        high_translationese_attention_mask = pad(
            high_translationese_attention_mask,
            padding_value=0,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=max_length
        )
        low_translationese_attention_mask = pad(
            low_translationese_attention_mask,
            padding_value=0,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=max_length
        )

        high_translationese_mask = pad(
            high_translationese_mask,
            padding_value=0,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=max_length
        )

        low_translationese_mask = pad(
            low_translationese_mask,
            padding_value=0,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=max_length
        )

        output = {}
        output['input_ids'] = torch.cat((prompt_high_translationese_ids, prompt_low_translationese_ids), dim=0)
        output['attention_mask'] = torch.cat((high_translationese_attention_mask, low_translationese_attention_mask), dim=0)
        output['completion_mask'] = torch.cat((high_translationese_mask, low_translationese_mask), dim=0)
        output['high_translationese_ids'] = prompt_high_translationese_ids
        output['low_translationese_ids'] = prompt_low_translationese_ids
    
        return output

def format_dataset(
    dataset: datasets.Dataset | datasets.DatasetDict | datasets.IterableDataset | datasets.IterableDatasetDict, 
    prompt_key: str | None = None,
    high_translationese_key: str | None = None,
    low_translationese_key: str | None = None,
    instruction_prompt: str | None = None,
    tgt_lang: str | None = None,
    convert_to_chat: bool = False
):
    def extract_prompt(
        example,
        high_translationese_key: str | None = None,
        low_translationese_key: str | None = None,
    ):
        for idx in range(min(len(example[high_translationese_key]), len(example[low_translationese_key]))):
            if example[high_translationese_key][idx] != example[low_translationese_key][idx]:
                if example[high_translationese_key][idx - 1] == " ":
                    idx -= 1
                break
        return {
            "prompt": example[high_translationese_key][:idx],
            "high_translationese": example[high_translationese_key][idx:],
            "low_translationese": example[low_translationese_key][idx:],
        }

    if prompt_key is None:
        dataset = dataset.map(extract_prompt, fn_kwargs={"high_translationese_key": high_translationese_key, "low_translationese_key": low_translationese_key})
        return dataset

    def convert_to_chat_template(
        examples,
        prompt_key: str | None = None,
        high_translationese_key: str | None = None,
        low_translationese_key: str | None = None,
        instruction_prompt: str | None = None,
        tgt_lang: str | None = None
    ):
        if tgt_lang is None and instruction_prompt is None:
            raise ValueError("'tgt_lang' and 'instruction_prompt' cannot both be None.")
        if instruction_prompt is None:
            instruction_prompt = "Translate the following text into {tgt_lang}:\n".format(tgt_lang.capitalize())
        
        prompts = []
        high_translationese_completions = []
        low_translationese_completions = []

        for prompt, ht_completion, lt_completion in zip(examples[prompt_key], examples[high_translationese_key], examples[low_translationese_key]):
            _prompt = [
                {
                    "role": "user", "content": instruction_prompt + prompt
                }
            ]

            _ht_completion = [
                {
                    "role": "assistant", "content": ht_completion
                }
            ]

            _lt_completion = [
                {
                    "role": "assistant", "content": lt_completion
                }
            ]

            prompts.append(_prompt)
            high_translationese_completions.append(_ht_completion)
            low_translationese_completions.append(_lt_completion)

        return {
            "prompt": prompts,
            "high_translationese": high_translationese_completions,
            "low_translationese": low_translationese_completions
        }
        
    def format_samples(examples, prompt_key, high_translationese_key, low_translationese_key):
        return {
            "prompt": examples[prompt_key],
            "high_translationese": examples[high_translationese_key],
            "low_translationese": examples[low_translationese_key]
        }

    if convert_to_chat:
        dataset = dataset.map(convert_to_chat_template, batched=True, fn_kwargs={"instruction_prompt": instruction_prompt, "tgt_lang": tgt_lang})
        return dataset
    
    dataset = dataset.map(format_samples, fn_kwargs={"prompt_key": prompt_key, "high_translationese_key": high_translationese_key, "low_translationese_key": low_translationese_key})
    return dataset

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

def unpair_dataset(examples):
    high_translationese = [ht for ht in examples['high_translationese']]
    low_translationese = [lt for lt in examples['low_translationese']]

    prompts = []
    completions = []
    labels = []
    for prompt, ht, lt in zip(examples['prompt'], high_translationese, low_translationese):
        prompts.append(prompt)
        completions.append(ht)
        labels.append(True)
        prompts.append(prompt)
        completions.append(lt)
        labels.append(False)
    return {
        "prompt": prompts,
        "completion": completions,
        'label': labels
    }

@dataclass
class DataCollatorForUnpairedPreference(DataCollatorMixin):
    pad_token_id: int
    max_length: int | None = None
    truncation_mode: str = "keep_start"
    padding_side: str = 'right'
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def prepare_data(self, examples):
        prompt_completion_ids = [example["prompt_ids"] + example['completion_ids'] for example in examples]
        completion_mask = [[0] * len(example['prompt_ids']) +  [1] * len(example['completion_ids']) for example in examples]
        labels = [example['labels'] for example in examples]

        if self.max_length is not None:
            if self.truncation_mode == "keep_start":
                sl = slice(None, self.max_length)
            elif self.truncation_mode == 'keep_end':
                sl = slice(-self.max_length, None)
            else:
                raise ValueError(f"Unsupported truncation type.")
            
            prompt_completion_ids = [ids[sl] for ids in prompt_completion_ids]
            completion_mask = [mask[sl] for mask in completion_mask]

        attention_mask = [[1] * len(ids) for ids in prompt_completion_ids]

        return prompt_completion_ids, completion_mask, attention_mask, labels
    

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_completion_ids, completion_mask, attention_mask, labels = self.prepare_data(examples)
        
        prompt_completion_ids = [torch.tensor(ids) for ids in prompt_completion_ids]
        attention_mask = [torch.tensor(mask) for mask in attention_mask]
        completion_mask = [torch.tensor(mask) for mask in completion_mask]
        labels = [torch.tensor(label).int() for label in labels]

        prompt_completion_ids = pad(
            prompt_completion_ids,
            padding_value=self.pad_token_id,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )


        attention_mask = pad(
            attention_mask,
            padding_value=0,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        completion_mask = pad(
            completion_mask,
            padding_value=0,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        labels = pad(
            labels,
            padding_value=self.pad_token_id,
            padding_side=self.padding_side, 
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        output = {}
        output['input_ids'] = prompt_completion_ids
        output['attention_mask'] = attention_mask
        output['completion_mask'] = completion_mask
        output['labels'] = labels
    
        return output