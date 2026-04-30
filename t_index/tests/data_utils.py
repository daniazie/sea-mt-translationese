from transformers.data.data_collator import DataCollatorMixin
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
        prompt_ht_ids = [example["prompt_ids"] + example['ht_ids'] for example in examples]
        prompt_lt_ids = [example['prompt_ids'] + example['lt_ids'] for example in examples]
        ht_mask = [[0] * len(example['prompt_ids']) +  [1] * len(example['ht_ids']) for example in examples]
        lt_mask = [[0] * len(example['prompt_ids']) +  [1] * len(example['lt_ids']) for example in examples]

        if self.max_length is not None:
            if self.truncation_mode == "keep_start":
                sl = slice(None, self.max_length)
            elif self.truncation_mode == 'keep_end':
                sl = slice(-self.max_length, None)
            else:
                raise ValueError(f"Unsupported truncation type.")
            
            prompt_ht_ids = [ids[sl] for ids in prompt_ht_ids]
            prompt_lt_ids = [ids[sl] for ids in prompt_lt_ids]
            ht_mask = [mask[sl] for mask in ht_mask]
            lt_mask = [mask[sl] for mask in lt_mask]

        ht_attention_mask = [[1] * len(ids) for ids in prompt_ht_ids]
        lt_attention_mask = [[1] * len(ids) for ids in prompt_lt_ids]

        return prompt_ht_ids, ht_attention_mask, ht_mask, prompt_lt_ids, lt_attention_mask, lt_mask
    

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_ht_ids, ht_attention_mask, ht_mask, prompt_lt_ids, lt_attention_mask, lt_mask = self.prepare_data(examples)
        
        prompt_ht_ids = [torch.tensor(ids) for ids in prompt_ht_ids]
        prompt_lt_ids = [torch.tensor(ids) for ids in prompt_lt_ids]
        ht_attention_mask = [torch.tensor(mask) for mask in ht_attention_mask]
        lt_attention_mask = [torch.tensor(mask) for mask in lt_attention_mask]
        ht_mask = [torch.tensor(mask) for mask in ht_mask]
        lt_mask = [torch.tensor(mask) for mask in lt_mask]

        max_length = np.max([np.max([t.shape for t in prompt_ht_ids], 0).item(), np.max([t.shape for t in prompt_lt_ids], 0).item()]).item()

        prompt_ht_ids = pad(
            prompt_ht_ids,
            padding_value=self.pad_token_id,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=max_length
        )

        prompt_lt_ids = pad(
            prompt_lt_ids,
            padding_value=self.pad_token_id,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=max_length
        )

        ht_attention_mask = pad(
            ht_attention_mask,
            padding_value=0,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=max_length
        )
        lt_attention_mask = pad(
            lt_attention_mask,
            padding_value=0,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=max_length
        )

        ht_mask = pad(
            ht_mask,
            padding_value=0,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=max_length
        )

        lt_mask = pad(
            lt_mask,
            padding_value=0,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=max_length
        )

        output = {}
        output['input_ids'] = torch.cat((prompt_ht_ids, prompt_lt_ids), dim=0)
        output['attention_mask'] = torch.cat((ht_attention_mask, lt_attention_mask), dim=0)
        output['completion_mask'] = torch.cat((ht_mask, lt_mask), dim=0)
        output['ht_ids'] = prompt_ht_ids
        output['lt_ids'] = prompt_lt_ids
    
        return output
    

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
        labels = [torch.tensor(label) for label in labels]

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