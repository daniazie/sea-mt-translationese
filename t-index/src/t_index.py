import os
import yaml
import argparse
import pandas as pd
from functools import partial
from glob import glob
from types import SimpleNamespace

from dataclasses import asdict
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizerBase
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from tqdm import tqdm


from utils import TIndexArgs, encode, preprocess_dataset, tokenize_fn, format_messages
        
class TranslationeseEval:
    """
    Wrapper for Translationese-Index by Liu et al., (2025).
    Original TranslationeseIndex code can be found here: https://github.com/yikang0131/TranslationeseIndex

    Args:
        positive_model ([`str`] or [`PreTrainedModel`]):
            Model fine-tuned on high-translationese samples.
        negative_model  ([`str`] or [`PreTrainedModel`]):
            Model fine-tuned on low-translationese samples.
        tokenizer ([`str`] or [`PreTrainedTokenizerBase`] or [`None`]):
            Processing class to be used to tokenize samples. Uses the model's tokenizer if None is passed.
    """
    def __init__(self, model: str| PreTrainedModel, tokenizer: str | PreTrainedTokenizerBase = None, data_collator=None, use_dpo=False, args: dict | TIndexArgs | None = None):
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
            if tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left', add_eos_token=True)
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, padding_side='left', add_eos_token=True)
        else:
            self.model = model
            self.tokenizer = tokenizer

        if args is None:
            args = TIndexArgs()
        
        if isinstance(args, dict):
            args = TIndexArgs(**args)

        self.args = asdict(args)
        self.batch_size = self.args.pop("batch_size")
        self.use_dpo = use_dpo

        if data_collator is None:
            try: 
                pad_token_id = tokenizer.pad_token_id
            except:
                pad_token_id = tokenizer.pad_token_type_id
            self.data_collator = DataCollatorForLanguageModeling(pad_token_id, max_length=args.max_length)
        else:
            self.data_collator = data_collator

    def compute_log_lklh(self, model: nn.Module, model_inputs: dict):
        outputs = model(**model_inputs)
        logits: torch.Tensor = outputs.logits[:, :-1]
        input_ids: torch.Tensor = model_inputs["input_ids"][:, 1:]
        response_mask: torch.Tensor = model_inputs['attention_mask'][:, 1:].bool()
        
        log_lklh = logits.log_softmax(dim=-1)
        log_lklh = log_lklh.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        log_lklh = log_lklh.masked_fill(~response_mask, 0)
        return log_lklh.sum(dim=1) / response_mask.sum(dim=1)


    def forward(self, dataset, model, tokenizer, data_collator=None):
        args = self.args
        dataset = preprocess_dataset(dataset, eos_token=tokenizer.eos_token, **args)
        if args.get("apply_chat_template") == True:
            dataset = dataset.map(format_messages)
        tokenize_func = partial(tokenize_fn, tokenizer=tokenizer, max_length=args['max_length'], apply_chat_template=args['apply_chat_template'])
  
        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     collate_fn=self.data_collator,
        # )
        
        # dataset = dataset.batch(batch_size=self.batch_size)
        dataset.set_transform(encode, output_all_columns=True)
        results = []
        with torch.no_grad():
            rewards = []
            for data in tqdm(dataset, desc="Evaluating...", total=len(dataset)):
                print(data)
                input_ids = torch.tensor(data['input_ids']).to(model.device)
                mask_label = "attention_mask" if data_collator else "completion_mask"
                completion_mask = torch.tensor(data[mask_label]).to(model.device)

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": completion_mask,
                    "return_output": True
                }

                reward = self.compute_log_lklh(model, model_inputs)
                reward = reward.flatten().cpu().tolist()
                rewards.append({
                    "file_path": data['file_path'],
                    "source": data['prompt'],
                    "translation": data['completion'],
                    "label": data['label'],
                    "log_lklh": reward
                })

        return results

    def __call__(self, data, **args):
        return self.forward(
            data,
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )