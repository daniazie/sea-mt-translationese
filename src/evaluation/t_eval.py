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
import numpy as np


from utils import TIndexArgs, encode, preprocess_dataset, tokenize_fn, format_messages, pad
        
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
    def __init__(self, model: str, data_collator=None, is_reward_model=False, apply_chat_template=False, batch_size: int = 32):
        self.positive_model = AutoModelForCausalLM.from_pretrained(f"{model}/positive")
        self.negative_model = AutoModelForCausalLM.from_pretrained(f"{model}/negative")
        self.tokenizer = AutoTokenizer.from_pretrained(f"{model}/positive", fix_mistral_regex=True)

        self.apply_chat_template = apply_chat_template
        self.batch_size = batch_size
        self.is_reward_model = is_reward_model

        if data_collator is None:
            try: 
                pad_token_id = self.tokenizer.pad_token_id
            except:
                pad_token_id = self.tokenizer.pad_token_type_id
            self.data_collator = DataCollatorForLanguageModeling(pad_token_id)
        else:
            self.data_collator = data_collator

    def compute_log_lklh(self, model: nn.Module, model_inputs: dict):
        model.cuda()
        model_inputs = {
            k: v.to(model.device)
            for k, v in model_inputs.items()
        }
        if self.is_reward_model:
            model_inputs['return_output'] = True
            reward, outputs = model(**model_inputs)
            return reward
        else:
            completion_mask: torch.Tensor = model_inputs.pop("completion_mask")
            outputs = model(**model_inputs)
            logits: torch.Tensor = outputs.logits[:, :-1]
            input_ids: torch.Tensor = model_inputs["input_ids"][:, 1:]
            response_mask: torch.Tensor = completion_mask[:, 1:].bool()
            
            log_lklh = logits.log_softmax(dim=-1)
            log_lklh = log_lklh.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
            try: 
                log_lklh = log_lklh.masked_fill(~response_mask, 0)
            except:
                print(log_lklh.shape, response_mask.shape)

            return log_lklh.sum(dim=1) / response_mask.sum(dim=1)

    def collate_fn(self, examples):
        input_ids = torch.tensor([example['input_ids'] for example in examples])
        attention_mask = torch.tensor([example['attention_mask'] for example in examples])
        completion_mask = torch.tensor([example['completion_mask'] for example in examples])

        input_ids = pad(
            input_ids,
            padding_value=self.tokenizer.pad_token_id,
            padding_side='left',
        )

        attention_mask = pad(
            attention_mask,
            padding_value=0,
            padding_side='left',
        )

        completion_mask = pad(
            completion_mask,
            padding_value=0,
            padding_side='left',
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask
        }

    def forward(self, dataset, tgt_lang):
        tokenizer = self.tokenizer
        data_collator = self.data_collator
        dataset = preprocess_dataset(dataset, tgt_lang=tgt_lang, eos_token=tokenizer.eos_token)
        if self.apply_chat_template:
            dataset = dataset.map(format_messages)
        tokenize_func = partial(tokenize_fn, tokenizer=tokenizer, apply_chat_template=self.apply_chat_template)
        dataset = dataset.map(tokenize_func)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        print(next(iter(dataloader)))
        
        #dataset = dataset.batch(batch_size=self.batch_size)
        #transform = partial(tokenize_fn, tokenizer=tokenizer, apply_chat_template=self.apply_chat_template)
        #dataset.set_transform(transform, output_all_columns=True)

        #dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'completion_mask'])
        
        results = []
        llrs = []
        with torch.no_grad():
            for data in tqdm(dataloader, desc="Evaluating...", total=len(dataloader)):
                model_inputs = {
                    "input_ids": data['input_ids'],
                    "attention_mask": data['attention_mask'],
                    "completion_mask": data['completion_mask']
                }

                log_lklh_positive = self.compute_log_lklh(self.positive_model, model_inputs).flatten().cpu()
                log_lklh_negative = self.compute_log_lklh(self.negative_model, model_inputs).flatten().cpu()
                llr = log_lklh_positive - log_lklh_negative
                llrs += llr.tolist()

        rewards = (np.array(llrs) < 0).astype(int)
        return {
            "scores": rewards,
            "mean_score": np.mean(rewards).item()
        }

    def compute(self, data, tgt_lang, **args):
        return self.forward(
            data,
            tgt_lang=tgt_lang
        )