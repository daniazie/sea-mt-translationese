from transformers import LlamaForCausalLM, PreTrainedTokenizerBase, ProcessorMixin, AutoProcessor, PreTrainedModel
from transformers.trainer_utils import unwrap_peft_model
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.utils import TransformersKwargs
from peft import PeftModel, PeftMixedModel
import torch.nn.functional as F
import torch.nn as nn
import torch

from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
from typing import Any

import numpy as np

from data_utils import pad

class TranslationeseEval(nn.Module):
    def __init__(self, ht_model: PreTrainedModel, lt_model: PreTrainedModel, processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None):
        super().__init__()

        self.ht_model = ht_model
        self.lt_model = lt_model

        if processing_class is None:
            model_name_or_path = self.ht_model.config._name_or_path
            processing_class = AutoProcessor.from_pretrained(model_name_or_path)

        if isinstance(processing_class, ProcessorMixin):
            self.tokenizer: PreTrainedTokenizerBase = processing_class.tokenizer
        else:
            self.tokenizer = processing_class

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def get_log_ratios(self, model_outputs: CausalLMOutputWithPast, input_ids: torch.LongTensor, completion_mask: torch.LongTensor, ht_ids: torch.Tensor = None, lt_ids: torch.Tensor = None, pairwise_eval=False):
        shift_logits = model_outputs.logits[:, :-1]
        shift_labels = input_ids[:, 1:]
        shift_completion_mask = completion_mask[:, 1:].bool()

        log_lklh = shift_logits.log_softmax(dim=-1)
        log_lklh = log_lklh.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        log_lklh = log_lklh.masked_fill(~shift_completion_mask, 0)
        log_lklh = log_lklh.sum(dim=1) / shift_completion_mask.sum(dim=1)

        if ht_ids is not None and lt_ids is not None:
            log_lklh_ht = log_lklh[:ht_ids.shape[0]]
            log_lklh_lt = log_lklh[ht_ids.shape[0]:]
            if self.in_training:
                return log_lklh_ht, log_lklh_lt
            return log_lklh_ht - log_lklh_lt
        return log_lklh

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        completion_mask: torch.Tensor | None = None,
        ht_ids: torch.Tensor | None = None,
        lt_ids: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        training_mode: bool | None = None,
        pairwise_eval: bool = False,
        **kwargs: Unpack[TransformersKwargs]
    ) -> torch.Tensor:
        self.in_training = training_mode
        
        ht_outputs: CausalLMOutputWithPast = self.ht_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs
        )

        lt_outputs: CausalLMOutputWithPast = self.lt_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs
        )

        

        ht_llrs = self.get_log_ratios(ht_outputs, input_ids, completion_mask, pairwise_eval=pairwise_eval, ht_ids=ht_ids, lt_ids=lt_ids)
        lt_llrs = self.get_log_ratios(lt_outputs, input_ids, completion_mask, pairwise_eval=pairwise_eval, ht_ids=ht_ids, lt_ids=lt_ids)
        
        if self.in_training:
            return ht_llrs, lt_llrs

        llrs = ht_llrs - lt_llrs

        return llrs
    
    def convert_to_conversational(self, role, contents):
        return [
            [
                {"role": role, "content": content}
            ]
            for content in contents
        ]
    
    def tokenize(self, text: str | list, **kwargs):
        if isinstance(text, list):
            return self.tokenizer.apply_chat_template(
                text,
                tokenize=True,
                return_dict=True,
                **kwargs
            )
        return self.tokenizer(text=text)
    
    def tokenize_fn(self, example):
        prompt_ids = self.tokenize(example['prompt'], add_generation_prompt=True)['input_ids']

        tokenize_completions = lambda completion: self.tokenize(example['prompt'] + completion)['input_ids']
        
        if example.get("completion"):
            completion_ids = tokenize_completions(example['completion'])
            return {
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids
            }
        else:
            ht_ids = tokenize_completions(example['ht'])
            lt_ids = tokenize_completions(example['lt'])

            return {
                "prompt_ids": prompt_ids,
                "ht_ids": ht_ids,
                "lt_ids": lt_ids
            }
    
    def collate_fn(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if examples[0].get("completion_ids"):
            input_ids = [example['prompt_ids'] + example['completion_ids'] for example in examples]
            completion_mask = [[0] * len(example['prompt_ids']) + [1] * len(example['completion_ids']) for example in examples]
            attention_mask = [[1] * len(ids) for ids in input_ids]

        else:
            prompt_ht_ids = [example['prompt_ids'] + example['ht_ids'] for example in examples]
            prompt_lt_ids = [example['prompt_ids'] + example['lt_ids'] for example in examples]

            ht_mask = [[0] * len(example['prompt_ids']) + [1] * len(example['ht_ids']) for example in examples]
            lt_mask = [[0] * len(example("prompt_ids")) + [1] * len(example['lt_ids']) for example in examples]

            ht_attention_mask = [[1] * len(ids) for ids in prompt_ht_ids]
            lt_attention_mask = [[1] * len(ids) for ids in prompt_lt_ids]

            input_ids = prompt_ht_ids + prompt_lt_ids
            completion_mask = ht_mask + lt_mask
            attention_mask = ht_attention_mask + lt_attention_mask

        input_ids = [torch.tensor(ids) for ids in input_ids]
        attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in attention_mask]
        completion_mask = [torch.tensor(mask, dtype=torch.long) for mask in completion_mask]

        output = {}

        output['input_ids'] = pad(
            input_ids,
            padding_value=self.tokenizer.pad_token_id,
            padding_side='right',
        )

        output["attention_mask"] = pad(
            attention_mask,
            padding_value=0,
            padding_side='right'
        )

        output['completion_mask'] = pad(
            completion_mask,
            padding_value=0,
            padding_side='right'
        )

        return output

    def prepare_data_for_computation(self, srcs: list[str], mts: list[str] | None = None, ht_mts: list[str] | None = None, lt_mts: list[str] | None = None, tgt_lang_name: str | None = None, seed: int = 42):
        if tgt_lang_name:
            prompt_prefix = f"Translate the following text to {tgt_lang_name}.\n\n"
        else:
            prompt_prefix = ""
        prompts = [(prompt_prefix + src).strip() for src in srcs]
        data = []

        if self.format_conversational:
            prompts = self.convert_to_conversational("user", prompts)
            if mts:
                mts = self.convert_to_conversational("assistant", mts)
            else:
                ht_mts = self.convert_to_conversational("assistant", ht_mts)
                lt_mts = self.convert_to_conversational("assistant", lt_mts)

        if mts:
            for prompt, mt in zip(prompts, mts):
                data.append({
                    "prompt": prompt,
                    "completion": mt
                })

        else:
            for prompt, ht, lt in zip(prompts, ht_mts, lt_mts):
                data.append({
                    "prompt": prompt,
                    "ht": ht,
                    "lt": lt
                })

        dataset = Dataset.from_list(data)
        dataset = dataset.map(self.tokenize_fn)

        dataloader = DataLoader(
            dataset,
            batch_size=self.compute_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        return dataloader

    def compute(self, srcs: list[str], mts: list[str] | None = None, ht_mts: list[str] | None = None, lt_mts: list[str] | None = None, tgt_lang_name: str | None = None, compute_batch_size: int = 16, return_as_probs: bool = False, format_conversational: bool = False, seed: int = 42):
        pairwise_eval = mts is None and (ht_mts is not None and lt_mts is not None)
        self.format_conversational = format_conversational
        self.compute_batch_size = compute_batch_size

        dataloader = self.prepare_data_for_computation(srcs, mts, ht_mts, lt_mts, tgt_lang_name, seed)
        with torch.no_grad():
            scores = []
            for data in tqdm(dataloader, desc="Evaluating...", total=len(dataloader)):
                input_ids, attention_mask, completion_mask = data

                llrs = self.forward(input_ids=input_ids, attention_mask=attention_mask, completion_mask=completion_mask, pairwise_eval=pairwise_eval)
                if return_as_probs:
                    probs = F.sigmoid(llrs)
                    scores += probs.tolist()
                else:
                    preds = (llrs.numpy() > 0).astype(int)
                    scores += preds.tolist()
            
        return {
            "scores": scores,
            "system_score": np.mean(scores).item()
        }
    
    def save_pretrained(self, output_dir, **kwargs):
        self.ht_model.save_pretrained(f"{output_dir}/ht_model", **kwargs)
        self.lt_model.save_pretrained(f"{output_dir}/lt_model", **kwargs)
        if isinstance(self.ht_model, (PeftMixedModel, PeftModel)):
            ht_model = self.ht_model.merge_and_unload()
            ht_model.save_pretrained(f"{output_dir}/ht_model", **kwargs)

        if isinstance(self.lt_model, (PeftMixedModel, PeftModel)):
            lt_model = self.lt_model.merge_and_unload()
            lt_model.save_pretrained(f"{output_dir}/lt_model", **kwargs)

