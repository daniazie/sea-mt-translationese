# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: sea-mt (3.12.13)
#     language: python
#     name: python3
# ---

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, get_scheduler, HfArgumentParser, set_seed
from peft import LoraConfig, BOFTConfig, AdaLoraConfig, HRAConfig, EvaConfig, PeftModel, get_peft_model, PeftMixedModel
from peft.tuners.lora.config import CordaConfig
from peft.tuners.lora.corda import preprocess_corda
from datasets import Dataset, DatasetDict
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch

# %%
from dataclasses import dataclass, asdict
from typing_extensions import override
from pathlib import Path
from functools import partial
from types import SimpleNamespace
from tqdm import tqdm
import argparse
import copy
import math
import wandb
import yaml
import json
import os

# %%
from data_utils import DataCollatorForPairedModelPreference
from utils import prepare_dataset, format_ds_for_corda

# %%
def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--dataset_name_or_path', type=str, default=None)
    parser.add_argument('--instruction_prompt', type=str, default=None)
    parser.add_argument('--model', type=str, help="Model to fine-tuned", default=None)    
    parser.add_argument('--warmup_ratio', type=float, default=None)
    parser.add_argument('--lr_scheduler', type=str, default=None)

    parser.add_argument('--peft_config', choices=['lora', 'adalora', 'boft', 'hra'], default=None)
    parser.add_argument('--init_lora_weights', choices=['gaussian', 'eva', 'olora', 'pissa', 'corda', 'loftq', 'orthogonal', True], default=True)
    parser.add_argument('--training_args', type=str, default=None)
    return parser


# %%
parser = init_parser()
args = parser.parse_args(args=['--config_file', '../recipes/train.yaml', '--training_args', '../recipes/training_args.yaml'])
set_seed(42)

if args.config_file is not None:
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)
    config_args = SimpleNamespace(**config)

    for k, v in vars(args).items():
        if k not in vars(config_args).keys():
            setattr(config_args, k, v)
    args = config_args

print(json.dumps(vars(args), indent=4))

# %%
tparser = HfArgumentParser(TrainingArguments)
targs = tparser.parse_yaml_file(yaml_file=args.training_args, allow_extra_keys=True)[0]
output_dir = f"{targs.output_dir}_lr_{targs.learning_rate}_ep_{targs.num_train_epochs}_wd_{targs.weight_decay}"
use_peft = args.peft_config is not None

if use_peft:
    output_dir += f"_peft_{args.peft_config}"
    if "lora" in args.peft_config and args.init_lora_weights:
        output_dir += f'_{args.init_lora_weights}'
targs.output_dir = output_dir
experiment_name = output_dir.split('/')[-1]

print(targs)

run = wandb.init(
    project="TranslationeseEval",
    name=experiment_name,
    config = {
        'epochs': targs.num_train_epochs,
        'lr': targs.learning_rate,
        'weight_decay': targs.weight_decay,
        'pairwise_loss_type': 'model' if targs.per_model_loss else 'sample',
        'loss_type': targs.loss_type,
        'peft': {
            'peft_type': args.peft_config,
            'init_lora_weights': args.init_lora_weights
        }
    },
    reinit="finish_previous",
    settings=wandb.Settings(init_timeout=120)
)

# %%
data_seed = torch.manual_seed(42)

# %%
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", add_eos_token=True)

# %%
dataset_dir = args.dataset_name_or_path

# %%
data_files = os.listdir(f"../../{dataset_dir}")
dataset = {}
for data_file in data_files:
    if not 'test' in data_file:
        if not 'cleaned' in data_file:
            continue
        with open(f"../../{dataset_dir}/{data_file}", 'r') as file:
            data_split = data_file.split('/')[-1].split('_')[0]
            data = []
            for line in file.readlines():
                data.append(json.loads(line))
            dataset[data_split] = Dataset.from_list(data)

# %%
dataset

# %%
dataset = DatasetDict(dataset)
tokenized_ds = prepare_dataset(dataset, args.instruction_prompt, tokenizer)
train_set = tokenized_ds['train']
dev_set = tokenized_ds['dev']

# %%
data_collator = DataCollatorForPairedModelPreference(pad_token_id=tokenizer.pad_token_id, padding_side='right')

# %%
train_dataloader = DataLoader(
    train_set,
    batch_size=targs.per_device_train_batch_size,
    collate_fn=data_collator,
    generator=data_seed
)

# %%
eval_dataloader = DataLoader(
    dev_set,
    batch_size=targs.per_device_eval_batch_size,
    collate_fn=data_collator,
    generator=data_seed
)

# %%
if targs.optim.lower() == 'adamw':
    optimizer_class = optim.AdamW
elif targs.optim.lower() == 'adagrad':
    optimizer_class = optim.Adagrad
else:
    optimizer_class = optim.Adam

# %%
if args.lr_scheduler == 'exponential':
    lr_scheduler_args = {
        "gamma": 0.95
    }
if args.lr_scheduler == 'warmup_stable_decay':
    lr_scheduler_args = {
        "num_decay_steps": 5
    }
else:
    lr_scheduler_args = {}

# %%
if args.init_lora_weights == 'corda':
    corda_config = CordaConfig(corda_method='kpm')
else:
    corda_config = None

# %%
target_modules = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ]

# %%
modules_to_save = ['embed_tokens', 'lm_head']

# %%
ensure_weight_tying = getattr(base_model.config.get_text_config(), "tie_word_embeddings", False)

# %%
steps_per_epoch = math.ceil(train_set.num_rows / (targs.per_device_train_batch_size * targs.gradient_accumulation_steps))
total_steps = steps_per_epoch * targs.num_train_epochs
warmup_steps = targs.warmup_steps if isinstance(targs.warmup_steps, int) else math.ceil(total_steps * targs.warmup_steps)

# %%
def run_model():
    ds = dataset['dev'].map(format_ds_for_corda, batched=True, fn_kwargs={"instruction_prompt": args.instruction_prompt, "corda_mode": corda_config.corda_method})
    if corda_config.corda_method == 'kpm':
        ds = ds.select(range(256))
    ds.batch(targs.per_device_eval_batch_size)
    for batch in ds:
        input_ids = batch['text']
        input_ids = input_ids.to(base_model.device)
        with torch.no_grad():
            base_model(input_ids)

# %%
if use_peft:
    if args.peft_config.lower() == 'lora':
        peft_config = LoraConfig(
            r=64,
            lora_alpha=256,
            lora_dropout=0.01,
            init_lora_weights=args.init_lora_weights if args.init_lora_weights else True,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            ensure_weight_tying=ensure_weight_tying,
            use_rslora=True,
            task_type='CAUSAL_LM',
            corda_config=corda_config,
        )
        if args.init_lora_weights == 'corda':
            preprocess_corda(base_model, peft_config, run_model)
    elif args.peft_config.lower() == 'adalora':
        peft_config = AdaLoraConfig(
            r=16,
            lora_alpha=128,
            lora_dropout=0.01,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            init_lora_weights=args.init_lora_weights if args.init_lora_weights else True,
            ensure_weight_tying=ensure_weight_tying,
            use_rslora=True,
            task_type='CAUSAL_LM',
            corda_config=corda_config,
            target_r=16,
            init_r=64,
            tinit=warmup_steps,
            tfinal=math.ceil(total_steps * 0.2),
        )
        if args.init_lora_weights == 'corda':
            preprocess_corda(base_model, peft_config, run_model)
    elif args.peft_config.lower() == 'boft':
        peft_config = BOFTConfig(
            boft_block_size=16,
            boft_n_butterfly_factor=2,
            boft_dropout=0.01,
            target_modules=target_modules,
            modules_to_save=modules_to_save
        )
    elif args.peft_config.lower() == 'hra':
        peft_config = HRAConfig(
            r = 32,
            apply_GS=True,
            target_modules=target_modules,
            modules_to_save=modules_to_save
        )

    ht_model = get_peft_model(copy.deepcopy(base_model), peft_config)
    lt_model = get_peft_model(copy.deepcopy(base_model), peft_config)

    ht_model.print_trainable_parameters()
    lt_model.print_trainable_parameters()

    del base_model

else:
    ht_model = copy.deepcopy(base_model)
    lt_model = copy.deepcopy(base_model)

    print(ht_model.num_parameters(only_trainable=True))
    print(lt_model.num_parameters(only_trainable=True))

    del base_model  

# %%
#ht_optimizer = optimizer_class(list(ht_model.named_parameters()), lr=args.lr, weight_decay=args.weight_decay)
#lt_optimizer = optimizer_class(list(lt_model.named_parameters()), lr=args.lr, weight_decay=args.weight_decay)
optimizer = optimizer_class(list(ht_model.named_parameters()) + list(lt_model.named_parameters()), lr=targs.learning_rate, weight_decay=targs.weight_decay)
if args.lr_scheduler == 'exponential':
    lr_scheduler = ExponentialLR(optimizer=optimizer, **lr_scheduler_args)
else:
    lr_scheduler =  get_scheduler(targs.lr_scheduler_type, optimizer, warmup_steps, total_steps, scheduler_specific_kwargs=lr_scheduler_args)

# %%
def get_llrs(model, input_ids, attention_mask, completion_mask, ht_ids, lt_ids):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    input_ids = input_ids[:, 1:]
    completion_mask = completion_mask[:, 1:].bool()
    log_lklh = logits.log_softmax(dim=-1)
    log_lklh = log_lklh.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    log_lklh = log_lklh.masked_fill(~completion_mask, 0)
    log_lklh_all = log_lklh.sum(dim=1) / completion_mask.sum(dim=1)
    ht_log_lklh = log_lklh_all[:ht_ids.shape[0]]
    lt_log_lklh = log_lklh_all[ht_ids.shape[0]:]
    return ht_log_lklh, lt_log_lklh


# %%
ht_model.cuda()
lt_model.cuda()

# %%
pbar = tqdm(total=total_steps, desc='Training...')
for epoch in range(targs.num_train_epochs):
    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        data = {
            k: v.to("cuda")
            for k, v in data.items()
        }
        log_lklh_ht = get_llrs(ht_model, **data)
        log_lklh_lt = get_llrs(lt_model, **data)
        
        ht_loss = - 2 * (log_lklh_ht[0] - log_lklh_ht[1]).mean()

        lt_loss = - 2 * (log_lklh_lt[0] - log_lklh_lt[1]).mean()

        loss = (ht_loss - lt_loss).mean()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        pbar.update(i)
        print(f"epoch: {epoch+1}, loss: {loss.item()} , ht_reward: {- ht_loss.item()}, lt_reward: {lt_loss.item()}")

pbar.close()

# %%
ht_model.save_pretrained(f"{targs.output_dir}/ht_model", safe_serialization=True)
lt_model.save_pretrained(f"{targs.output_dir}/lt_model", safe_serialization=True)

# %%
if isinstance(ht_model, (PeftModel, PeftMixedModel)):
    ht_model.merge_and_unload()
    lt_model.merge_and_unload()

    ht_model.save_pretrained(f"{args.output_dir}/ht_model/merged", safe_serialization=True)
    lt_model.save_pretrained(f"{args.output_dir}/ht_model/merged", safe_serialization=True)


# %%
tokenizer.save_pretrained(f"{targs.output_dir}")
