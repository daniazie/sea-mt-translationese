from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    EarlyStoppingCallback,
    HfArgumentParser, 
    set_seed
)
from peft import LoraConfig, BOFTConfig, AdaLoraConfig, HRAConfig, EvaConfig, PeftModel, get_peft_model, PeftMixedModel
from peft.tuners.lora.config import CordaConfig
from peft.tuners.lora.corda import preprocess_corda
from datasets import Dataset, DatasetDict
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from safetensors.torch import save_model
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch

from dataclasses import dataclass, asdict
from typing_extensions import override
from pathlib import Path
from functools import partial
from types import SimpleNamespace
from tqdm import tqdm
import argparse
import wandb
import copy
import math
import yaml
import json
import os

from ensemble_model import TranslationeseEval
from t_index.src.ensemble_trainer import TEvalTrainer, TEvalTrainingArguments
from data_utils import DataCollatorForPairedModelPreference, format_dataset, format_ds_for_corda
from data_utils import DataCollatorForPairedModelPreference

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--dataset_name_or_path', type=str, default=None)
    parser.add_argument('--instruction_prompt', type=str, default=None)
    parser.add_argument('--model', type=str, help="Model to fine-tuned", default=None)    
    parser.add_argument('--warmup_ratio', type=float, default=None)
    parser.add_argument('--use_custom_opt_params', action='store_true', default=False)
    parser.add_argument('--early_stopping', action='store_true', default=False)

    parser.add_argument('--peft_config', choices=['lora', 'adalora', 'boft', 'hra'], default=None)
    parser.add_argument('--init_lora_weights', choices=['gaussian', 'eva', 'olora', 'pissa', 'corda', 'loftq', 'orthogonal', True], default=True)
    parser.add_argument('--training_args', type=str, default=None)
    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    set_seed(42)

    if args.config_file is not None:
        with open(args.config_file, "r") as file:
            config = yaml.safe_load(file)
        config_args = SimpleNamespace(**config)

        for k, v in vars(args).items():
            if k not in vars(config_args).keys():
                setattr(config_args, k, v)
        args = config_args

    tparser = HfArgumentParser(TEvalTrainingArguments)
    targs = tparser.parse_yaml_file(yaml_file=args.training_args)[0]
    output_dir = f"{targs.output_dir}_lr_{targs.learning_rate}_ep_{targs.num_train_epochs}_wd_{targs.weight_decay}"
    use_peft = args.peft_config is not None

    if use_peft:
        output_dir += f"_peft_{args.peft_config}"
        if "lora" in args.peft_config and isinstance(args.init_lora_weights, str):
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
            'early_stopping': args.early_stopping,
            'peft': {
                'peft_type': args.peft_config,
                'init_lora_weights': args.init_lora_weights
            }
        },
        reinit="finish_previous",
        settings=wandb.Settings(init_timeout=120)
    )

    data_seed = torch.manual_seed(42)

    base_model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_eos_token=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset_dir = args.dataset_name_or_path

    data_files = os.listdir(dataset_dir)
    dataset = {}
    for data_file in data_files:
        if not "_" in data_file and not "test" in data_file:
            with open(f"{dataset_dir}/{data_file}", 'r') as file:
                data_split = data_file.split('/')[-1].split('.')[0].split('_')[0]
                data = []
                for line in file.readlines():
                    data.append(json.loads(line))
                dataset[data_split] = Dataset.from_list(data)

    dataset = DatasetDict(dataset)
    dataset = format_dataset(dataset, high_translationese_key="messages_foreignization", low_translationese_key="messages_domestication")
    train_set = dataset['train']
    dev_set = dataset['dev']

    data_collator = DataCollatorForPairedModelPreference(pad_token_id=tokenizer.pad_token_id, padding_side="right")

    if args.init_lora_weights == 'corda':
        corda_config = CordaConfig(corda_method='ipm')
    else:
        corda_config = None

    if args.init_lora_weights == 'eva':
        eva_config = EvaConfig(
            rho=2.0
        )
    else:
        eva_config = None

    target_modules = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ]

    modules_to_save = ['embed_tokens', 'lm_head']

    ensure_weight_tying = getattr(base_model.config.get_text_config(), "tie_word_embeddings", False)

    steps_per_epoch = math.ceil(train_set.num_rows / (targs.per_device_train_batch_size * targs.gradient_accumulation_steps))
    total_steps = steps_per_epoch * targs.num_train_epochs
    warmup_steps = targs.warmup_steps if isinstance(targs.warmup_steps, int) else math.ceil(total_steps * targs.warmup_steps)

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
    
    if use_peft:
        if args.peft_config.lower() == 'lora':
            peft_config = LoraConfig(
                r=16,
                lora_alpha=64,
                lora_dropout=0.01,
                init_lora_weights=args.init_lora_weights,
                target_modules=target_modules,
                modules_to_save=modules_to_save,
                ensure_weight_tying=ensure_weight_tying,
                use_rslora=True,
                task_type='CAUSAL_LM',
                corda_config=corda_config,
                eva_config=eva_config
            )
            if args.init_lora_weights == 'corda':
                preprocess_corda(base_model, peft_config, run_model)
        elif args.peft_config.lower() == 'adalora':
            peft_config = AdaLoraConfig(
                r=16,
                lora_alpha=64,
                lora_dropout=0.01,
                target_modules=target_modules,
                modules_to_save=modules_to_save,
                init_lora_weights=args.init_lora_weights if args.init_lora_weights else True,
                ensure_weight_tying=ensure_weight_tying,
                use_rslora=True,
                task_type='CAUSAL_LM',
                corda_config=corda_config,
                eva_config=eva_config,
                target_r=8,
                init_r=32,
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

    model = TranslationeseEval(ht_model, lt_model, tokenizer)

    if args.use_custom_opt_params:
        optimizer_cls_and_kwargs = (optim.AdamW, {"params": list(model.high_translationese_model.named_parameters()) + list(model.low_translationese_model.named_parameters())})
    else:
        optimizer_cls_and_kwargs = None

    if args.early_stopping:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.0001
        )
        callbacks = [early_stopping]
    else:
        callbacks = None

    trainer = TEvalTrainer(
        model=model,
        args=targs,
        data_collator=data_collator,
        train_dataset=train_set,
        eval_dataset=dev_set,
        processing_class=model.tokenizer,
        optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
        callbacks=callbacks
    )

    with run:
        trainer.train()
    run.finish()

    os.makedirs(f"{targs.output_dir}/final_checkpoint/", exist_ok=True)
    os.makedirs(f"{targs.output_dir}/final/", exist_ok=True)

    save_model(trainer.model, f"{targs.output_dir}/final_checkpoint/model.safetensors", metadata={"format": "pt"})
    model.tokenizer.save_pretrained(f"{targs.output_dir}/final_checkpoint")

    model.high_translationese_model.save_pretrained(f"{targs.output_dir}/final_checkpoint/high_translationese_model", safe_serialization=True)
    model.low_translationese_model.save_pretrained(f"{targs.output_dir}/final_checkpoint/low_translationese_model", safe_serialization=True)

    if isinstance(model.high_translationese_model, (PeftModel, PeftMixedModel)):
        try:
            model.high_translationese_model.merge_and_unload()
            model.high_translationese_model.save_pretrained(f"{targs.output_dir}/merged_final/high_translationese_model", safe_serialization=True)
        except:
            pass    
    if isinstance(model.low_translationese_model, (PeftModel, PeftMixedModel)):
        try:
            model.low_translationese_model.merge_and_unload()
            model.low_translationese_model.save_pretrained(f"{targs.output_dir}/merged_final/low_translationese_model", safe_serialization=True)
        except:
            pass
        save_model(trainer.model, f"{targs.output_dir}/merged_final/model.safetensors", metadata={"format": "pt"})
        model.tokenizer.save_pretrained(f"{targs.output_dir}/merged_final", safe_serialization=True)
    
    torch.save(model.state_dict(), f"{targs.output_dir}/model.pt")