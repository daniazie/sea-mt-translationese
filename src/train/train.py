from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, EarlyStoppingCallback, HfArgumentParser
from trl import SFTConfig, SFTTrainer, apply_chat_template
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModelForCausalLM
from datasets import load_dataset, concatenate_datasets

from data_utils import format_prompt_completion, format_messages, preprocess_dataset, get_data_collator
from utils import load_trainer, get_training_args_class ,get_lora_modules, compute_metrics, preprocess_logits_for_metrics
from evaluate import load

from types import SimpleNamespace
from functools import partial
from pathlib import Path
import numpy as np
import random
import math
import wandb
import argparse
import yaml
import torch
import sys
import gc
import os

torch.cuda.empty_cache()
random.seed(42)

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--model', type=str, help="Model to fine-tuned", default=None)
    parser.add_argument('--is_vl', action='store_true', default=None)
    parser.add_argument('--dataset_name_or_path', type=str, help='Dataset to be used', default=None)
    parser.add_argument('--prompt_type', type=str, help='Prompt language', default=None)
    parser.add_argument('--prompt_completion_format', action='store_true', default=None)
    parser.add_argument('--train_mode', choices=['sft', 'dpo'], default=None)
    parser.add_argument('--training_args', type=str, default=None)
    parser.add_argument('--early_stopping', action='store_true', default=None)

    parser.add_argument('--lora_r', type=int, default=None)
    parser.add_argument('--lora_alpha', type=int, default=None)
    parser.add_argument('--lora_dropout', type=float, default=None)
    parser.add_argument('--lora_target_modules', type=get_lora_modules, default=None)

    return parser

if __name__ == "__main__":
    parser = init_parser()
    args, train_args = parser.parse_known_args()
    if args.config_file:
        with open(args.config_file, "r") as file:
            config = yaml.safe_load(file)
        config_args = SimpleNamespace(**config)
        for k, v in vars(args).items():
            if k not in vars(config_args).keys():
                setattr(config_args, k, v)
        args = config_args
    hf_parser = HfArgumentParser(get_training_args_class(args.train_mode))

    if args.training_args:
        train_config_args = hf_parser.parse_yaml_file(args.training_args)[0]
        targs = train_config_args

    if train_args:
        train_args = hf_parser.parse_args(args=train_args)
        for k, v in vars(train_args).items():
            if v is not None:
                setattr(train_config_args, k, v)
        targs = train_config_args
    

    dataset_name = args.dataset_name_or_path.split('/')[-1]

    experiment_name = f"{args.train_mode.upper()}_{dataset_name}_lr_{targs.learning_rate}_ep_{targs.num_train_epochs}_wd_{targs.weight_decay}_r_{args.lora_r}_alpha_{args.lora_alpha}_dropout_{args.lora_dropout}"
    output_dir = targs.output_dir + args.model.split('/')[-1] + "-" + experiment_name
    run = wandb.init(
            project=f"SEAMT_{args.model.split('/')[-1]}",
            name=experiment_name,
            config = {
                'epochs': targs.num_train_epochs,
                'lr': targs.learning_rate,
                'weight_decay': targs.weight_decay,
                'completion_only_loss': targs.completion_only_loss,
                'lora': {
                    'r': args.lora_r,
                    'alpha': args.lora_alpha,
                    'dropout': args.lora_dropout,
                    'target_modules': args.lora_target_modules
                }
            },
            reinit="finish_previous"
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_method="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if targs.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    if args.is_vl:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            quantization_config=quantization_config,
            device_map='auto',
            dtype=torch.bfloat16 if targs.bf16 else torch.float16 if targs.fp16 else None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization_config,
            device_map='auto',
            dtype=None,
        )

    tokenizer = AutoProcessor.from_pretrained(
        args.model,
        add_eos_token=True,
        padding_side='left',
    )

    if Path(args.dataset_name_or_path).exists():
        data_files = {
            split: f"{args.dataset_name_or_path}/{split}.json"
            for split in ['train', 'dev', 'test']
        }
        dataset = load_dataset("json", data_files=data_files)
    else:
        dataset = load_dataset(args.dataset_name_or_path)
        
    formatting_func = partial(format_prompt_completion, tokenizer=tokenizer, is_vl=args.is_vl, use_messages_format=True)
    dataset = dataset.map(formatting_func, batched=True)
    
    # if not args.prompt_completion_format:
    #     apply_chat_template = partial(preprocess_dataset, tokenizer=tokenizer)
    #     dataset = dataset.map(apply_chat_template)
    train_set = dataset['train']
    valid_set = dataset['dev']

    os.makedirs(targs.output_dir, exist_ok=True)
    output_dir = f'{targs.output_dir}/{args.model}_{experiment_name}'
    targs.output_dir = output_dir

    compute_metrics = partial(compute_metrics, tokenizer=tokenizer)
    early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=10,
            early_stopping_threshold=0.001
        )

    model = prepare_model_for_kbit_training(model)
    data_collator = get_data_collator(args.train_mode, args.is_vl, processor=tokenizer, pad_token_id=tokenizer.pad_token_type_id, return_tensors='pt')
    
    peft_config = LoraConfig(
    r=args.lora_r, 
    lora_alpha=args.lora_alpha, 
    lora_dropout=args.lora_dropout,
    target_modules=args.lora_target_modules,
    task_type="CAUSAL_LM",
    use_rslora=True,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = load_trainer(
        trainer=args.train_mode,
        model=model,
        processing_class=tokenizer,
        train_dataset=train_set,
        eval_dataset=valid_set,
        training_args=targs,
        data_collator=data_collator,
        callbacks=[early_stopping_callback] if args.early_stopping else None,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    with run:
        trainer.train()
    run.finish()

    trainer.save_model(output_dir)

    # trainer.push_to_hub()
    trainer.model.save_pretrained(f'{output_dir}/final_checkpoint')
    tokenizer.save_pretrained(f'{output_dir}/final_checkpoint')

    try:
        model = model.merge_and_unload()
    except Exception as e:
        print(e)


    model.save_pretrained(f'{output_dir}/merged_final', safe_serialization=True)
    tokenizer.save_pretrained(f'{output_dir}/merged_final', safe_serialization=True)

    model.push_to_hub(repo_id=f"daniazie/{args.model.split('/')[-1]}-SFT-SEA")
    tokenizer.push_to_hub(repo_id=f"daniazie/{args.model.split('/')[-1]}-SFT-SEA")
    torch.cuda.empty_cache()
    gc.collect()