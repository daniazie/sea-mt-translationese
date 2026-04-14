from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, EarlyStoppingCallback
from trl.experimental.cpo import CPOTrainer, CPOConfig
from trl.data_utils import apply_chat_template
from trl.experimental.utils import DPODataCollatorWithPadding
from trl.trainer.dpo_trainer import DataCollatorForPreference
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModelForCausalLM
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict

from train_utils import get_lora_modules, format_prompt_completion, format_messages, compute_metrics, preprocess_dataset, preprocess_logits_for_metrics
from utils import format_messages_for_preference
from evaluate import load

from functools import partial
from pathlib import Path
import numpy as np
import random
import math
import wandb
import argparse
import torch
import json
import sys
import gc
import os

torch.cuda.empty_cache()
random.seed(42)

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Model to fine-tuned",)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--is_vl', action='store_true')
    parser.add_argument('--dataset_name_or_path', type=str, help='Dataset to be used')
    parser.add_argument('--prompt_type', type=str, help='Prompt language')
    parser.add_argument('--llm_int8_enable_fp32_cpu_offload', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--packing', action='store_true')
    parser.add_argument('--per_device_train_batch_size', type=int, default=5)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--prompt_completion_format', action='store_true', default=False)
    parser.add_argument('--eval_accumulation_steps', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--lr_scheduler', type=str, default='inverse_sqrt')
    parser.add_argument('--max_seq_length', type=int, default=200)
    parser.add_argument('--evaluation_strategy', type=str, default=None)
    parser.add_argument('--save_strategy', type=str, default='steps')
    parser.add_argument('--completion_only_loss', action='store_true', default=False)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=0)
    parser.add_argument('--save_steps', type=int, default=0)
    parser.add_argument('--save_total_limit', type=int, default=-1)
    parser.add_argument('--use_torch_compile', action='store_true', default=False)
    parser.add_argument('--use_liger_kernel', action='store_true', default=False)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--report_to', type=str, default='wandb')
    parser.add_argument('--loss_type', type=str, default='nll')
    parser.add_argument('--local_rank', type=int, default=32)
    parser.add_argument('--activation_offloading', action='store_true', default=False)

    parser.add_argument('--enable_peft', action='store_true', default=False)
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_bias', type=str, default='none')
    parser.add_argument('--lora_target_modules', type=get_lora_modules)

    parser.set_defaults(is_vl=False)
    parser.set_defaults(packing=False)
    parser.set_defaults(bf16=False)
    parser.set_defaults(fp16=False)
    return parser

def format_dataset(example, tokenizer):
    prompt = tokenizer.apply_chat_template(
        example['prompt'],
        tokenize=False,
        add_generation_prompt=True
    )

    chosen = tokenizer.apply_chat_template(
        example['chosen'],
        tokenize=False
    )

    rejected = tokenizer.apply_chat_template(
        example['rejected'],
        tokenize=False
    )

    return {
        "prompt": prompt,
        "chosen": chosen,
        'rejected': rejected
    }

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    dataset_name = args.dataset_name_or_path.split('/')[-1]
    dataset_type = 'prompt-completion' if args.prompt_completion_format else 'messages'

    experiment_name = f"SFT_{dataset_name}_lr_{args.learning_rate}_ep_{args.num_train_epochs}_wd_{args.weight_decay}_{args.loss_type}_loss_r_{args.lora_r}_alpha_{args.lora_alpha}_dropout_{args.lora_dropout}_{dataset_type}"
    
    run = wandb.init(
            project=f"SEAMT_{args.model.split('/')[-1]}",
            name=experiment_name,
            config = {
                'epochs': args.num_train_epochs,
                'lr': args.learning_rate,
                'weight_decay': args.weight_decay,
                'loss_type': args.loss_type,
                'completion_only_loss': args.completion_only_loss,
                'dataset_type': dataset_type,
                'lora': {
                    'r': args.lora_r,
                    'alpha': args.lora_alpha,
                    'dropout': args.lora_dropout,
                    'bias': args.lora_bias,
                    'target_modules': args.lora_target_modules
                }
            },
            reinit="finish_previous"
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_method="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=args.llm_int8_enable_fp32_cpu_offload
    ) if args.enable_peft else None

    if args.is_vl:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            quantization_config=quantization_config,
            device_map='auto',
            dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None,
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
            split: f"{args.dataset_name_or_path}/{split}.jsonl"
            for split in ['train', 'valid', 'test']
        }

        for split, data_file in data_files.items():
            data_split = []
            with open(data_file, "r") as file:
                for line in file.readlines():
                    data_split.append(json.loads(line))
            data_files[split] = Dataset.from_list(data_split)

        dataset = DatasetDict(data_files)
    else:
        dataset = load_dataset(args.dataset_name_or_path)
        
    formatting_func = partial(format_messages_for_preference, is_vl=args.is_vl, src_lang='English', tgt_lang='Malay')
    dataset = dataset.map(formatting_func, batched=True)
    apply_pref_chat_template = partial(format_dataset, tokenizer=tokenizer)
    dataset = dataset.map(apply_pref_chat_template)

    
    
    # if not args.prompt_completion_format:
    #     apply_chat_template = partial(preprocess_dataset, tokenizer=tokenizer)
    #     dataset = dataset.map(apply_chat_template)
    train_set = dataset['train']
    valid_set = dataset['valid']

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = f'{args.output_dir}/{args.model}_{experiment_name}'

    compute_metrics = partial(compute_metrics, tokenizer=tokenizer)
    early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=10,
            early_stopping_threshold=0.001
        )

    model = prepare_model_for_kbit_training(model)
    pad_token_id = tokenizer.pad_token_id if not args.is_vl else tokenizer.tokenizer.pad_token_id
    data_collator = DataCollatorForPreference(pad_token_id=pad_token_id)

    num_devices = torch.cuda.device_count()
    steps_per_epoch = math.ceil(train_set.num_rows / (args.per_device_train_batch_size * args.gradient_accumulation_steps * num_devices))
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    
    training_args = CPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        max_length=args.max_seq_length,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        prediction_loss_only=args.completion_only_loss,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        warmup_steps=warmup_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        max_grad_norm=1.0,
        optim='adamw_8bit',
        lr_scheduler_type=args.lr_scheduler,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        torch_compile=args.use_torch_compile,
        use_liger_kernel=args.use_liger_kernel,
        loss_type=args.loss_type,
        seed=10,
        data_seed=10,
        eval_on_start=True,
        report_to=args.report_to,
    )

    if args.enable_peft:
        peft_config = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
            use_rslora=True,
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    trainer = CPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_set,
        eval_dataset=valid_set,
        args=training_args,
        callbacks=[early_stopping_callback] if args.early_stopping else None,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    with run:
        trainer.train()
    run.finish()

    trainer.save_model(output_dir)

    # trainer.push_to_hub()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.enable_peft:
        try:
            model = model.merge_and_unload()
        except Exception as e:
            print(e)


        model.save_pretrained(f'{output_dir}/merged_final', safe_serialization=True)
        tokenizer.save_pretrained(f'{output_dir}/merged_final', safe_serialization=True)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir, safe_serialization=True)

    torch.cuda.empty_cache()
    gc.collect()