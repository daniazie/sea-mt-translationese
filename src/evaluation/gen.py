from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, AutoTokenizer, ProcessorMixin, PreTrainedTokenizerBase
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from dataclasses import dataclass, asdict
from functools import partial

import torch
import argparse
import json
import os
import gc

@dataclass
class PromptContent:
    text: str
    type: str = "text"

@dataclass
class TranslateGemmaPromptContent:
    text: str
    source_lang_code: str
    target_lang_code: str
    type: str = "text"

torch.cuda.empty_cache()
gc.collect()

lang_name_to_code = {
    "english": ["en", "eng"],
    "malay": ["ms", "msa", "zsm"]
}

lang_code_to_name = {
    v: key.capitalize()
    for key, value in lang_name_to_code.items()
    for v in value
}

lang_code_for_translate_gemma = {
    key: "en" if "eng" in v.lower() else "ms"
    for key, v in lang_code_to_name.items()
}

lang_code_for_ntrex = {
    key: "eng" if "en" in v.lower() else "msa"
    for key, v in lang_code_to_name.items()
}

lang_ntrex_aliases = {
    "msa": ["ms", "zsm", "malay", "msa"],
    "eng": ["en", "english", "eng"]
}

lang_ntrex_aliases = {
    v: key
    for key, value in lang_ntrex_aliases.items()
    for v in value
}

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--is_vl', action='store_true')
    parser.add_argument('--src_lang', type=str, default='en')
    parser.add_argument('--tgt_lang', type=str, default='ms')
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--result_file', type=str, default=None)

    parser.set_defaults(is_vl=False)
    return parser

def system_prompt_supported(tokenizer):
    try: 
        _ = tokenizer.apply_chat_template([{"role": "system", "content": "Lorem ipsum"}])
        return True
    except:
        return False

def format_dataset_with_system_prompt(examples, src_lang, tgt_lang):
    system_prompt = f"""You are a translator. You are to translate a user-provided text from {lang_code_to_name[src_lang]} to {lang_code_to_name[tgt_lang]}. Return ONLY your translation, without commentary or explanation."""
    instruction = "{src_lang}: {src}\n\n{tgt_lang}:"

    user_prompts = [instruction.format(src_lang=lang_code_to_name[src_lang], tgt_lang=lang_code_to_name[tgt_lang], src=src) for src in examples['src']]

    if args.is_vl:
        system_prompt = [asdict(PromptContent(text=system_prompt))]
    messages = []
    for prompt in user_prompts:
        if args.is_vl:
            prompt = [asdict(PromptContent(text=prompt))]
        
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        messages.append(message)
    print(messages[:3])
    return {"messages": messages}

def format_dataset_user_prompt_only(examples, src_lang, tgt_lang):
    instruction = f"""Translate the following text from {lang_code_to_name[src_lang]} to {lang_code_to_name[tgt_lang]}. Return ONLY your translation, without commentary or explanation.\n\n"""
    prompts = [instruction + f"""{lang_code_to_name[src_lang]}: {example}\n\n{lang_code_to_name[tgt_lang]}:""".strip() for example in examples['src']]
    messages = []
    for prompt in prompts:
        content = [asdict(PromptContent(text=prompt))]
        messages.append([
            {"role": "user", "content": content}
        ])
    print(messages[:3])
    return {"messages": messages}

def format_dataset_for_translategemma(examples, src_lang, tgt_lang):
    srcs = [example for example in examples['src']]
    messages = []
    for src in srcs:
        content = TranslateGemmaPromptContent(text=src, source_lang_code=lang_code_for_translate_gemma[src_lang], target_lang_code=lang_code_for_translate_gemma[tgt_lang])
        message = [
            {"role": "user", "content": [asdict(content)]}
        ]
        messages.append(message)
    return {"messages": messages}

def get_dataset() -> Dataset:
    if "ntrex" in args.data_path.lower():
        data_file = "newstest2019-{type}.{lang}.txt"
        src_file = data_file.format(type='src', lang=lang_code_for_ntrex[args.src_lang])
        tgt_file = data_file.format(type='ref', lang=lang_code_for_ntrex[args.tgt_lang])
        with open(f"{args.data_path}/{src_file}", "r") as file:
            src_texts = []
            for line in file.readlines():
                src_texts.append(line)
        with open(f"{args.data_path}/{tgt_file}", "r") as file:
            tgt_texts = []
            for line in file.readlines():
                tgt_texts.append(line)
        data = {
            "src": src_texts,
            "ref": tgt_texts
        }
        return Dataset.from_dict(data)
    
    with open(f"{args.data_path}", 'r') as file:
        data = json.load(file)
    return Dataset.from_list(data)

def generate_data(dataset: Dataset):
    preds = []
    dataset = dataset.batch(batch_size=16)
    for batch in tqdm(dataset, desc=f"Predicting..."):
        if args.is_vl:
            inputs = tokenizer.apply_chat_template(
                batch['messages'],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors='pt',
                return_dict=True,
                processor_kwargs={
                    "enable_thinking": False,
                    "truncation": True,
                    "padding": True,
                }
            ).to(model.device)
        else:
            inputs = tokenizer.apply_chat_template(
                batch['messages'],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                truncation=True,
                padding=True,
                return_dict=True,
                return_tensors='pt',
            ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, num_beams=args.num_beams)
        input_length = [len(input_ids.flatten()) for input_ids in inputs['input_ids']]
        mts = tokenizer.batch_decode([output[input_length[i]:] for i, output in enumerate(outputs)], skip_special_tokens=True)
        for src, mt, ref in zip(batch['src'], mts, batch['ref']):
            preds.append({
                "src": src,
                "mt": mt,
                "ref": ref
            })

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f'{args.output_dir}/{args.result_file}.json', 'w') as file:
        json.dump(preds, file, indent=2)

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_method="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if args.is_vl or "gemma" in args.model.lower():
        model = AutoModelForImageTextToText.from_pretrained(args.model, device_map='auto', dtype=torch.bfloat16)
        tokenizer = AutoProcessor.from_pretrained(args.model, add_eos_token=True, padding_side='left')
        if tokenizer.tokenizer.pad_token_id is None:
            tokenizer.tokenizer.pad_token_id = tokenizer.tokenizer.eos_token_id
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(args.model, add_eos_token=True, padding_side='left')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    torch.cuda.empty_cache()
    gc.collect()

    dataset = get_dataset()

    if system_prompt_supported(tokenizer):
        format_dataset = format_dataset_with_system_prompt
    else:
        if "translategemma" in args.model.lower():
            format_dataset = format_dataset_for_translategemma
        else:
            format_dataset = format_dataset_user_prompt_only

    formatting_func = partial(format_dataset, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    dataset = dataset.map(formatting_func, batched=True)

    generate_data(dataset)