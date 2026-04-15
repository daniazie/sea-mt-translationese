from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, AutoTokenizer
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
    key: "en_US" if "eng" in v.lower() else "ms_MY"
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
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--is_vl', action='store_true')
    parser.add_argument('--src_lang', type=str, default='en')
    parser.add_argument('--tgt_lang', type=str, default='ms')
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--num_beams', type=int, default=1)

    parser.set_defaults(is_vl=False)
    return parser

def format_dataset(examples, src_lang, tgt_lang):
    system_prompt = f"""Translate the given text from {lang_code_to_name[src_lang]} to {lang_code_to_name[tgt_lang]}."""
    
    user_prompts = [f"""{lang_code_to_name[src_lang]}: {example}
### {lang_code_to_name[tgt_lang]}: 
""".strip() for example in examples['source']
]
    if 'gemma' in args.model.lower():
        user_prompts = [system_prompt + prompt for prompt in user_prompts]
        
    if args.is_vl:
        system_prompt = asdict(PromptContent(text=system_prompt))
    messages = []
    for prompt in user_prompts:
        user_prompt = asdict(PromptContent(text=prompt))
        if not args.is_vl:
            user_prompt = user_prompt['text']
        if not 'gemma' in args.model.lower():
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        else:
            message = [
                {"role": "user", "content": user_prompt}
            ]
        messages.append(message)
    return {"messages": messages}

def format_dataset_for_translategemma(examples, src_lang, tgt_lang):
    srcs = [example for example in examples['source']]
    messages = []
    for src in srcs:
        content = TranslateGemmaPromptContent(text=src, source_lang_code=lang_code_for_translate_gemma[src_lang], target_lang_code=lang_code_for_translate_gemma[tgt_lang])
        message = [
            {"role": "user", "content": asdict(content)}
        ]

def get_dataset() -> DatasetDict:
    data_files = os.listdir(args.dataset_dir)
    dataset = {}
    for data_file in data_files:
        with open(f"{args.dataset_dir}/{data_file}", 'r') as file:
            data_split = data_file.split('/')[-1].split('.')[0]
            data = []
            for line in file.readlines():
                data.append(json.loads(line))
            dataset[data_split] = Dataset.from_list(data)
    return DatasetDict(dataset)

def generate_data(split: str, dataset: DatasetDict):
    preds = []

    dataset: Dataset = dataset[split]
    dataset = dataset.batch(batch_size=16)
    for batch in tqdm(dataset, desc=f"Generating {split} data..."):
        inputs = tokenizer.apply_chat_template(
            batch['messages'],
            add_generation_prompt=True,
            tokenize=True,
            truncation=True,
            padding=True,
            return_dict=True,
            return_tensors='pt',
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, num_beams=args.num_beams)
        input_length = [len(input_ids) for input_ids in inputs['input_ids']]
        mts = tokenizer.batch_decode([output[input_length[i]:] for i, output in enumerate(outputs)], skip_special_tokens=True)
        for src, mt, ref in zip(batch['source'], mts, batch['domestication']):
            preds.append({
                "source": src,
                "foreignization": mt,
                "domestication": ref
            })

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f'{args.output_dir}/{split}_{args.num_beams}.jsonl', 'w') as file:
        for pred in preds:
            json.dump(pred, file)
            file.write('\n')

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_method="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', dtype=torch.bfloat16) if not args.is_vl else AutoModelForImageTextToText.from_pretrained(args.model, device_map='auto')
    tokenizer = AutoProcessor.from_pretrained(args.model, add_eos_token=True, padding_side='left')

    torch.cuda.empty_cache()
    gc.collect()

    dataset = get_dataset()
    formatting_func = partial(format_dataset if not 'translategemma' in args.model.lower() else format_dataset_for_translategemma, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    dataset = dataset.map(formatting_func, batched=True)

    generate = partial(generate_data, dataset=dataset)
    for split in dataset:
        generate(split)