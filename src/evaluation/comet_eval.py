from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint
import torch
import argparse
import numpy as np
import json
import gc
import os

from utils import clean_results

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)
model_path = download_model('Unbabel/wmt22-cometkiwi-da')
cometkiwi_model = load_from_checkpoint(model_path)
model_path = download_model('Unbabel/XCOMET-XL')
xcomet_model = load_from_checkpoint(model_path)

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_dir', type=str, help="Path to result file(s).", default=None)
    parser.add_argument('--output_file', type=str, default=None)
    return parser


def comet(data):
    print("Calculating Comet")

    scores = comet_model.predict(data)
    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')    
    return scores.system_score


def xcomet(data):
    print("Calculating XCOMET")
    scores = xcomet_model.predict(data)
    torch.cuda.empty_cache()
    gc.collect()  
    print('Done!')
    return scores.system_score


def cometkiwi(data):
    print("Calculating CometKiwi")
    data = list(map(lambda x: {'src': x['src'].strip(), 'mt': x['mt'].strip()}, data))
    
    scores = cometkiwi_model.predict(data)
    torch.cuda.empty_cache()
    gc.collect()  
    print('Done!')
    return scores.system_score


parser = init_parser()
args = parser.parse_args()

print(torch.cuda.is_available())

def run_evaluation(data):
    print(data[:3])
    data = clean_results(data)
    comet_score = comet(data)
    xcomet_score = xcomet(data)
    cometkiwi_score = cometkiwi(data)

    final = {
        'Comet': comet_score * 100,
        'XCOMET': xcomet_score * 100,
        'CometKiwi': cometkiwi_score * 100,
    }

    print(final)

    return final

def process_jsonl(data_file):
    processed = []
    with open(data_file, 'r') as file:
        data = []
        for line in file.readlines():
            data.append(json.loads(line))

    for item in data:
        processed.append({
            "src": item['source'].strip(),
            "mt": item['foreignization'].strip(),
            "ref": item['domestication'].strip()
        })

    return processed

if os.path.isdir(args.data_path):
    data_files = os.listdir(args.data_path)
    for data_file in data_files:
        if data_file.endswith("jsonl"):
            data = process_jsonl(f'{args.data_path}/{data_file}')
        else:
            with open(f'{args.data_path}/{data_file}', 'r') as file:
                data = json.load(file)

        results = run_evaluation(data)

        if not "evaluation/scores" in args.output_dir:
            output_path = f'evaluation/scores/{args.output_dir}'
        else:
            output_path = args.output_dir
        file_name = data_file.split('/')[-1].replace('.json', '_scores.json')
        os.makedirs(output_path, exist_ok=True)
        with open(f"{output_path}/{file_name}", 'w') as file:
            json.dump(results, file, indent=2)
else:
    if args.data_path.endswith("jsonl"):
        data = process_jsonl(args.data_path)
    else:
        with open(f"{args.data_path}", "r") as file:
            data = json.load(file)

    results = run_evaluation(data)
    
    file_extension = args.data_path.split('.')[-1]
    if args.output_dir:
        if args.output_file:
            if not args.output_file.endswith(file_extension):
                output_file = f'{args.output_file}.json'
            result_path = f"{args.output_dir}/{args.output_file}"
        else:
            output_file = args.data_path.split('/')[-1].replace(file_extension, '_scores.json')
            result_path = f"{args.output_dir}/{output_file}"
    elif args.output_file:
        output_path = "evaluation/scores"
        result_path = output_path + '/' + args.output_file
    else:
        assert ValueError("`args.output_dir` and `args.output_file` cannot both be None.")
    os.makedirs(output_path, exist_ok=True)
    with open(result_path, 'w') as file:
        json.dump(results, file, indent=2)
