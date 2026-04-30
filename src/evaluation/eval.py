from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sacrebleu.metrics import BLEU
from bert_score import score as BERT
from comet import download_model, load_from_checkpoint
from rouge_score.rouge_scorer import RougeScorer
from pathlib import Path
import torch
import argparse
import numpy as np
import json
import gc
import os

from t_eval import TranslationeseEval
from utils import clean_results

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--tgt_lang', type=str, default='Malay')
    parser.add_argument('--output_dir', type=str, help="Path to result file(s).", default=None)
    parser.add_argument('--output_file', type=str, default=None)
    return parser

meteor = load('meteor')
spbleu = BLEU(tokenize='spBLEU-1K')
teval = TranslationeseEval(model="t_index/models/sft/qwen3-0.6b-parallel_asian_treebank-5382-10", apply_chat_template=True, batch_size=4)

def calc_rouge(data):
    print("Calculating ROUGE")
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_1, rouge_l, rouge_2 = [], [], []

    for item in data:
        scores = scorer.score(prediction=item['mt'].strip(), target=item['ref'].strip())

        rouge_1.append(scores['rouge1'][2])
        rouge_l.append(scores['rougeL'][2])
        rouge_2.append(scores['rouge2'][2])

    results = {
        'rouge_1': float(np.mean(rouge_1) * 100),
        'rouge_2': float(np.mean(rouge_2) * 100),
        'rouge_l': float(np.mean(rouge_l) * 100),
    }

    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')
    return results

def calc_bleu(data):
    print("Calculating BLEU")
    bleu_1, bleu_2, bleu_3, bleu_4 = [], [], [], []

    smoothie = SmoothingFunction().method4

    for item in data:
        target_text = [item['ref'].strip().split()]
        prediction = item['mt'].strip().split()

        bleu_1.append(sentence_bleu(references=target_text, hypothesis=prediction, weights=(1, 0, 0, 0), smoothing_function=smoothie))
        bleu_2.append(sentence_bleu(references=target_text, hypothesis=prediction, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
        bleu_3.append(sentence_bleu(references=target_text, hypothesis=prediction, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
        bleu_4.append(sentence_bleu(references=target_text, hypothesis=prediction, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

    results = {
        'bleu_1': np.mean(bleu_1) * 100,
        'bleu_2': np.mean(bleu_2) * 100,
        'bleu_3': np.mean(bleu_3) * 100,
        'bleu_4': np.mean(bleu_4) * 100,
        'bleu_avg': (np.mean(bleu_1) + np.mean(bleu_2) + np.mean(bleu_3) + np.mean(bleu_4))/4  * 100
    }
    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')
    return results


def calc_spbleu(data):
    print("Calculating spBLEU")
    mts = [item['mt'].strip() for item in data]
    refs = [[item['ref'].strip()] for item in data]
    scores = spbleu.corpus_score(mts, refs)
    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')
    return scores.score


def calc_bert(data):
    print("Calculating BERTScore")
    predictions = [item['mt'].strip() for item in data]
    target = [item['ref'].strip() for item in data]
    P, R, F1 = BERT(cands=predictions, refs=target, lang='ms', device='cuda:0')
    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')
    return F1.mean().item() * 100


def calc_meteor(data):
    print("Calculating METEOR")
    

    scores = []
    for item in data:
        score = meteor.compute(predictions=[item['mt'].strip()], references=[item['ref'].strip()])
        scores.append(score['meteor'])

    results = np.mean(scores) * 100
    torch.cuda.empty_cache()
    gc.collect()
    print('Done!')
    return results

def calc_teval(data):
    
    results = teval.compute(data, tgt_lang=args.tgt_lang)
    return results['mean_score']
    

parser = init_parser()
args = parser.parse_args()

print(torch.cuda.is_available())

def run_evaluation(data):
    data = clean_results(data)
    spbleu_score = calc_spbleu(data)
    meteor_score = calc_meteor(data)

    final = {
        'spBLEU': spbleu_score,
        'METEOR': meteor_score,
    }

    try:
        teval_score = calc_teval(data)
        final.update({"1 - T-index": teval_score * 100})
    except Exception as e:
        print(e)

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
        result_path = f"{output_path}/{file_name}"
        os.makedirs(output_path, exist_ok=True)
        if Path(result_path).exists():
            with open(result_path, "r") as file:
                res = json.load(file)

            res.update(results)
            results = res
        with open(result_path, 'w') as file:
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
    if Path(result_path).exists():
        with open(result_path, "r") as file:
            res = json.load(file)

        res.update(results)
        results = res
    with open(result_path, 'w') as file:
        json.dump(results, file, indent=2)
