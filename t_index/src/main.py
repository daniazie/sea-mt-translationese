from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
from types import SimpleNamespace
from openrlhf.models import get_llm_for_sequence_regression

from glob import glob
import argparse
import json
import yaml
import os

from ensemble_model import TranslationeseEval
from utils import format_messages

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir_or_path', type=str)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--prompt_key', type=str, default=None)
    parser.add_argument('--prompt_template', type=str, default=None)
    parser.add_argument('--positive_key', type=str, default=None)
    parser.add_argument('--negative_key', type=str, default=None)
    parser.add_argument('--completion_key', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--apply_chat_template', type=bool, default=False)
    parser.add_argument('--result_file', type=str)
    return parser

parser = init_parser()
args = parser.parse_args()

if args.config is not None:
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    config_args = SimpleNamespace(**config)
    for k, v in vars(args).items():
        if v is not None:
            setattr(config_args, k, v)
    args = vars(config_args)
    
print(args)
model_dir_or_path = args.pop('model_dir_or_path')
data_path = args.pop('data_path')
result_file = args.pop('result_file')
args.pop('config', None)
is_reward_model = False
if "rm" in model_dir_or_path:
    model = get_llm_for_sequence_regression(
        model_dir_or_path,
        model_type="reward",
        device_map='auto'
    )
    is_reward_model = True
else:
    model = AutoModelForCausalLM.from_pretrained(model_dir_or_path, dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_dir_or_path, padding_side='left')

data = []
for data_file in glob(data_path):
    with open(data_file, "r") as file:
        for line in file.readlines():
            line = json.loads(line)
            line.update({"file_path": data_file})
            data.append(line)

dataset = Dataset.from_list(data)



tindex_args = TIndexArgs(
    **args
)

teval = TranslationeseEval(
    model=model,
    tokenizer=tokenizer,
    is_reward_model=is_reward_model,
    args=tindex_args
)

results = teval(dataset)

results_dir = '/'.join(result_file.split('/')[:-1])
os.makedirs(results_dir, exist_ok=True)
with open(result_file, "w") as file:
    for result in results:
        json.dump(result, file)
        file.write('\n')