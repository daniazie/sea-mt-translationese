model=$1
config=$2
model_dir="parallel_asian_treebank_qwen"

uv run t_index/src/t_index.py \
    --config t_index/recipes/wild.yaml \
    --data_path /local_datasets/t-index_data/wild/enms/pointwise.jsonl \
    --model_positive models/sft/${model}-${model_dir}-6000-10/positive \
    --model_negative models/sft/${model}-${model_dir}-6000-10/negative \
    --prompt_field source \
    --completion_field translation \
    --output_file results/wild_t_index_results.jsonl
uv run t_index/src/supervised.py \
    --config t_index/recipes/wild.yaml \
    --model_path models/rm/${model}-${model_dir}-6000-10 \
    --output_file results/wild_rm_results.jsonl \
    --model_type rm
uv run t_index/src/supervised.py \
    --config t_index/recipes/wild.yaml \
    --model_path models/dpo/${model}-${model_dir}-6000-10 \
    --output_file results/wild_dpo_results.jsonl \
    --model_type dpo
