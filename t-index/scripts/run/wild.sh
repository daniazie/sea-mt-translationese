model=$1
config=$2
model_dir="parallel_asian_treebank_qwen"

# python src/t_index.py \
#     --config recipes/wild.yaml \
#     --data_path /local_datasets/t-index_data/wild/enms/pointwise.jsonl \
#     --model_positive models/sft/${model}-${model_dir}-6000-10/positive \
#     --model_negative models/sft/${model}-${model_dir}-6000-10/negative \
#     --prompt_field source \
#     --completion_field translation \
#     --output_file results/wild_t_index_results.jsonl
# python src/supervised.py \
#     --config recipes/wild.yaml \
#     --model_path models/rm/${model}-${model_dir}-6000-10 \
#     --output_file results/wild_rm_results.jsonl \
#     --model_type rm
# python src/supervised.py \
#     --config recipes/wild.yaml \
#     --model_path models/dpo/${model}-${model_dir}-6000-10 \
#     --output_file results/wild_dpo_results.jsonl \
#     --model_type dpo

uv run python src_reproduce/unsupervised.py \
    --config recipes/${config}.yaml \
    --model_positive ${DATA_DIR}/sea-mt/models/t-index/sft/${model}-${model_dir}-6000-10/positive \
    --model_negative ${DATA_DIR}/sea-mt/models/t-index/sft/${model}-${model_dir}-6000-10/negative \
    --output_file evaluation/t-index_results/${model}/synthetic_enms_unsupervised_valid_results.jsonl
uv run torchrun src_reproduce/supervised.py \
    --config recipes/${config}.yaml \
    --model_path ${DATA_DIR}/sea-mt/models/t-index/rm/${model}-${model_dir}-6000-10 \
    --output_file evaluation/t-index_results/${model}/synthetic_enms_supervised_rm_valid_results.jsonl \
    --model_type rm
uv run torchrun src_reproduce/supervised.py \
    --config recipes/${config}.yaml \
    --model_path ${DATA_DIR}/sea-mt/models/t-index/dpo/${model}-${model_dir}-6000-10 \
    --output_file evaluation/t-index_results/${model}/synthetic_enms_supervised_dpo_valid_results.jsonl \
    --model_type dpo