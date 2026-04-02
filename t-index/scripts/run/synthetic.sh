model=$1
model_dir="parallel_asian_treebank_qwen"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2 

uv run python src_reproduce/unsupervised.py \
    --config recipes/synthetic_enms.yaml \
    --model_positive ${DATA_DIR}/sea-mt/models/t-index/sft/${model}-${model_dir}-6000-10/positive \
    --model_negative ${DATA_DIR}/sea-mt/models/t-index/sft/${model}-${model_dir}-6000-10/negative \
    --output_file evaluation/t-index_results/${model}/synthetic_enms_unsupervised_valid_results.jsonl
uv run torchrun src_reproduce/supervised.py \
    --config recipes/synthetic_enms.yaml \
    --model_path ${DATA_DIR}/sea-mt/models/t-index/rm/${model}-${model_dir}-6000-10 \
    --output_file evaluation/t-index_results/${model}/synthetic_enms_supervised_rm_valid_results.jsonl \
    --model_type rm
uv run torchrun src_reproduce/supervised.py \
    --config recipes/synthetic_enms.yaml \
    --model_path ${DATA_DIR}/sea-mt/models/t-index/dpo/${model}-${model_dir}-6000-10 \
    --output_file evaluation/t-index_results/${model}/synthetic_enms_supervised_dpo_valid_results.jsonl \
    --model_type dpo