model=$1
config=$2
data=$3
model_dir="parallel_asian_treebank_qwen"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2 

uv run python src_reproduce/unsupervised.py \
    --config recipes/${config}.yaml \
    --model_positive ${DATA_DIR}/sea-mt/models/t-index/dpo/${model}-pos-${model_dir}-6000-10 \
    --model_negative ${DATA_DIR}/sea-mt/models/t-index/dpo/${model}-neg-${model_dir}-6000-10 \
    --output_file results/${model}/synthetic_enms_unsupervised_${data}_results.jsonl
# uv run python src_reproduce/supervised.py \
#     --config recipes/${config}.yaml \
#     --model_path ${DATA_DIR}/sea-mt/models/t-index/rm/${model}-${model_dir}-6000-10 \
#     --output_file results/${model}/synthetic_enms_supervised_rm_${data}_results.jsonl \
#     --model_type rm
# uv run python src_reproduce/supervised.py \
#     --config recipes/${config}.yaml \
#     --model_path ${DATA_DIR}/sea-mt/models/t-index/dpo/${model}-${model_dir}-6000-10 \
#     --output_file results/${model}/synthetic_enms_supervised_dpo_${data}_results.jsonl \
#     --model_type dpo
# uv run python src_reproduce/supervised.py \
#     --config recipes/${config}.yaml \
#     --model_path ${DATA_DIR}/sea-mt/models/t-index/cpo/${model}-${model_dir}-6000-10 \
#     --output_file results/${model}/synthetic_enms_supervised_cpo_${data}_results.jsonl \
#     --model_type cpo
