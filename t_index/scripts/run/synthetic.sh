model=$1
config=$2
data=$3

train_data_dir="parallel_asian_treebank"

uv run t_index/src_reproduce/unsupervised.py \
    --config t_index/recipes/${config}.yaml \
    --model_positive t_index/models/sft/${model}-${train_data_dir}/positive \
    --model_negative t_index/models/sft/${model}-${train_data_dir}/negative \
    --output_file t_index/results/${model}/synthetic_enms_unsupervised_${data}_results.jsonl
uv run t_index/src_reproduce/supervised.py \
    --config t_index/recipes/${config}.yaml \
    --model_path t_index/models/rm/${model}-${train_data_dir} \
    --output_file t_index/results/${model}/synthetic_enms_supervised_rm_${data}_results.jsonl \
    --model_type rm
uv run t_index/src_reproduce/supervised.py \
    --config t_index/recipes/${config}.yaml \
    --model_path t_index/models/dpo/${model}-${train_data_dir} \
    --output_file t_index/results/${model}/synthetic_enms_supervised_dpo_${data}_results.jsonl \
    --model_type dpo
