model=$1
config=$2
data=$3

train_data_dir="parallel_asian_treebank_translategemma"

uv run t_index/src/unsupervised.py \
    --config t_index/recipes/${config}.yaml \
    --model_positive t_index/models/ensemble/${model}_alt-tgemma_lr_1e-06_ep_10_wd_0.01/final_checkpoint/high_translationese_model \
    --model_negative t_index/models/ensemble/${model}_alt-tgemma_lr_1e-06_ep_10_wd_0.01/final_checkpoint/low_translationese_model \
    --output_file t_index/results/${model}/synthetic_enms_teval_unsupervised_${data}_results.jsonl

