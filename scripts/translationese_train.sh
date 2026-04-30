model=$1
model_id=$2

# bash t_index/scripts/train/sft_foreign.sh ${model} ${model_id} parallel_asian_treebank 10 5382 1e-6 3 16
# bash t_index/scripts/train/sft_domestic.sh ${model} ${model_id} parallel_asian_treebank 10 5382 1e-6 3 16
# bash t_index/scripts/train/dpo.sh ${model} ${model_id} parallel_asian_treebank 10 5382 16
# bash t_index/scripts/train/rm.sh ${model} ${model_id} parallel_asian_treebank 10 5382

bash t_index/scripts/run/synthetic.sh ${model} synthetic_enms alt
