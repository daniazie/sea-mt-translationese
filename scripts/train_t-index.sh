model=$1
model_id=$2

bash scripts/train/sft_foreign.sh ${model} ${model_id} parallel_asian_treebank_qwen 10 6000 1e-5 3 16
bash scripts/train/sft_domestic.sh ${model} ${model_id} parallel_asian_treebank_qwen 10 6000 1e-5 3 16
bash scripts/train/dpo.sh ${model} ${model_id} parallel_asian_treebank_qwen 10 6000 16
bash scripts/train/rm.sh ${model} ${model_id} parallel_asian_treebank_qwen 10 6000

bash scripts/run/synthetic.sh ${model}
