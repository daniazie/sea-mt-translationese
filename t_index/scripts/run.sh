model=$1
config=$2
data=$3

uv run python src/main.py --config recipes/${config}.yaml --model ${DATA_DIR}/sea-mt/models/t-index/dpo/${model}-pos-parallel_asian_treebank_qwen-6000-10 --result_file results/${model}/t_eval_dpo_pos_${data}_results.jsonl
uv run python src/main.py --config recipes/${config}.yaml --model ${DATA_DIR}/sea-mt/models/t-index/dpo/${model}-neg-parallel_asian_treebank_qwen-6000-10 --result_file results/${model}/t_eval_dpo_neg_${data}_results.jsonl

uv run python src/main.py --config recipes/${config}.yaml --model ${DATA_DIR}/sea-mt/models/t-index/rm/${model}-parallel_asian_treebank_qwen-6000-10 --result_file results/${model}/t_eval_rm_${data}_results.jsonl
uv run python src/main.py --config recipes/${config}.yaml --model ${DATA_DIR}/sea-mt/models/t-index/cpo/${model}-parallel_asian_treebank_qwen-6000-10 --result_file results/${model}/t_eval_cpo_${data}_results.jsonl
uv run python src/main.py --config recipes/${config}.yaml --model ${DATA_DIR}/sea-mt/models/t-index/sft/${model}-parallel_asian_treebank_qwen-6000-10/positive --result_file results/${model}/t_eval_sft_pos_${data}_results.jsonl
uv run python src/main.py --config recipes/${config}.yaml --model ${DATA_DIR}/sea-mt/models/t-index/sft/${model}-parallel_asian_treebank_qwen-6000-10/negative --result_file results/${model}/t_eval_sft_neg_${data}_results.jsonl