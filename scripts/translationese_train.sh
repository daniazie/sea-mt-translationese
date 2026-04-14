model=$1
model_id=$2

bash scripts/train/sft_foreign.sh ${model} ${model_id} parallel_asian_treebank_qwen 10 6000 1e-6 3 16
bash scripts/train/sft_domestic.sh ${model} ${model_id} parallel_asian_treebank_qwen 10 6000 1e-6 3 16
bash scripts/train/dpo.sh ${model} ${model_id} parallel_asian_treebank_qwen 10 6000 16
bash scripts/train/rm.sh ${model} ${model_id} parallel_asian_treebank_qwen 10 6000
bash scripts/train/cpo.sh qwen3-0.6b Qwen/Qwen3-0.6B parallel_asian_treebank_qwen

bash scripts/run/synthetic.sh ${model} synthetic_enms
bash scripts/run/synthetic.sh ${model} wild
bash scripts/run.sh ${model} t_eval_ntrex ntrex-128
bash scripts/run.sh ${model} t_eval_synth alt-test
