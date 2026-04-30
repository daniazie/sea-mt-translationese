# uv run src/evaluation/gen.py --model Qwen/Qwen3.5-27B --data_path data/parallel_asian_treebank/mt/enms/test.json --output_dir evaluation/predictions/_baseline --result_file qwen35-27b_ALT_predictions
# uv run src/evaluation/gen.py --model Qwen/Qwen3.5-27B --data_path data/NTREX/NTREX-128 --output_dir evaluation/predictions/_baseline --result_file qwen35-27b_ntrex-128_predictions

# uv run --with transformers==4.57.6 src/evaluation/comet_eval.py --data_path evaluation/predictions/_baseline --output_dir evaluation/scores/_baseline
# uv run src/evaluation/eval.py --data_path evaluation/predictions/_baseline --output_dir evaluation/scores/_baseline
uv run --with transformers==4.57.6 src/evaluation/comet_eval.py --data_path evaluation/predictions/baseline --output_dir evaluation/scores/baseline
uv run src/evaluation/eval.py --data_path evaluation/predictions/baseline --output_dir evaluation/scores/baseline
# ALT_DIR=data/translationese/synthetic/enms

# for dataset in $(ls ${ALT_DIR}); do
#     uv run --with transformers==4.47.0 src/evaluation/eval.py --data_path ${ALT_DIR}/${dataset}/test.jsonl --output_dir evaluation/scores/baseline
# done 