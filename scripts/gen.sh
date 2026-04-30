uv run src/evaluation/gen.py --model aisingapore/Qwen-SEA-LION-v4-8B-VL --is_vl --data_path data/parallel_asian_treebank/mt/enms/test.json --output_dir evaluation/predictions/baseline --result_file qwen-sealion-8b-vl_ALT_predictions
uv run src/evaluation/gen.py --model aisingapore/Gemma-SEA-LION-v4-4B-VL --is_vl --data_path data/parallel_asian_treebank/mt/enms/test.json --output_dir evaluation/predictions/baseline --result_file gemma-sealion-4b-vl_ALT_predictions
uv run src/evaluation/gen.py --model google/translategemma-4b-it --data_path data/parallel_asian_treebank/mt/enms/test.json --output_dir evaluation/predictions/baseline --result_file translategemma-4b-it_ALT_predictions
uv run src/evaluation/gen.py --model aisingapore/Apertus-SEA-LION-v4-8B-IT --data_path data/parallel_asian_treebank/mt/enms/test.json --output_dir evaluation/predictions/baseline --result_file apertus-sealion-8b-it_ALT_predictions
uv run src/evaluation/gen.py --model sail/Sailor2-8B-Chat --data_path data/parallel_asian_treebank/mt/enms/test.json --output_dir evaluation/predictions/baseline --result_file sailor-8b-chat_ALT_predictions
uv run src/evaluation/gen.py --model Qwen/Qwen3.5-9B --data_path data/parallel_asian_treebank/mt/enms/test.json --output_dir evaluation/predictions/baseline --result_file qwen35-9b_ALT_predictions

uv run src/evaluation/gen.py --model aisingapore/Qwen-SEA-LION-v4-8B-VL --is_vl --data_path data/NTREX/NTREX-128 --output_dir evaluation/predictions/baseline --result_file qwen-sealion-8b-vl_ntrex-128_predictions
uv run src/evaluation/gen.py --model aisingapore/Gemma-SEA-LION-v4-4B-VL --is_vl --data_path data/NTREX/NTREX-128 --output_dir evaluation/predictions/baseline --result_file gemma-sealion-4b-vl_ntrex-128_predictions
uv run src/evaluation/gen.py --model google/translategemma-4b-it --data_path data/NTREX/NTREX-128 --output_dir evaluation/predictions/baseline --result_file translategemma-4b-it_ntrex-128_predictions
uv run src/evaluation/gen.py --model aisingapore/Apertus-SEA-LION-v4-8B-IT --data_path data/NTREX/NTREX-128 --output_dir evaluation/predictions/baseline --result_file apertus-sealion-8b-it_ntrex-128_predictions
uv run src/evaluation/gen.py --model sail/Sailor2-8B-Chat --data_path data/NTREX/NTREX-128 --output_dir evaluation/predictions/baseline --result_file sailor-8b-chat_ntrex-128_predictions
uv run src/evaluation/gen.py --model Qwen/Qwen3.5-9B --data_path data/NTREX/NTREX-128 --output_dir evaluation/predictions/baseline --result_file qwen35-9b_ntrex-128_predictions

# uv run src/evaluation/gen.py --model aisingapore/Qwen-SEA-LION-v4-32B-IT --is_vl --data_path data/parallel_asian_treebank/mt/enms/test.json --output_dir evaluation/predictions/_baseline --result_file qwen-sealion-32b-vl_ALT_predictions
# uv run src/evaluation/gen.py --model aisingapore/Gemma-SEA-LION-v4-27B-IT --is_vl --data_path data/parallel_asian_treebank/mt/enms/test.json --output_dir evaluation/predictions/_baseline --result_file gemma-sealion-27b-vl_ALT_predictions
# uv run src/evaluation/gen.py --model google/translategemma-27b-it --data_path data/parallel_asian_treebank/mt/enms/test.json --output_dir evaluation/predictions/_baseline --result_file translategemma-27b-it_ALT_predictions
# uv run src/evaluation/gen.py --model aisingapore/Apertus-SEA-LION-v4-8B-IT --data_path data/parallel_asian_treebank/mt/enms/test.json --output_dir evaluation/predictions/_baseline --result_file apertus-sealion-8b-it_ALT_predictions
# uv run src/evaluation/gen.py --model sail/Sailor2-20B-Chat --data_path data/parallel_asian_treebank/mt/enms/test.json --output_dir evaluation/predictions/_baseline --result_file sailor-20b_ALT_predictions
# uv run src/evaluation/gen.py --model Qwen/Qwen3.5-27B --data_path data/parallel_asian_treebank/mt/enms/test.json --output_dir evaluation/predictions/_baseline --result_file qwen35-27b_ALT_predictions

# uv run src/evaluation/gen.py --model aisingapore/Qwen-SEA-LION-v4-32B-IT --is_vl --data_path data/NTREX/NTREX-128 --output_dir evaluation/predictions/_baseline --result_file qwen-sealion-32b-vl_ntrex-128_predictions
# uv run src/evaluation/gen.py --model aisingapore/Gemma-SEA-LION-v4-27B-IT --is_vl --data_path data/NTREX/NTREX-128 --output_dir evaluation/predictions/_baseline --result_file gemma-sealion-27b-vl_ntrex-128_predictions
# uv run src/evaluation/gen.py --model google/translategemma-27b-it --data_path data/NTREX/NTREX-128 --output_dir evaluation/predictions/_baseline --result_file translategemma-27b-it_ntrex-128_predictions
# uv run src/evaluation/gen.py --model aisingapore/Apertus-SEA-LION-v4-8B-IT --data_path data/NTREX/NTREX-128 --output_dir evaluation/predictions/_baseline --result_file apertus-sealion-8b-it_ntrex-128_predictions
# uv run src/evaluation/gen.py --model sail/Sailor2-20B-Chat --data_path data/NTREX/NTREX-128 --output_dir evaluation/predictions/_baseline --result_file sailor-20b_ntrex-128_predictions
# uv run src/evaluation/gen.py --model Qwen/Qwen3.5-27B --data_path data/NTREX/NTREX-128 --output_dir evaluation/predictions/_baseline --result_file qwen35-27b_ntrex-128_predictions

bash scripts/run_metrics.sh