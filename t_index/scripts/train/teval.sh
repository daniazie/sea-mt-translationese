uv run t_index/src/train.py --config_file t_index/recipes/train.yaml
uv run t_index/src/train.py --config_file t_index/recipes/train.yaml --peft_config lora
uv run t_index/src/train.py --config_file t_index/recipes/train.yaml --peft_config lora --init_lora_weights eva
uv run t_index/src/train.py --config_file t_index/recipes/train.yaml --peft_config lora --init_lora_weights gaussian
uv run t_index/src/train.py --config_file t_index/recipes/train.yaml --peft_config lora --init_lora_weights olora
uv run t_index/src/train.py --config_file t_index/recipes/train.yaml --peft_config lora --init_lora_weights orthogonal
