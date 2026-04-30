uv run src/train/train.py \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --config_file configs/train.yaml \
    --train_mode sft


