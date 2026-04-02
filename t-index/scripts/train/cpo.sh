MODEL=$1
DATASET=$2
IS_VL=$3

export PYTORCH_ALLOC_CONF="expandable_segments:True"
uv run torchrun --nproc_per_node 1 src/cpo.py \
    --model ${MODEL} ${IS_VL} \
    --dataset_name_or_path ${DATASET} \
    --bf16 \
    --output_dir /data/${USER}/sea-mt/models \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=32 \
    --per_device_eval_batch_size=4 \
    --eval_accumulation_step=32 \
    --eval_steps=5 \
    --warmup_ratio=0.1 \
    --evaluation_strategy=steps \
    --save_strategy=best \
    --num_train_epochs=5 \
    --weight_decay=0.01 \
    --learning_rate=5e-5 \
    --lr_scheduler=cosine \
    --max_seq_length=1024 \
    --logging_steps=5 \
    --save_steps=5 \
    --save_total_limit 2 \
    --report_to=wandb \
    --loss_type=simpo \
    --completion_only_loss \
    --lora_r=64 \
    --lora_alpha=256 \
    --lora_dropout=0.0 \
    --lora_target_modules=q_proj,k_proj,v_proj,o_proj \
    --lora_bias=none