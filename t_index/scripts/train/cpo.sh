MODEL=$1
MODEL_ID=$2
DATASET=$3
IS_VL=$4

export PYTORCH_ALLOC_CONF="expandable_segments:True"
uv run torchrun --master_port 29400 --nproc_per_node 1 src/cpo.py \
    --model ${MODEL_ID} ${IS_VL} \
    --dataset_name_or_path /data/dania/sea-mt/data/t-index_data/synthetic/enms/${DATASET} \
    --bf16 \
    --output_dir /data/${USER}/sea-mt/models/t-index/cpo/${MODEL}-${DATASET}-6000-10 \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=16 \
    --eval_accumulation_step=1 \
    --eval_steps=5 \
    --warmup_ratio=0.1 \
    --evaluation_strategy=steps \
    --save_strategy=best \
    --num_train_epochs=3 \
    --weight_decay=0.05 \
    --learning_rate=1e-6 \
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