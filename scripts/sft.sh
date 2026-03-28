MODEL=$1
DATASET=$2
IS_VL=$3

export PYTORCH_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=2 uv run accelerate launch --use_deepspeed --config_file configs/accelerate_single_gpu_config.yaml --main_process_port 0 src/train/sft.py \
    --model ${MODEL} ${IS_VL} \
    --dataset_name_or_path ${DATASET} \
    --fp16 \
    --output_dir /data/${USER}/sea-mt/models \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=32 \
    --per_device_eval_batch_size=4 \
    --llm_int8_enable_fp32_cpu_offload \
    --eval_accumulation_step=32 \
    --eval_steps=5 \
    --warmup_ratio=0.1 \
    --evaluation_strategy='steps' \
    --num_train_epochs=5 \
    --weight_decay=0.1 \
    --learning_rate=2e-4 \
    --lr_scheduler='cosine' \
    --max_seq_length=2048 \
    --logging_steps=5 \
    --save_steps=5 \
    --save_total_limit 2 \
    --report_to="wandb" \
    --loss_type='dft' \
    --prompt_completion_format \
    --use_liger_kernel \
    --lora_r=64 \
    --lora_alpha=256 \
    --lora_dropout=0.0 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --lora_bias="none" \
    --activation_offloading
exit