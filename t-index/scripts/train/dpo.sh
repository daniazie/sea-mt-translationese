model=$1
model_id=$2
train_data_dir=$3
seed=$4
max_samples=$5
micro_train_batch_size=$6

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
deepspeed --module openrlhf.cli.train_dpo \
   --max_len 1024 \
   --dataset ${DATA_DIR}/sea-mt/data/t-index_data/synthetic/enms/${train_data_dir}/train.jsonl \
   --chosen_key messages_foreignization \
   --rejected_key messages_domestication \
   --apply_chat_template \
   --train_batch_size 16 \
   --micro_train_batch_size ${micro_train_batch_size} \
   --max_epochs 3 \
   --pretrain ${model_id} \
   --save_path ${DATA_DIR}/sea-mt/models/t-index/dpo/${model}-${train_data_dir}-${max_samples}-${seed} \
   --save_steps -1 \
   --logging_steps 10 \
   --gradient_checkpointing \
   --use_liger_kernel \
   --zero_stage 3 \
   --max_samples ${max_samples} \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --use_tensorboard logs/dpo/${model}-${train_data_dir}-${max_samples}-${seed} \
   --learning_rate 4e-6 \
   --l2 0.05 \
   --lr_warmup_ratio 0.1 \
   --seed ${seed} \
   --adam_offload \
   --full_determinism \
   --packing_samples 
