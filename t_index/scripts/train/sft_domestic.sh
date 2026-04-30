model=$1
model_id=$2
train_data_dir=$3
seed=$4
max_samples=$5
learning_rate=$6
epoch=$7
batch_size=$8

uv run deepspeed --module openrlhf.cli.train_sft \
   --max_len 1024 \
   --dataset data/translationese/synthetic/enms/${train_data_dir}/train.jsonl \
   --input_key messages_domestication \
   --apply_chat_template \
   --train_batch_size ${batch_size} \
   --micro_train_batch_size 16 \
   --max_epochs ${epoch} \
   --pretrain ${model_id} \
   --save_path t_index/models/sft/${model}-${train_data_dir}/negative \
   --save_steps -1 \
   --logging_steps 10 \
   --zero_stage 2 \
   --max_samples ${max_samples} \
   --param_dtype bf16 \
   --use_tensorboard logs/sft/${model}-${train_data_dir}/negative \
   --learning_rate ${learning_rate} \
   --l2 0.05 \
   --lr_warmup_ratio 0.1 \
   --seed ${seed} \
   --attn_implementation flash_attention_2 \
   --gradient_checkpointing \
   --full_determinism
