model=$1
model_id=$2
train_data_dir=$3
seed=$4
max_samples=$5

uv run deepspeed --module openrlhf.cli.train_rm \
   --max_len 1024 \
   --dataset data/translationese/synthetic/enms/${train_data_dir}/train.jsonl \
   --chosen_key messages_foreignization \
   --rejected_key messages_domestication \
   --apply_chat_template \
   --train_batch_size 16 \
   --micro_train_batch_size 16 \
   --max_epochs 3 \
   --pretrain ${model_id} \
   --save_path t_index/models/rm/${model}-${train_data_dir} \
   --save_steps -1 \
   --logging_steps 10 \
   --zero_stage 2 \
   --gradient_checkpointing \
   --max_samples ${max_samples} \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --use_tensorboard logs/rm/${model}-${train_data_dir} \
   --learning_rate 4e-6 \
   --l2 0.05 \
   --lr_warmup_ratio 0.1 \
   --seed ${seed} \
   --learning_rate 4e-6 \

