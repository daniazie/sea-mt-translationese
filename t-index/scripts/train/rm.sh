model=$1
model_id=$2
train_data_dir=$3
seed=$4
max_samples=$5

deepspeed --module openrlhf.cli.train_rm \
   --max_len 1024 \
   --dataset ${DATA_DIR}/sea-mt/data/t-index_data/synthetic/enms/${train_data_dir}/train.jsonl \
   --chosen_key messages_foreignization \
   --rejected_key messages_domestication \
   --apply_chat_template \
   --train_batch_size 16 \
   --micro_train_batch_size 16 \
   --max_epochs 3 \
   --pretrain ${model_id} \
   --save_path ${DATA_DIR}/sea-mt/models/t-index/rm/${model}-${train_data_dir}-${max_samples}-${seed} \
   --save_steps -1 \
   --logging_steps 10 \
   --zero_stage 3 \
   --gradient_checkpointing \
   --max_samples ${max_samples} \
   --param_dtype bf16 \
   --attn_implementation flash_attention_2 \
   --use_tensorboard logs/rm/${model}-${train_data_dir}-${max_samples}-${seed} \
   --learning_rate 4e-6 \
   --l2 0.05 \
   --lr_warmup_ratio 0.1 \
   --seed ${seed} \
   --adam_offload \
   --learning_rate 4e-6 \
   --packing_samples 

