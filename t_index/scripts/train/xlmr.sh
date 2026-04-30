train_data_dir=$1
seed=$2

accelerate launch --config_file accelerate_config.yaml src/run_classification.py \
    --model_name_or_path FacebookAI/xlm-roberta-large \
    --max_seq_length 512 \
    --shuffle_seed ${seed} \
    --train_file ${DATA_DIR}/${train_data_dir}/train.jsonl \
    --validation_file ${DATA_DIR}/${train_data_dir}/valid.jsonl \
    --output_dir models/clf/xlm-roberta-large-${train_data_dir}-${seed} \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --logging_dir logs/clf/xlm-roberta-large-${train_data_dir}-${seed} \
    --logging_steps 1 \
    --save_strategy no \
    --seed ${seed} \
    --bf16 \
    --eval_steps 50 \
    --report_to tensorboard

