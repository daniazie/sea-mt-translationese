MODEL_DIR=$1
MODEL_NAME=$2

export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=2 uv run torchrun src/evaluation/gen.py \
    --model ${MODEL_DIR} \
    --src_lang en \
    --tgt_lang ms \
    --output_file ${MODEL_NAME}_en_ms_ntrex-128_results