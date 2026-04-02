export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

bash scripts/sft.sh aisingapore/Gemma-SEA-LION-v4-27B-IT $DATA_DIR/sea-mt/data/parallel_asian_treebank
bash scripts/gen.sh $DATA_DIR/sea-mt/models/aisingapore/Gemma-SEA-LION-v4-27B-IT_SFT_parallel_asian_treebank_lr_5e-05_ep_5_wd_0.01_dft_loss_r_64_alpha_256_dropout_0.0_messages/merged_final

cd t-index
bash scripts/run/synthetic.sh qwen3-4b
bash scripts/run/synthetic.sh qwen3-0.6b