#!/bin/bash
mkdir -p ./log/recon/TransformerDecoder

python -u Train_Solarclip_recon_TransformerDecoder.py \
--batch_size 64 \
--device cuda:2 \
--transformer_layers 6 \
--checkpoint_path /mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/recon/TransformerDecoder/trans_layer_6 \
2>&1 | tee ./log/recon/TransformerDecoder/trans_layer_6_trainAdamW1.out

