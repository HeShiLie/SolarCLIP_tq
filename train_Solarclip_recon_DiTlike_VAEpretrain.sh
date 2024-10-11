#!/bin/bash
# torchrun --nnodes=1 --nproc_per_node=3 --master_addr="10.200.48.105" --master_port=9000 Train_Solarclip_recon_DiTlike_VAEpretrain.py \
# 2>&1 | tee ./log/recon/DiTlike/VAEpretrain/trainAdamW1.out

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_addr="10.200.48.105" --master_port=9000 Train_Solarclip_recon_DiTlike_VAEpretrain.py \
--lambda_ratio 3e-1 2>&1 | tee ./log/recon/DiTlike/VAEpretrain/trainAdamW1_lambda_3e-1_.out &

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_addr="10.200.48.105" --master_port=9001 Train_Solarclip_recon_DiTlike_VAEpretrain.py \
--lambda_ratio 6e-1 2>&1 | tee ./log/recon/DiTlike/VAEpretrain/trainAdamW1_lambda_6e-1_.out &

CUDA_VISIBLE_DEVICES=2 torchrun --nnodes=1 --nproc_per_node=1 --master_addr="10.200.48.105" --master_port=9002 Train_Solarclip_recon_DiTlike_VAEpretrain.py \
--lambda_ratio 9e-1 2>&1 | tee ./log/recon/DiTlike/VAEpretrain/trainAdamW1_lambda_9e-1.out &
