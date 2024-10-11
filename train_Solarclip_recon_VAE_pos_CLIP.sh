#!/bin/bash

mkdir -p ./log/recon/VAE_pos_CLIP

python -u Train_Solarclip_recon_VAE_pos_CLIP.py \
    --config_dir ./configs/recon/VAE_pos_CLIP/args1.json \
    2>&1 | tee ./log/recon/VAE_pos_CLIP/train_0094_0094.out
