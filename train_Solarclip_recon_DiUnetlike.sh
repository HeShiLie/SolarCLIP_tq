#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=3 --master_addr="10.200.48.108" --master_port=3000 Train_Solarclip_recon_DiUnetlike.py \
2>&1 | tee ./log/recon/DiUnetlike/trainAdamW1.out

