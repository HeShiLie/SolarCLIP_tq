#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=3 --master_addr="10.200.48.105" --master_port=9000 Train_Solarclip_recon_DiTlike.py \
2>&1 | tee ./log/recon/DiTlike/trainAdamW1.out

