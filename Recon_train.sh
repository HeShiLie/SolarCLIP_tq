#!/bin/bash

python -u Solarclip_reconstruction.py \
    --encoder_config_dir ./configs/args1.json \
    --decoder_config_dir ./configs/decoder/args1.json \
    2>&1 | tee ./log/recon/train1.out