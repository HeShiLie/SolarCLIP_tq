#!/bin/bash
mkdir -p ./log/recon/LinearDecoder

python -u Train_Solarclip_recon_LinearDecoder.py \
2>&1 | tee ./log/recon/LinearDecoder/trainAdamW1.out

