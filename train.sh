#!/bin/bash


python -u Solarclip_train.py \
    --config_dir ./configs/args1.json \
    2>&1 | tee ./log/train1.out

# python -u Solarclip_train.py \
#     --config_dir ./configs/args2.json \
#     2>&1 | tee ./log/train2.out

# python -u Solarclip_train.py \
#     --config_dir ./configs/args3.json \
#     2>&1 | tee ./log/train3.out

# python -u Solarclip_train.py \
#     --config_dir ./configs/args4.json \
#     2>&1 | tee ./log/train4.out
