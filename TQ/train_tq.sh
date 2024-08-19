#!/bin/bash


python -u Solarclip_train_tq.py \
    --config_dir ./configs/args1_tq.json \
    2>&1 | tee ./log/train1_tq.out

python -u Solarclip_train_tq.py \
    --config_dir ./configs/args2_tq.json \
    2>&1 | tee ./log/train2_tq.out

python -u Solarclip_train_tq.py \
    --config_dir ./configs/args3_tq.json \
    2>&1 | tee ./log/train3_tq.out

python -u Solarclip_train_tq.py \
    --config_dir ./configs/args4_tq.json \
    2>&1 | tee ./log/train4_tq.out
