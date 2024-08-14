#!/bin/bash
ulimit -v 514572800

python -u Solarpretrain_tq.py \
    --config_dir "./configs/pretrain/args1.json" \
    2>&1 | tee ./log/pretrain/pretrain1.out

python -u Solarpretrain_tq.py \
    --config_dir "./configs/pretrain/args2.json" \
    2>&1 | tee ./log/pretrain/pretrain2.out

python -u Solarpretrain_tq.py \
    --config_dir "./configs/pretrain/args3.json" \
    2>&1 | tee ./log/pretrain/pretrain3.out

python -u Solarpretrain_tq.py \
    --config_dir "./configs/pretrain/args4.json" \
    2>&1 | tee ./log/pretrain/pretrain4.out