#!/bin/bash
ulimit -v 514572800

mkdir -p ./log/pretrain_v2/ && python -u Solarpretrain_tq_v2.py \
    2>&1 | tee ./log/pretrain_v2/pretrain.out

