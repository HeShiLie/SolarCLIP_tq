#!/bin/bash
ulimit -v 514572800

mkdir -p ./log/pretrain_v1/ && python -u Solarpretrain_tq_v1.py \
    2>&1 | tee ./log/pretrain_v1/pretrain1.out

