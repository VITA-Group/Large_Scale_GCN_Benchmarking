#!/usr/bin/env bash
python main.py --type_model EnGCN --dataset ogbn-arxiv \
    --cuda_num 0 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --dropout 0.1 \
    --epochs 100 \
    --dim_hidden 512 \
    --num_layers 8 \
    --batch_size 10000 \
    --use_batch_norm False \
    --SLE_threshold 0.5 \
    --N_exp 10 \
    --tosparse

