#!/usr/bin/env bash
python main.py --type_model EnGCN --dataset ogbn-arxiv \
    --cuda_num 3 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --dropout 0.2 \
    --epochs 10 \
    --dim_hidden 256 \
    --num_layers 4 \
    --batch_size 5000 \
    --use_batch_norm False \
    --SLE_threshold 0.5 \
    --N_exp 10 \
    --tosparse
