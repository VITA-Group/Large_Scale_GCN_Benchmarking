#!/usr/bin/env bash
python main.py --type_model EnGCN --dataset ogbn-products --cuda_num 1 \
    --lr 0.001 \
    --weight_decay 0 \
    --dropout 0.6 \
    --epochs 50 \
    --dim_hidden 1024 \
    --num_layers 8 \
    --use_batch_norm True \
    --batch_size 10000 \
    --SLE_threshold 0.7 \
    --num_heads 3 \
    --N_exp 10 \
    --tosparse 2>&1 | tee logs/ogbn-products_EnGCN.log
