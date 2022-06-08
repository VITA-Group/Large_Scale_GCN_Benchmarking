#!/usr/bin/env bash
python main.py --type_model EnGCN --dataset Flickr \
    --cuda_num 3 \
    --lr 0.0001 \
    --weight_decay 0.0001 \
    --dropout 0.2 \
    --epochs 70 \
    --dim_hidden 256 \
    --num_layers 4 \
    --batch_size 1000 \
    --use_batch_norm True \
    --SLE_threshold 0.9 \
    --N_exp 10 \
    --tosparse
