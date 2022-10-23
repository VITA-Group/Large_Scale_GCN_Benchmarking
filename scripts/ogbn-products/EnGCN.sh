#!/usr/bin/env bash
python main.py --type_model EnGCN --dataset ogbn-products --cuda_num 1 --lr 0.01 --weight_decay 0 --dropout 0.2 --epochs 70 --dim_hidden 512 --num_layers 8 --use_batch_norm True --batch_size 10000 --SLE_threshold 0.8 --N_exp 10 --tosparse
