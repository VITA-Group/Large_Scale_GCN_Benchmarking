#!/usr/bin/env bash

python main.py --type_model AdaGCN --dataset Reddit --cuda_num 1 --lr 0.001 --weight_decay 0 --dropout 0.2 --epochs 70 --dim_hidden 512 --num_layers 4 --batch_size 5000 --use_batch_norm True --SLE_threshold 0.95 --N_exp 10
