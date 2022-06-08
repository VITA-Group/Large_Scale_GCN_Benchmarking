#!/usr/bin/env bash

python main.py --type_model=AdaGCN_SLE --dataset=ogbn-products --num_layers=9 \
    --num_mlp_layers=3 --use_batch_norm=True \
    --dropout=0.5 --SLE_threshold=0.9 \
    --use_label_mlp=True --batch_size=50000 \
    --dim_hidden=512 --lr=0.001
