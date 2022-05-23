#!/usr/bin/env bash

python main.py --type_model=AdaGCN_SLE --dataset=Reddit --num_layers=4 \
    --num_mlp_layers=3 \
    --use_batch_norm=True \
    --dropout=0.7 --lr=0.001 \
