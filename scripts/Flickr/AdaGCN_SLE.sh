#!/usr/bin/env bash

python main.py --type_model=AdaGCN_SLE --dataset=Flickr --num_layers=4 \
    --num_mlp_layers=2 --use_batch_norm=False \
    --dropout=0.7 --SLE_threshold=0.5 \
    --use_label_mlp=True
