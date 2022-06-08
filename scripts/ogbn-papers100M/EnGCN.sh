#!/usr/bin/env bash
python main.py --cuda_num=2 --dataset=ogbn-papers100M --type_model=EnGCN --lr=0.001 --weight_decay=0 --dim_hidden=128 --num_layers=4 --epochs=5 --tosparse
