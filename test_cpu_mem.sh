#!/usr/bin/env bash

for dataset in 'Flickr' 'Reddit'; do
# for i in 1 2 3 4 5 6 7 8; do
# python test_cpu_mem.py --dataset=$dataset --type_model=SIGN --epochs=6 --N_exp=1 --num_layers=$i
# done
# for i in 1 2 3 4 5 6 7 8; do
# python test_cpu_mem.py --dataset=$dataset --type_model=SAGN --epochs=6 --N_exp=1 --num_layers=$i
# done
# for i in 1 2 3 4 5 6 7 8 9; do
# python test_cpu_mem.py --dataset=$dataset --type_model=EnGCN --epochs=6 --N_exp=1 --num_layers=$i --tosparse
# done
for i in 1 2 3 4 5 6 7 8; do
python test_cpu_mem.py --dataset=$dataset --type_model=ClusterGCN --epochs=6 --N_exp=1 --num_layers=$i --cuda_num=3
done
for i in 2 3 4 5 6 7 8; do
python test_cpu_mem.py --dataset=$dataset --type_model=GraphSAGE --epochs=6 --N_exp=1 --num_layers=$i --cuda_num=3
done
done
