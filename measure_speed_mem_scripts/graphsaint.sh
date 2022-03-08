# num of sampled nodes per epoch: batch_size * walk_length
python main.py --type_model GraphSAINT --dataset Reddit --sample_coverage 0 --batch_size 2000 --num_layers 2 --dropout 0.7 --dim_hidden 128 --cuda 1 --debug_mem_speed
