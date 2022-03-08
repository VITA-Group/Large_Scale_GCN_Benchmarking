# NOTE: given the same batch size 5000, the actual memory usage of laides is twice as much as fastgcn
python main.py --type_model LADIES --dataset Reddit --sample_coverage 0 --batch_size 5000 --num_layers 2 --dropout 0.7 --dim_hidden 128 --cuda 1 --debug_mem_speed
