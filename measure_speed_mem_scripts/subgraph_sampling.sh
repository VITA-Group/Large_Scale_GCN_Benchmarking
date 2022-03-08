bs=5000
# for dataset in Reddit; do
for dataset in Reddit Flickr; do
# for model in ClusterGCN GraphSAINT GraphSAGE FastGCN LADIES SGC SIGN SAGN GAMLP; do
for model in GraphSAINT; do
    if [ "$model" = "GraphSAINT" ]; then
        bs=1695
    fi
    python main.py --type_model ${model} --dataset ${dataset} --batch_size ${bs} --num_layers 2 --dropout 0.7 --dim_hidden 128 --cuda 1 --debug_mem_speed 2>&1 | tee ./mem_speed_logs/${dataset}_${model}_bs_${bs}.log
done
done