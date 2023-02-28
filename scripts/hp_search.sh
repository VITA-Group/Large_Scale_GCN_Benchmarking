dataset=$1
gpu=$2
mkdir -p logs
python scripts/$dataset/HP_search_${dataset}.py $gpu 2>&1 | tee logs/HP_search_${dataset}.log
