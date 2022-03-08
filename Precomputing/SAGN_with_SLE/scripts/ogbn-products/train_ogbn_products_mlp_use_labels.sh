cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../../src/sagn.py \
    --dataset ogbn-products \
    --gpu $gpu \
    --aggr-gpu -1 \
    --model mlp \
    --seed 0 \
    --num-runs 10 \
    --threshold 0.9 \
    --epoch-setting 1000 500 500 \
    --lr 0.001 \
    --batch-size 50000 \
    --num-hidden 512 \
    --dropout 0.5 \
    --input-drop 0. \
    --K 9 \
    --mlp-layer 4 \
    --label-K 9 \
    --use-labels \
    --weight-decay 0 \
    --warmup-stage -1