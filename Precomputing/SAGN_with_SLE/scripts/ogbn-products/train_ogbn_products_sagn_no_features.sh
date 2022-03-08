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
    --aggr-gpu $gpu \
    --model sagn \
    --seed 0 \
    --num-runs 10 \
    --epoch-setting 1000 200 200 \
    --lr 0.001 \
    --batch-size 50000 \
    --num-hidden 512 \
    --dropout 0.5 \
    --attn-drop 0.4 \
    --input-drop 0.2 \
    --label-K 9 \
    --avoid-features \
    --use-labels \
    --weight-decay 0 \
    --warmup-stage -1