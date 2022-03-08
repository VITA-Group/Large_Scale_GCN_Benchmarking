cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../../src/sagn.py \
    --dataset ogbn-mag \
    --epoch-setting 200 200 200 \
    --model sagn \
    --gpu $gpu \
    --aggr-gpu -1 \
    --lr 0.002 \
    --threshold 0.4 \
    --batch-size 50000 \
    --num-hidden 512 \
    --dropout 0.3 \
    --attn-drop 0.2 \
    --input-drop 0. \
    --K 5 \
    --label-K 3 \
    --use-norm \
    --use-labels \
    --weight-decay 0 \
    --warmup-stage -1