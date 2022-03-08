cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../../src/sagn.py \
    --dataset ogbn-mag \
    --model sagn \
    --mag-emb \
    --aggr-gpu -1 \
    --gpu $gpu \
    --epoch-setting 200 200 200 \
    --lr 0.002 \
    --threshold 0.4 \
    --batch-size 50000 \
    --num-hidden 512 \
    --dropout 0.5 \
    --attn-drop 0. \
    --input-drop 0. \
    --K 5 \
    --label-K 3 \
    --use-labels \
    --weight-decay 0 \
    --warmup-stage 0