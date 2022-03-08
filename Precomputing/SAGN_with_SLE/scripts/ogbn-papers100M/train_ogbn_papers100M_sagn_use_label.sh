cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../../src/sagn.py \
    --dataset ogbn-papers100M \
    --gpu $gpu \
    --aggr-gpu -1 \
    --eval-every 1 \
    --model sagn \
    --zero-inits \
    --chunks 1 \
    --memory-efficient \
    --load-embs \
    --load-label-emb \
    --seed 0 \
    --num-runs 10 \
    --threshold 0.5 \
    --epoch-setting 100 100 100 100\
    --lr 0.001 \
    --batch-size 5000 \
    --num-hidden 1024 \
    --dropout 0.5 \
    --attn-drop 0. \
    --input-drop 0.0 \
    --label-drop 0.5 \
    --K 3 \
    --label-K 9 \
    --use-labels \
    --weight-decay 0 \
    --warmup-stage -1