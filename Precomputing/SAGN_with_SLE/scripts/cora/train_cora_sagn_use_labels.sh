cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../../src/sagn.py \
    --dataset cora \
    --gpu $gpu \
    --aggr-gpu $gpu \
    --model sagn \
    --acc-loss acc \
    --threshold 0.9 \
    --eval-every 1 \
    --epoch-setting 1000 1000 1000 \
    --lr 0.001 \
    --batch-size 50 \
    --num-hidden 64 \
    --dropout 0.5 \
    --attn-drop 0. \
    --input-drop 0.\
    --K 9 \
    --label-K 9 \
    --use-labels \
    --mlp-layer 2 \
    --use-norm \
    --weight-decay 5e-4 \
    --warmup-stage -1