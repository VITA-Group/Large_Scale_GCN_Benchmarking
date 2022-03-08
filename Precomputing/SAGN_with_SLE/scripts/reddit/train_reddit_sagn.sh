cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../../src/sagn.py \
    --dataset reddit \
    --gpu $gpu \
    --aggr-gpu $gpu \
    --model sagn \
    --inductive \
    --zero-inits \
    --threshold 0.9 0.9 \
    --eval-every 10 \
    --epoch-setting 1000 1000 1000 \
    --lr 0.001 \
    --batch-size 10000 \
    --num-hidden 512 \
    --num-heads 1 \
    --dropout 0.7 \
    --attn-drop 0.4 \
    --input-drop 0.0 \
    --label-drop 0.7 \
    --K 3 \
    --use-labels \
    --label-K 4 \
    --weight-decay 0 \
    --warmup-stage -1