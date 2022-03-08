cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../../src/sagn.py \
    --gpu $gpu \
    --aggr-gpu $gpu \
    --model sagn \
    --seed 0 \
    --dataset ppi_large \
    --inductive \
    --threshold 0.7 \
    --epoch-setting 500 500 500\
    --eval-every 10 \
    --lr 0.001 \
    --batch-size 1024 \
    --mlp-layer 2 \
    --num-hidden 2048 \
    --dropout 0.3 \
    --attn-drop 0.1 \
    --input-drop 0.1 \
    --K 5 \
    --label-K 9 \
    --use-labels \
    --weight-decay 5e-6 \
    --warmup-stage 0 \