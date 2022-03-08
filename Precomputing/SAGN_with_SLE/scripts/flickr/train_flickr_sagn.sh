cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../../src/sagn.py \
    --gpu $gpu \
    --aggr-gpu $gpu\
    --model sagn \
    --dataset flickr \
    --zero-inits \
    --inductive \
    --eval-every 1 \
    --epoch-setting 50 50 50 \
    --threshold 0.5 \
    --lr 0.001 \
    --batch-size 256 \
    --num-hidden 512 \
    --dropout 0.7 \
    --attn-drop 0.0 \
    --input-drop 0.0 \
    --label-drop 0.7 \
    --label-K 2 \
    --use-labels \
    --use-norm \
    --K 2 \
    --weight-decay 3e-6 \
    --warmup-stage -1