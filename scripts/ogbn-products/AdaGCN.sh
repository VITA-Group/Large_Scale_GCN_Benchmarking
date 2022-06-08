# python main.py --dataset=ogbn-products --type_model=AdaGCN_CandS \
#     --num_layers=1 --epochs=300 --lr=0.01 --dim_hidden=256 \
#     --num_mlp_layers=3 --dropout=0.5 --N_exp=3 \
#     --LP__num_propagations1=50 --LP__num_propagations2=50 \
#     --LP__alpha1=1.0 --LP__alpha2=0.8 # test_acc 83.98%
#

python main.py --dataset=ogbn-products --type_model=AdaGCN \
    --num_layers=4 --epochs=100 --lr=0.01 --dim_hidden=256 \
    --num_mlp_layers=3 --dropout=0.5 --N_exp=3 \
    --LP__num_propagations1=50 --LP__num_propagations2=50 \
    --LP__alpha1=1.0 --LP__alpha2=0.8
