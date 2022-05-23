# python main.py --type_model=SADAGCN --num_layers=4 --batch_size=5000 --dataset=Flickr --N_exp=5 --cuda_num=2
# python main.py --type_model=SADAGCN --num_layers=4 --batch_size=5000 --dataset=Reddit --N_exp=5 --cuda_num=2
python main.py --type_model=SADAGCN --num_layers=4 --batch_size=10000 --dataset=ogbn-products --N_exp=5 --cuda_num=1
python main.py --type_model=SADAGCN --num_layers=4 --batch_size=10000 --dataset=ogbn-papers100M --N_exp=5 --cuda_num=1
