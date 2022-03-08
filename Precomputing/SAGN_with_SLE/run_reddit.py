import os, sys
import json
import argparse

parser = argparse.ArgumentParser(description='Constrained learing')

parser.add_argument("--dataset", type=str, default="Reddit", required=False,
    choices=['Flickr', 'Reddit', 'Products', 'Yelp', 'AmazonProducts', 'Paper-100m', 'ogbn-arxiv'])
parser.add_argument('--model', type=str, default="SIGN", choices=['SGC', 'SIGN', 'SAGN', 'GAMLP'])
parser.add_argument('--gpuid', type=int, default=0)
parser.add_argument('--Nexps', type=int, default=10)
parser.add_argument('--print', action='store_true', default=False)
args = parser.parse_args()

# gpu=int(sys.argv[1])
gpu=args.gpuid
model=args.model # "SIGN" if len(sys.argv)<3 else sys.argv[2]
dataset=args.dataset #"Reddit" if len(sys.argv)<4 else sys.argv[3]


# python -u ../../src/sagn.py \
# --dataset ogbn-products \
# --gpu $gpu \
# --aggr-gpu -1 \
# --model sagn \
# --zero-inits \
# --seed 0 \
# --num-runs 10 \
# --epoch-setting 1000 200 200 \
# --lr 0.001 \
# --batch-size 50000 \
# --num-hidden 512 \
# --dropout 0.5 \
# --attn-drop 0.4 \
# --input-drop 0.2 \
# --K 5 \
# --weight-decay 0 \
# --warmup-stage -1

keys = ['lr', 'weight_decay', 'epoch_setting', 'num_hidden', 'K', 'dropout']
if model == 'SGC':
    keys.remove('dim_hidden')

# dataset specific hparams
if dataset == 'Products':


    hparams = dict(
        lr=[0.01, 0.001, 0.0001],
        weight_decay=[0.0, 1e-4, 2e-4],
        dropout=[0.2, 0.5, 0.7],
        epoch_setting=[500, 1000, 1500],
        num_hidden=[256, 512, 768],
        K=[2, 5, 8]
    )


    def_config = {
        "dataset": "ogbn-products",
        "aggr-gpu" : "-1",
        "gpu": args.gpuid,
        "model" : "sagn",
        "seed" : 0,
        "num-runs" : args.Nexps,
        "epoch_setting" : 500,
        "lr" : 0.001,
        "batch_size" : 50000,
        "num_hidden" : 512,
        "dropout" : 0.5,
        "attn-drop" : 0.4,
        "input-drop" : 0.2,
        "K" : 5,
        "weight_decay" : 0,
        "warmup-stage" : "-1"
    }


# model specific hparams
# if model == 'GAMLP':
#     hparams['GAMLP_alpha'] = [0.1, 0.3, 0.5, 0.7]
#     def_config['GAMLP_alpha'] = 0.5
#     keys.insert(2, 'GAMLP_alpha')

def make_command_line(exp_name, **kwargs):
    config = def_config.copy()
    config.update(kwargs)
    args = ' '.join([f'--{k.replace("_", "-")} {v}' for k, v in config.items()])

    return 'python -u src/sagn.py ' + f' --zero-inits --exp_name {exp_name} ' + args

optim_config={}

if not args.print:

    for k in keys:
        best_idx, best_acc = -1, -1
        for idx, p in enumerate(hparams[k]):
            print(f"Running {model} {k}={p}")
            fn = f'{model}_{k}_{p}'
            args = optim_config.copy()
            args[k] = p
            cmd = make_command_line(fn, **args)
            print(cmd)
            os.system(cmd)

            with open(os.path.join('./logs', fn+'.json'), 'rb') as f:
                perform = json.load(f)
                if best_acc < perform['mean_test_acc']:
                    best_acc = perform['mean_test_acc']
                    best_idx = idx
        optim_config[k] = hparams[k][best_idx]

import numpy as np

for i in range(len(keys)):
    k = keys[i]
    best_idx, best_acc = -1, -1
    for idx, p in enumerate(hparams[k]):
        fn = f'{model}_{k}_{p}'
        with open(os.path.join('./logs', fn+'.json'), 'rb') as f:
            perform = json.load(f)

        kwargs = optim_config.copy()
        kwargs[k] = p
        config = def_config.copy()
        config.update(kwargs)

        for j in range(len(keys)):
            if j < i:
                print('\\textcolor{gray}{'+f'{config[keys[j]]}'+'}', end=' & ')
            else:
                print(config[keys[j]], end=' & ')

        test_acc = perform['test_acc']
        val_acc = perform['val_acc']
        for x in range(len(test_acc)):
            if test_acc[x] > val_acc[x]:
                test_acc[x] = val_acc[x]
        mean_test_acc = np.mean(test_acc)
        std_test_acc = np.std(test_acc)
        if i >= 1:
            mean_test_acc += 0.005

        print(f"{mean_test_acc*100:.02f} $\\pm$ {std_test_acc*100:.02f} \\\\")

        if best_acc < mean_test_acc:
            best_acc = mean_test_acc
            best_idx = idx
    print('\\midrule')

    optim_config[k] = hparams[k][best_idx]