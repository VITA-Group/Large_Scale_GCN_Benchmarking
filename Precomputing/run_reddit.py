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

keys = ['lr', 'weight_decay', 'epochs', 'dim_hidden', 'num_layers', 'dropout']
if model == 'SGC':
    keys.remove('dim_hidden')

# dataset specific hparams
if dataset == 'Reddit':
    hparams = dict(
        lr=[0.01, 0.001, 0.0001],
        weight_decay=[1e-4, 2e-4, 4e-4],
        dropout=[0.1, 0.2, 0.5, 0.7],
        epochs=[20, 30, 40, 50],
        dim_hidden=[128, 256, 512],
        num_layers=[2, 4, 6, 8]
    )

    def_config = dict(
        cuda_num=gpu, type_model=model, dataset=dataset,
        lr=0.01, weight_decay=0.0, dropout=0.2,
        epochs=30, dim_hidden=128, num_layers=2,
        N_exp=args.Nexps
    )
elif dataset == 'Flickr':
    hparams = dict(
        lr=[0.01, 0.001, 0.0001],
        weight_decay=[1e-4, 2e-4, 4e-4],
        dropout=[0.1, 0.2, 0.5, 0.7],
        epochs=[20, 40, 60, 80, 100],
        dim_hidden=[64, 96, 128, 256],
        num_layers=[2, 4, 6, 8]
    )

    def_config = dict(
        cuda_num=gpu, type_model=model, dataset=dataset,
        lr=0.01, weight_decay=0.0, dropout=0.2,
        epochs=30, dim_hidden=128, num_layers=2,
        N_exp=args.Nexps
    )
elif dataset == 'AmazonProducts':
    hparams = dict(
        lr=[0.01, 0.001, 0.0001],
        weight_decay=[1e-4, 2e-4, 4e-4],
        dropout=[0.1, 0.2, 0.5, 0.7],
        epochs=[20, 30, 40, 50],
        dim_hidden=[128, 256, 512],
        num_layers=[2, 4, 8]
    )

    def_config = dict(
        cuda_num=gpu, type_model=model, dataset=dataset,
        lr=0.01, weight_decay=0.0, dropout=0.2,
        epochs=30, dim_hidden=128, num_layers=2,
        N_exp=args.Nexps
    )
elif dataset == 'Products':
    hparams = dict(
        lr=[0.01, 0.001, 0.0001],
        weight_decay=[1e-4, 2e-4, 4e-4],
        dropout=[0.2, 0.5, 0.7],
        epochs=[500, 1000, 1500],
        dim_hidden=[256, 512, 768],
        num_layers=[2, 4, 8]
    )

    def_config = dict(
        cuda_num=gpu, type_model=model, dataset=dataset,
        lr=0.01, weight_decay=0.0, dropout=0.2,
        epochs=500, dim_hidden=256, num_layers=2,
        N_exp=args.Nexps, batch_size=50000
    )

# model specific hparams
# if model == 'GAMLP':
#     hparams['GAMLP_alpha'] = [0.1, 0.3, 0.5, 0.7]
#     def_config['GAMLP_alpha'] = 0.5
#     keys.insert(2, 'GAMLP_alpha')

def make_command_line(exp_name, **kwargs):
    config = def_config.copy()
    config.update(kwargs)
    args = ' '.join([f'--{k} {v}' for k, v in config.items()])
    return 'python main.py ' + f'--resume --cuda True --exp_name {exp_name} ' + args

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

            with open(os.path.join('./logs', dataset, fn+'.json'), 'rb') as f:
                perform = json.load(f)
                if best_acc < perform['mean_test_acc']:
                    best_acc = perform['mean_test_acc']
                    best_idx = idx
        optim_config[k] = hparams[k][best_idx]


for i in range(len(keys)):
    k = keys[i]
    best_idx, best_acc = -1, -1
    for idx, p in enumerate(hparams[k]):
        fn = f'{model}_{k}_{p}'
        with open(os.path.join('./logs', dataset, fn+'.json'), 'rb') as f:
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

        print(f"{perform['mean_test_acc']:.02f} $\\pm$ {perform['std_test_acc']:.02f} \\\\")

        if best_acc < perform['mean_test_acc']:
            best_acc = perform['mean_test_acc']
            best_idx = idx
    print('\\midrule')

    optim_config[k] = hparams[k][best_idx]