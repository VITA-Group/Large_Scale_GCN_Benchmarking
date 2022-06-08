#!/usr/bin/env python3
import json
import os
import sys

hparams = dict(
    # lr=[0.01, 0.001],
    # weight_decay=[0, 1e-5, 1e-4],
    # dropout=[0.2, 0.5, 0.7],
    # epochs=[50, 70],
    # dim_hidden=[512, 1024],
    # use_batch_norm=[True, False],
    # batch_size=[5000, 10000],
    SLE_threshold=[0.95, 0.98],
    num_layers=[4, 8],
)

gpu = int(sys.argv[1])
model = "EnGCN"
dataset = "ogbn-papers100M"

def_config = dict(
    cuda_num=gpu,
    type_model=model,
    dataset=dataset,
    lr=0.001,
    weight_decay=0.0,
    dropout=0.5,
    epochs=70,
    dim_hidden=1024,
    num_layers=4,
    batch_size=10000,
    use_batch_norm=True,
    SLE_threshold=0.95,
    N_exp=1,
)


def make_command_line(exp_name, **kwargs):
    config = def_config.copy()
    config.update(kwargs)
    args = " ".join([f"--{k} {v}" for k, v in config.items()])
    return "python main.py " + f"--resume --exp_name {exp_name} --tosparse " + args


optim_config = {}
keys = [
    # "lr",
    # "weight_decay",
    # "dropout",
    # "epochs",
    # "dim_hidden",
    # "use_batch_norm",
    # "batch_size",
    "SLE_threshold",
    "num_layers",
]
for k in keys:
    best_idx, best_acc = -1, -1
    for idx, p in enumerate(hparams[k]):
        print(f"Running {model} {k}={p}")
        fn = f"{model}_{k}_{p}"
        args = optim_config.copy()
        args[k] = p
        cmd = make_command_line(fn, **args)
        print(cmd)
        os.system(cmd)

        with open(os.path.join("./logs", dataset, fn + ".json"), "rb") as f:
            perform = json.load(f)
            if best_acc < perform["mean_test_acc"]:
                best_acc = perform["mean_test_acc"]
                best_idx = idx
    optim_config[k] = hparams[k][best_idx]