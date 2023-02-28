#!/usr/bin/env python3
import json
import os
import sys

hparams = dict(
    lr=[0.01, 0.001, 0.0001],
    weight_decay=[0, 1e-5, 1e-4],
    dropout=[0.1, 0.2, 0.5, 0.7],
    epochs=[30, 50, 70],
    dim_hidden=[128, 256, 512],
    use_batch_norm=[True, False],
    batch_size=[5000, 10000],
    SLE_threshold=[0.8, 0.9, 0.95],
    num_layers=[4, 5, 8],
)

gpu = int(sys.argv[1])
model = "EnGCN"
dataset = "Reddit"

def_config = dict(
    cuda_num=gpu,
    type_model=model,
    dataset=dataset,
    lr=0.001,
    weight_decay=0.0,
    dropout=0.5,
    epochs=50,
    dim_hidden=256,
    num_layers=4,
    batch_size=5000,
    use_batch_norm=True,
    SLE_threshold=0.9,
    N_exp=3,
)


def make_command_line(exp_name, **kwargs):
    config = def_config.copy()
    config.update(kwargs)
    args = " ".join([f"--{k} {v}" for k, v in config.items()])
    return "python main.py " + f"--resume --exp_name {exp_name} " + args


optim_config = {}
keys = [
    "lr",
    "weight_decay",
    "dropout",
    "epochs",
    "dim_hidden",
    "use_batch_norm",
    "batch_size",
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
