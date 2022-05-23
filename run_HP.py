import json
import os
import sys

hparams = dict(
    lr=[0.01, 0.001, 0.0001],
    # weight_decay=[0, 1e-4, 1e-3, 1e-2],
    weight_decay=[0, 1e-5, 5e-5, 1e-4],
    dropout=[0.1, 0.2, 0.5, 0.7],
    epochs=[30, 50, 70],
    dim_hidden=[128, 256, 512],
    num_layers=[2, 4, 8],
    # batch_size=[1000, 2000, 5000],
)

gpu = int(sys.argv[1])
model = sys.argv[2]
dataset = sys.argv[3]

num_steps = 5
if dataset in ["AmazonProducts", "Products"]:
    num_steps = 30

def_config = dict(
    cuda_num=gpu,
    type_model=model,
    dataset=dataset,
    lr=0.01,
    weight_decay=0.0,
    dropout=0.2,
    epochs=50,
    dim_hidden=128,
    num_layers=4,
    batch_size=5000,
    N_exp=3,
    num_steps=num_steps,
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
    "num_layers",
    # "batch_size",
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
