import gc
import json
import os
import random
from datetime import datetime

import numpy as np
import optuna
import torch
from optuna.trial import TrialState

from options.base_options import BaseOptions
from trainer import trainer


def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


def objective(trial):
    args = BaseOptions().initialize()
    args.N_exp = 1
    args.type_model = "EnGCN"
    args.lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    args.SLE_threshold = trial.suggest_uniform("SLE_threshold", 0.5, 0.9)
    args.dropout = trial.suggest_uniform("dropout", 0.1, 0.7)
    args.num_heads = trial.suggest_int("num_heads", 1, 8)
    args.dim_hidden = trial.suggest_categorical("dim_hidden", [256, 512, 1024])
    args.use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    args.num_layers = trial.suggest_categorical("num_layers", [4, 8])

    seed = 123
    args.random_seed = seed
    set_seed(args)
    # torch.cuda.empty_cache()
    trnr = trainer(args, trial=trial)
    _, valid_acc, test_acc = trnr.train_ensembling(seed)

    del trnr
    torch.cuda.empty_cache()
    gc.collect()
    return test_acc


def main():
    args = BaseOptions().initialize()
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna_engcn.db",
        study_name=f"{args.dataset}_{args.type_model}",
        load_if_exists=True,
    )
    # study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
