import argparse
import math
import os
import random
import time
import json
from copy import deepcopy

import dgl
import dgl.function as fn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load_dataset
from gen_models import get_model
from pre_process import prepare_data
from train_process import test, train
from utils import read_subset_list, generate_subset_list, get_n_params, seed

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run(args, data, device, stage=0, subset_list=None):
    feats, label_emb, teacher_probs, labels, labels_with_pseudos, in_feats, n_classes, \
        train_nid, train_nid_with_pseudos, val_nid, test_nid, evaluator, _ = data
    if args.dataset == "ogbn-papers100M":
        # We only store test/val/test nodes' features for ogbn-papers100M
        labels = labels[torch.cat([train_nid, val_nid, test_nid], dim=0)]
        labels_with_pseudos = labels_with_pseudos[torch.cat([train_nid, val_nid, test_nid], dim=0)]
        id_map = dict(zip(torch.cat([train_nid, val_nid, test_nid], dim=0).cpu().long().numpy(), np.arange(len(train_nid) + len(val_nid) + len(test_nid))))
        map_func = lambda x: torch.from_numpy(np.array([id_map[a] for a in x.cpu().numpy()])).to(device)
        train_nid = map_func(train_nid)
        val_nid = map_func(val_nid)
        test_nid = map_func(test_nid)
        train_nid_with_pseudos = map_func(train_nid_with_pseudos)
    # Raw training set loader
    train_loader = torch.utils.data.DataLoader(
        train_nid, batch_size=args.batch_size, shuffle=True, drop_last=False)
    # Enhanced training set loader (but equal to raw one if stage == 0)
    train_loader_with_pseudos = torch.utils.data.DataLoader(
        train_nid_with_pseudos, batch_size=args.batch_size, shuffle=True, drop_last=False)
    # Validation set loader
    val_loader = torch.utils.data.DataLoader(
        val_nid, batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False)
    # Test set loader
    test_loader = torch.utils.data.DataLoader(
        torch.cat([train_nid, val_nid, test_nid], dim=0), batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False)
    # All nodes loader (including nodes without labels)
    all_loader = torch.utils.data.DataLoader(
        torch.arange(len(labels)), batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False)

    # Initialize model and optimizer for each run
    label_in_feats = label_emb.shape[1] if label_emb is not None else n_classes
    model = get_model(in_feats, label_in_feats, n_classes, stage, args, subset_list=subset_list)
    model = model.to(device)
    print("# Params:", get_n_params(model))
    
    if args.dataset in ["ppi", "ppi_large", "yelp"]:
        # For multilabel classification
        loss_fcn = nn.BCEWithLogitsLoss()
    else:
        # For multiclass classification
        loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                    weight_decay=args.weight_decay)

    # Start training
    best_epoch = 0
    best_val = 0
    best_val_loss = 1e9
    best_test = 0
    num_epochs = args.epoch_setting[stage]
    train_time = []
    inference_time = []
    val_accs = []
    val_loss = []


    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train(model, feats, label_emb, teacher_probs, labels_with_pseudos, loss_fcn, optimizer, train_loader_with_pseudos, args)
        med = time.time()

        if epoch % args.eval_every == 0:
            with torch.no_grad():
                acc = test(model, feats, label_emb, teacher_probs, labels, loss_fcn, val_loader, test_loader, evaluator,
                           train_nid, val_nid, test_nid, args)
            end = time.time()

            # We can choose val_acc or val_loss to select best model (usually it does not matter)
            if (acc[1] > best_val and args.acc_loss == "acc") or (acc[3] < best_val_loss and args.acc_loss == "loss"):
                best_epoch = epoch
                best_val = acc[1]
                best_test = acc[2]
                best_val_loss = acc[3]
                best_model = deepcopy(model)

            train_time.append(med - start)
            inference_time.append(acc[-1])
            val_accs.append(acc[1])
            val_loss.append(acc[-2])
            log = "Epoch {}, Time(s): {:.4f} {:.4f}, ".format(epoch, med - start, acc[-1])
            log += "Best Val loss: {:.4f}, Accs: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best Val: {:.4f}, Best Test: {:.4f}".format(best_val_loss, acc[0], acc[1], acc[2], best_val, best_test)
            print(log)
            
    print("Stage: {}, Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
        stage, best_epoch, best_val, best_test))
    with torch.no_grad():
        best_model.eval()
        probs = []
        if (args.model in ["sagn", "plain_sagn"] and args.weight_style=="attention") and (not args.avoid_features):
            attn_weights = []
        else:
            attn_weights = None
        for batch in test_loader:
            if args.dataset == "ogbn-mag":
                batch_feats = {rel_subset: [x[batch].to(device) for x in feat] for rel_subset, feat in feats.items()}
            else:
                batch_feats = [x[batch].to(device) for x in feats] if isinstance(feats, list) else feats[batch].to(device)
            if label_emb is not None:
                batch_label_emb = label_emb[batch].to(device)
            else:
                batch_label_emb = None
            if (args.model in ["sagn", "plain_sagn"]) and (not args.avoid_features):
                out, a = best_model(batch_feats, batch_label_emb)
            else:
                out = best_model(batch_feats, batch_label_emb)
            if args.dataset in ['yelp', 'ppi', 'ppi_large']:
                out = out.sigmoid()
            else:
                out = out.softmax(dim=1)
            # remember to transfer output probabilities to cpu
            probs.append(out.cpu())
            if (args.model in ["sagn", "plain_sagn"] and args.weight_style=="attention") and (not args.avoid_features):
                attn_weights.append(a.cpu().squeeze(1).squeeze(1))
        probs = torch.cat(probs, dim=0)
        if (args.model in ["sagn", "plain_sagn"] and args.weight_style=="attention") and (not args.avoid_features):
            attn_weights = torch.cat(attn_weights)
        
    del model, best_model
    del feats, label_emb, teacher_probs, labels, labels_with_pseudos
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    return best_val, best_test, probs, train_time, inference_time, val_accs, val_loss, attn_weights


def main(args):
    device = torch.device("cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu))
    aggr_device = torch.device("cpu" if args.aggr_gpu < 0 else f"cuda:{args.aggr_gpu}")
    
    # initial_emb_path = os.path.join("..", "embeddings", args.dataset, 
                            # args.model if (args.model != "simple_sagn") else (args.model + "_" + args.weight_style),
                            # "initial_smoothed_features.pt")

    total_best_val_accs = []
    total_best_test_accs = []
    total_val_accs = []
    total_val_losses = []
    total_preprocessing_times = []
    total_train_times = []
    total_inference_times = []

    path_json = os.path.join('../logs', f'{args.exp_name}.json')
    args.resume = True
    resume_seed = 0
    if os.path.exists(path_json):
        if args.resume:
            with open(path_json, 'r') as f:
                saved = json.load(f)
                resume_seed = saved['seed'] + 1
                total_best_val_accs = [[v] for v in saved['test_acc']]
                total_best_test_accs = [[v] for v in saved['val_acc']]
        else:
            t = os.path.getmtime(path_json)
            tstr = datetime.fromtimestamp(t).strftime('%Y_%m_%d_%H_%M_%S')
            os.rename(path_json, os.path.join(filedir, filename + '_' + tstr + '.json'))
    if resume_seed >= args.num_runs:
        print('Training already finished!')
        return

    for i in range(resume_seed, args.num_runs):
        print("-" * 100)
        print(f"Run {i} start training")
        seed(seed=args.seed + i)
        
        if args.dataset == "ogbn-mag":
            if args.fixed_subsets:
                subset_list = read_subset_list(args.dataset, args.example_subsets_path)
            else:
                g, _, _, _, _, _, _ = load_dataset(aggr_device, args)
                subset_list = generate_subset_list(g, args.sample_size)
        else:
            subset_list = None

        best_val_accs = []
        best_test_accs = []
        val_accs = []
        val_losses = []

        preprocessing_times = []
        train_times = []
        inference_times = []

        for stage in range(len(args.epoch_setting)):
            
            if args.warmup_stage > -1:
                if stage <= args.warmup_stage:
                    probs_path = os.path.join(args.probs_dir, 
                                              args.dataset, 
                                              args.model if (args.weight_style == "attention") else (args.model + "_" + args.weight_style),
                                              f'use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_K_{args.K}_label_K_{args.label_K}_probs_seed_{args.seed + i}_stage_{stage}.pt')
                    print(probs_path)
                    if os.path.exists(probs_path):
                        print(f"bypass stage {stage} since warmup_stage is set and associated file exists.")
                        continue
            print("-" * 100)
            print(f"Stage {stage} start training")
            if stage > 0:
                probs_path = os.path.join(args.probs_dir, args.dataset, 
                                args.model if (args.weight_style == "attention") else (args.model + "_" + args.weight_style),
                                f'use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_K_{args.K}_label_K_{args.label_K}_probs_seed_{args.seed + i}_stage_{stage - 1}.pt')
            else:
                probs_path = ''

            with torch.no_grad():
                data = prepare_data(device, args, probs_path, stage, load_embs=args.load_embs, load_label_emb=args.load_label_emb, subset_list=subset_list)
            preprocessing_times.append(data[-1])
            print(f"Preprocessing costs {(data[-1]):.4f} s")
            best_val, best_test, probs, train_time, inference_time, val_acc, val_loss, attn_weights = run(args, data, device, stage, subset_list=subset_list)
            train_times.append(train_time)
            inference_times.append(inference_time)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            new_probs_path = os.path.join(args.probs_dir, args.dataset, 
                                args.model if (args.weight_style == "attention") else (args.model + "_" + args.weight_style),
                                f'use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_K_{args.K}_label_K_{args.label_K}_probs_seed_{args.seed + i}_stage_{stage}.pt')
            if not os.path.exists(os.path.dirname(new_probs_path)):
                os.makedirs(os.path.dirname(new_probs_path))
            torch.save(probs, new_probs_path)
            best_val_accs.append(best_val)
            best_test_accs.append(best_test)

            # path = os.path.join("../converge_stats", args.dataset, 
            #                     args.model if (args.weight_style == "attention") else (args.model + "_" + args.weight_style),
            #                     f"use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_K_{args.K}_label_K_{args.label_K}_seed_{args.seed + i}_stage_{stage}.csv")
            # if not os.path.exists(os.path.dirname(path)):
            #     os.makedirs(os.path.dirname(path))
            # # print(val_acc)
            # df = pd.DataFrame()
            # df['epoch'] = np.arange(args.eval_every, args.epoch_setting[stage] + 1, args.eval_every)
            # df['val_acc'] = val_acc
            # df.to_csv(path)
            # fig_path = os.path.join("../converge_stats", args.dataset, 
            #             args.model if (args.weight_style == "attention") else (args.model + "_" + args.weight_style),
            #             f"use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_K_{args.K}_label_K_{args.label_K}_seed_{args.seed + i}_stage_{stage}.png")
            # sns.set()
            # line_plt = sns.lineplot(data=df, x='epoch', y='val_acc')
            # line = line_plt.get_figure()
            # line.savefig(fig_path)
            # plt.close()

            # path = os.path.join("../converge_stats", args.dataset, 
            #                     args.model if (args.weight_style == "attention") else (args.model + "_" + args.weight_style),
            #                     f"val_loss_use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_K_{args.K}_label_K_{args.label_K}_seed_{args.seed + i}_stage_{stage}.csv")
            # if not os.path.exists(os.path.dirname(path)):
            #     os.makedirs(os.path.dirname(path))
            # # print(val_loss)
            # df = pd.DataFrame()
            # df['epoch'] = np.arange(args.eval_every, args.epoch_setting[stage] + 1, args.eval_every)
            # df['val_loss'] = val_loss
            # df.to_csv(path)
            # fig_path = os.path.join("../converge_stats", args.dataset, 
            #             args.model if (args.weight_style == "attention") else (args.model + "_" + args.weight_style),
            #             f"val_loss_use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_K_{args.K}_label_K_{args.label_K}_seed_{args.seed + i}_stage_{stage}.png")
            # sns.set()
            # line_plt = sns.lineplot(data=df, x='epoch', y='val_loss')
            # line = line_plt.get_figure()
            # line.savefig(fig_path)
            # plt.close()
            
            # if (args.model in ["sagn", "plain_sagn"] and args.weight_style=="attention") and (not args.avoid_features):
            #     path = os.path.join("../attn_weights", args.dataset, args.model,
            #         f"use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_K_{args.K}_label_K_{args.label_K}_seed_{args.seed + i}_stage_{stage}.csv")
            #     if not os.path.exists(os.path.dirname(path)):
            #         os.makedirs(os.path.dirname(path))
            #     df = pd.DataFrame(data=attn_weights.cpu().numpy())
            #     df.to_csv(path)
            #     fig_path = os.path.join("../attn_weights", args.dataset, args.model,
            #         f"use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_K_{args.K}_label_K_{args.label_K}_seed_{args.seed + i}_stage_{stage}.png")
            #     sns.set()
            #     heatmap_plt = sns.heatmap(df)
            #     heatmap = heatmap_plt.get_figure()
            #     heatmap.savefig(fig_path)
            #     plt.close()

            # del data, df, probs, attn_weights
            # with torch.cuda.device(device):
                # torch.cuda.empty_cache()
        total_best_val_accs.append(best_val_accs)
        total_best_test_accs.append(best_test_accs)
        total_val_accs.append(val_accs)
        total_val_accs.append(val_losses)
        total_preprocessing_times.append(preprocessing_times)
        total_train_times.append(train_times)
        total_inference_times.append(inference_times)
        print(total_best_test_accs, total_best_val_accs)

        path_json = os.path.join('../logs', f'{args.exp_name}.json')
        to_save = dict(
            seed=i,
            test_acc=[v[0] for v in total_best_test_accs],
            val_acc=[v[0] for v in total_best_val_accs],
            mean_test_acc=np.mean([v[0] for v in total_best_test_accs]),
            std_test_acc=np.std([v[0] for v in total_best_test_accs])
        )
        print(to_save)
        with open(path_json, 'w') as f:
            json.dump(to_save, f)

    total_best_val_accs = np.array(total_best_val_accs)
    total_best_test_accs = np.array(total_best_test_accs)
    total_val_accs = np.array(total_val_accs)
    total_preprocessing_times = np.array(total_preprocessing_times)
    total_train_times = np.array(total_train_times, dtype=object)
    total_inference_times = np.array(total_inference_times, dtype=object)
    # print(total_preprocessing_times)
    # print(total_train_times)
    # print(total_inference_times)

    for stage in range(len(args.epoch_setting)):
        print(f"Stage: {stage}, Val accuracy: {np.mean(total_best_val_accs[:, stage]):.4f}±"
            f"{np.std(total_best_val_accs[:, stage]):.4f}")
        print(f"Stage: {stage}, Test accuracy: {np.mean(total_best_test_accs[:, stage]):.4f}±"
            f"{np.std(total_best_test_accs[:, stage]):.4f}")
        print(f"Stage: {stage}, Preprocessing time: {np.mean(total_preprocessing_times[:, stage]):.4f}±"
            f"{np.std(total_preprocessing_times[:, stage]):.4f}")
        print(f"Stage: {stage}, Training time: {np.hstack(total_train_times[:, stage]).mean():.4f}±"
            f"{np.hstack(total_train_times[:, stage]).std():.4f}")
        print(f"Stage: {stage}, Inference time: {np.hstack(total_inference_times[:, stage]).mean():.4f}±"
            f"{np.hstack(total_inference_times[:, stage]).std():.4f}")


def define_parser():
    parser = argparse.ArgumentParser(description="Scalable Adaptive Graph neural Networks with Self-Label-Enhance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default='')
    parser.add_argument("--epoch-setting", nargs='+',type=int, default=[500, 200, 150])
    parser.add_argument("--warmup-stage", type=int, default=-1,
                        help="(Only for testing) select the stage from which the script starts to train \
                              based on trained files, -1 for cold starting")
    parser.add_argument("--load-embs", action="store_true",
                        help="This option is used to save memory cost when performing aggregations.")
    parser.add_argument("--load-label-emb", action="store_true",
                        help="This option is used to save memory cost when performing first label propagation.")         
    parser.add_argument("--acc-loss", type=str, default="acc")
    parser.add_argument("--avoid-features", action="store_true")
    parser.add_argument("--use-labels", action="store_true")
    parser.add_argument("--inductive", action="store_true")
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--use-norm", action='store_true')
    parser.add_argument("--memory-efficient", action='store_true')
    parser.add_argument("--num-hidden", type=int, default=512)
    parser.add_argument("--K", type=int, default=3,
                        help="number of hops")
    parser.add_argument("--label-K", type=int, default=9,
                        help="number of label propagation hops")
    parser.add_argument("--zero-inits", action="store_true", 
                        help="Whether to initialize hop attention vector as zeros")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--dataset", type=str, default="ppi")
    parser.add_argument("--data_dir", type=str, default="../../dataset")
    parser.add_argument("--model", type=str, default="sagn")
    parser.add_argument("--pretrain-model", type=str, default="ComplEx")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--weight-style", type=str, default="attention")
    parser.add_argument("--focal", type=str, default="first")
    parser.add_argument("--mag-emb", action="store_true")
    parser.add_argument("--position-emb", action="store_true")
    parser.add_argument("--label-residual", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--input-drop", type=float, default=0.2,
                        help="dropout on input features")
    parser.add_argument("--attn-drop", type=float, default=0.4,
                        help="dropout on hop-wise attention scores")
    parser.add_argument("--label-drop", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--aggr-gpu", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--eval-batch-size", type=int, default=100000,
                        help="evaluation batch size")
    parser.add_argument("--mlp-layer", type=int, default=2,
                        help="number of MLP layers")
    parser.add_argument("--label-mlp-layer", type=int, default=4,
                        help="number of label MLP layers")
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.9, 0.9],
                        help="threshold used to generate pseudo hard labels")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="number of times to repeat the experiment")
    parser.add_argument("--example-subsets-path", type=str, default="/home/scx/NARS/sample_relation_subsets/examples")
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--fixed-subsets", action="store_true")
    parser.add_argument("--emb-path", type=str, default="/home/scx/NARS/")
    parser.add_argument("--probs_dir", type=str, default="../intermediate_outputs")
    return parser

if __name__ == "__main__":
    parser = define_parser()
    args = parser.parse_args()
    print(args)
    main(args)


