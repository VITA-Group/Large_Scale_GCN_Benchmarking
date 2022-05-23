import gc
import json
import os
import random
from datetime import datetime

import numpy as np
import seaborn as sns
import torch
import torch_geometric.datasets
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from sklearn.metrics import f1_score
from torch_sparse import SparseTensor

from GraphSampling import *
from LP.LP_Adj import LabelPropagation_Adj
from options.base_options import BaseOptions
from Precomputing import (JK_GAMLP, R_GAMLP, SAGN, SGC, SIGN, Ensembling,
                          SIGN_v2)
from Precomputing.Ensembling import Ensembling
from trainer import trainer
from utils import print_args


def load_data(dataset_name):
    if dataset_name in ["ogbn-products", "ogbn-papers100M"]:
        root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        dataset = PygNodePropPredDataset(name=dataset_name, root=root)
        processed_dir = dataset.processed_dir
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name=dataset_name)
        data = dataset[0]
        split_masks = {}
        for split in ["train", "valid", "test"]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[split_idx[split]] = True
            data[f"{split}_mask"] = mask
            split_masks[f"{split}"] = data[f"{split}_mask"]
        x = data.x
        y = data.y = data.y.squeeze()

    elif dataset_name in ["Reddit", "Flickr", "AmazonProducts", "Yelp"]:
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        dataset_class = getattr(torch_geometric.datasets, dataset_name)
        dataset = dataset_class(path)
        processed_dir = dataset.processed_dir
        data = dataset[0]
        evaluator = None
        split_masks = {}
        split_masks["train"] = data.train_mask
        split_masks["valid"] = data.val_mask
        split_masks["test"] = data.test_mask
        x = data.x
        y = data.y
        E = data.edge_index.shape[1]
        N = data.train_mask.shape[0]
        data.edge_idx = torch.arange(0, E)
        data.node_idx = torch.arange(0, N)

    else:
        raise Exception(f"the dataset of {dataset} has not been implemented")
    return data, x, y, split_masks, evaluator, processed_dir


def idx2mask(idx, N_nodes):
    mask = torch.tensor([False] * N_nodes, device=idx.device)
    mask[idx] = True
    return mask


class trainer(object):
    def __init__(self, args):

        self.dataset = args.dataset
        self.device = torch.device(f"cuda:{args.cuda_num}" if args.cuda else "cpu")
        self.args = args
        self.args.device = self.device

        self.type_model = args.type_model
        self.epochs = args.epochs
        self.eval_steps = args.eval_steps
        self.type_run = args.type_run
        self.filter_rate = args.filter_rate

        # used to indicate multi-label classification.
        # If it is, using BCE and micro-f1 performance metric
        self.multi_label = args.multi_label
        if self.multi_label:
            self.loss_op = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_op = torch.nn.NLLLoss()

        if self.dataset == "ogbn-arxiv":
            self.data, self.split_idx = load_ogbn(self.dataset)
            self.x, self.y = self.data.x, self.data.y
            self.train_idx = self.split_idx["train"]
            self.evaluator = Evaluator(name="ogbn-arxiv")
            train_mask = idx2mask(self.split_idx["train"], args.N_nodes)
            valid_mask = idx2mask(self.split_idx["valid"], args.N_nodes)
            test_mask = idx2mask(self.split_idx["test"], args.N_nodes)
            self.split_masks = {
                "train": train_mask,
                "valid": valid_mask,
                "test": test_mask,
            }

        else:
            (
                self.data,
                self.x,
                self.y,
                self.split_masks,
                self.evaluator,
                self.processed_dir,
            ) = load_data(args.dataset)
            if self.type_run == "filtered":
                self.data = filtrate_data_with_grads(
                    args.dataset, rate=self.filter_rate
                )
                self.x, self.y = self.data.x, self.data.y
                self.split_masks = {
                    "train": self.data.train_mask,
                    "valid": self.data.val_mask,
                    "test": self.data.val_mask,
                }

        if self.type_model == "GradientSampling":
            self.model = GradientSampling(
                args, self.data, self.split_masks["train"], self.processed_dir
            )
        elif self.type_model == "SIGN":
            self.model = SIGN(
                args, self.data, self.split_masks["train"], self.processed_dir
            )
        self.model.to(self.device)

        if len(list(self.model.parameters())) != 0:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            self.optimizer = None

    def mem_speed_bench(self):
        input_dict = self.get_input_dict(0)
        self.model.mem_speed_bench(input_dict)

    def train_net_w_grad_sampling(self, epoch):
        self.model.train()
        input_dict = self.get_input_dict(epoch)
        if self.type_model == 'GradientSampling':
            train_loss, train_acc, grads = self.model.train_net(input_dict)
        else:
            train_loss, train_acc = self.model.train_net(input_dict)
            grads = 0
        return train_loss, train_acc, grads

    def train_and_test_w_grad_sampling(self, seed):
        results = []
        grads = torch.tensor([0.0] * self.data.num_nodes)
        for epoch in range(self.epochs):
            train_loss, train_acc, tem_grads = self.train_net_w_grad_sampling(
                epoch
            )  # -wz-run
            grads += tem_grads
            print(
                f"Seed: {seed:02d}, "
                f"Epoch: {epoch:02d}, "
                f"Loss: {train_loss:.4f}, "
                f"Approx Train Acc: {train_acc:.4f}"
            )

            if epoch % self.eval_steps == 0 and epoch != 0:
                out, result = self.test_net()
                results.append(result)
                train_acc, valid_acc, test_acc = result
                print(
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {train_loss:.4f}, "
                    f"Train: {100 * train_acc:.2f}%, "
                    f"Valid: {100 * valid_acc:.2f}% "
                    f"Test: {100 * test_acc:.2f}%"
                )

        # plot = sns.histplot(grads)
        # fig = plot.get_figure()
        # fig.savefig(f"./figs/{self.dataset}_grads.png")
        results = 100 * np.array(results)
        best_idx = np.argmax(results[:, 1])
        best_train = results[best_idx, 0]
        best_valid = results[best_idx, 1]
        best_test = results[best_idx, 2]
        print(
            f"Best train: {best_train:.2f}%, "
            f"Best valid: {best_valid:.2f}% "
            f"Best test: {best_test:.2f}%"
        )

        return best_train, best_valid, best_test, grads

    def get_input_dict(self, epoch):
        input_dict = {
            "x": self.x,
            "y": self.y,
            "optimizer": self.optimizer,
            "loss_op": self.loss_op,
            "device": self.device,
            "epoch": epoch,
            "split_masks": self.split_masks,
        }
        return input_dict

    @torch.no_grad()
    def test_net(self):
        self.model.eval()
        input_dict = {"x": self.x, "y": self.y, "device": self.device}
        out = self.model.inference(input_dict)

        if self.evaluator is not None:
            y_true = self.y.unsqueeze(-1)
            y_pred = out.argmax(dim=-1, keepdim=True)

            train_acc = self.evaluator.eval(
                {
                    "y_true": y_true[self.split_masks["train"]],
                    "y_pred": y_pred[self.split_masks["train"]],
                }
            )["acc"]
            valid_acc = self.evaluator.eval(
                {
                    "y_true": y_true[self.split_masks["valid"]],
                    "y_pred": y_pred[self.split_masks["valid"]],
                }
            )["acc"]
            test_acc = self.evaluator.eval(
                {
                    "y_true": y_true[self.split_masks["test"]],
                    "y_pred": y_pred[self.split_masks["test"]],
                }
            )["acc"]
        else:

            if not self.multi_label:
                pred = out.argmax(dim=-1).to("cpu")
                y_true = self.y
                correct = pred.eq(y_true)
                train_acc = (
                    correct[self.split_masks["train"]].sum().item()
                    / self.split_masks["train"].sum().item()
                )
                valid_acc = (
                    correct[self.split_masks["valid"]].sum().item()
                    / self.split_masks["valid"].sum().item()
                )
                test_acc = (
                    correct[self.split_masks["test"]].sum().item()
                    / self.split_masks["test"].sum().item()
                )

            else:
                pred = (out > 0).float().numpy()
                y_true = self.y.numpy()
                # calculating F1 scores
                train_acc = (
                    f1_score(
                        y_true[self.split_masks["train"]],
                        pred[self.split_masks["train"]],
                        average="micro",
                    )
                    if pred[self.split_masks["train"]].sum() > 0
                    else 0
                )

                valid_acc = (
                    f1_score(
                        y_true[self.split_masks["valid"]],
                        pred[self.split_masks["valid"]],
                        average="micro",
                    )
                    if pred[self.split_masks["valid"]].sum() > 0
                    else 0
                )

                test_acc = (
                    f1_score(
                        y_true[self.split_masks["test"]],
                        pred[self.split_masks["test"]],
                        average="micro",
                    )
                    if pred[self.split_masks["test"]].sum() > 0
                    else 0
                )

        return out, (train_acc, valid_acc, test_acc)


def load_ogbn(dataset="ogbn-arxiv"):
    import torch_geometric.transforms as T
    from torch_geometric.utils import to_undirected

    dataset = PygNodePropPredDataset(name=dataset)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    data.y = data.y.squeeze(1)
    return data, split_idx


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


def main(args):  # complete or filtered
    list_test_acc = []
    list_valid_acc = []
    list_train_loss = []

    filedir = f"./logs/{args.dataset}"
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    if not args.exp_name:
        filename = f"{args.type_model}.json"
    else:
        filename = f"{args.exp_name}.json"
    path_json = os.path.join(filedir, filename)

    try:
        resume_seed = 0
        if os.path.exists(path_json):
            if args.resume:
                with open(path_json, "r") as f:
                    saved = json.load(f)
                    resume_seed = saved["seed"] + 1
                    list_test_acc = saved["test_acc"]
                    list_valid_acc = saved["val_acc"]
                    list_train_loss = saved["train_loss"]
            else:
                t = os.path.getmtime(path_json)
                tstr = datetime.fromtimestamp(t).strftime("%Y_%m_%d_%H_%M_%S")
                os.rename(
                    path_json, os.path.join(filedir, filename + "_" + tstr + ".json")
                )
        if resume_seed >= args.N_exp:
            print("Training already finished!")
            return
    except:
        pass

    print_args(args)

    if args.debug_mem_speed:
        trnr = trainer(args)
        trnr.mem_speed_bench()

    for seed in range(resume_seed, args.N_exp):
        print(f"seed (which_run) = <{seed}>")

        args.random_seed = seed
        set_seed(args)
        # torch.cuda.empty_cache()
        trnr = trainer(args)
        print(
            f"---- run with {args.filter_rate} {args.type_run} dataset {args.dataset} ----"
        )
        train_loss, valid_acc, test_acc, grads = trnr.train_and_test_w_grad_sampling(
            seed
        )
        list_test_acc.append(test_acc)
        list_valid_acc.append(valid_acc)
        list_train_loss.append(train_loss)

        if not os.path.exists("grads"):
            os.mkdir("./grads/")
        if trnr.type_run == "complete":
            torch.save(grads, f"./grads/{args.dataset}.pt")

        # del trnr
        # torch.cuda.empty_cache()
        # gc.collect()

        ## record training data
        print(
            "mean and std of test acc: ", np.mean(list_test_acc), np.std(list_test_acc)
        )

        try:
            to_save = dict(
                seed=seed,
                test_acc=list_test_acc,
                val_acc=list_valid_acc,
                train_loss=list_train_loss,
                mean_test_acc=np.mean(list_test_acc),
                std_test_acc=np.std(list_test_acc),
            )
            with open(path_json, "w") as f:
                json.dump(to_save, f)
        except:
            pass
    print(
        "final mean and std of test acc: ",
        f"{np.mean(list_test_acc):.2f} $\\pm$ {np.std(list_test_acc):.2f}",
    )


def filtrate_data_with_grads(dataset, rate):
    grads = torch.load(f"grads/{dataset}.pt")
    data, x, y, split_masks, _, _ = load_data(dataset)
    # 1. grads[train_mask] 2. topk 3. topk(grads) k=len(train_masks)
    indicator = -torch.abs(grads)
    _, idx = torch.topk(indicator, k=int(rate * data.num_nodes))
    # only remove those in training set
    mask = data.train_mask[idx]
    idx = idx[mask]
    mask = torch.ones((data.num_nodes), dtype=torch.bool)
    mask[idx] = False
    node_idx = torch.arange(0, data.num_nodes)[mask]
    print(f"---- reduced training nodes: {float(len(idx))/data.train_mask.sum()} ----")

    def __collate__(data, node_idx):
        adj = SparseTensor(
            row=data.edge_index[0],
            col=data.edge_index[1],
            value=torch.arange(data.num_edges, device=data.edge_index.device),
            sparse_sizes=(data.num_nodes, data.num_nodes),
        )
        N, E = data.num_nodes, data.num_edges
        # filtering
        adj, _ = adj.saint_subgraph(node_idx)
        data.num_nodes = node_idx.size(0)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)
        for key, item in data:
            if key in ["edge_index", "num_nodes"]:
                continue
            if isinstance(item, torch.Tensor) and item.size(0) == N:
                data[key] = item[node_idx]
            elif isinstance(item, torch.Tensor) and item.size(0) == E:
                data[key] = item[edge_idx]
            else:
                data[key] = item
        return data

    return __collate__(data, node_idx)


if __name__ == "__main__":
    args = BaseOptions().initialize()
    main(args)
