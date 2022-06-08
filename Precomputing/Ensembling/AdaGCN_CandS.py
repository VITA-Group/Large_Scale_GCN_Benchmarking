import os
import time

import torch
import torch.nn.functional as F
from torch_sparse.tensor import SparseTensor
from Precomputing.base import PrecomputingBase
from torch_geometric.nn.models import CorrectAndSmooth

from .WeakLearners import MLP


# DONE reproduce the results of 3 layers MLP on ogbn-products 83.94%
# TODO first try add [[C&S]] after Ensembling (AdaBoosting)
# TODO Then try add [[C&S]] before adaboosting, i.e. when we do the majority vote.
class AdaGCN_CandS(PrecomputingBase):
    def __init__(self, args, data, train_idx, processed_dir, evaluator):
        super(AdaGCN_CandS, self).__init__(args, data, train_idx, processed_dir)
        self.mlp = MLP(args)

        sample_weights = torch.ones(data.num_nodes)
        sample_weights = sample_weights[train_idx]
        self.sample_weights = sample_weights / sample_weights.sum()
        self.evaluator = evaluator
        self.post = CorrectAndSmooth(
            num_correction_layers=args.LP__num_propagations1,
            correction_alpha=args.LP__alpha1,
            num_smoothing_layers=args.LP__num_propagations2,
            smoothing_alpha=args.LP__alpha2,
            autoscale=False,
            scale=20.0,
        )
        num_nodes = int(data.edge_index.max()) + 1 if data.edge_index.numel() > 0 else 0
        self.adj_t = SparseTensor(
            row=data.edge_index[0],
            col=data.edge_index[1],
            value=None,
            sparse_sizes=(num_nodes, num_nodes),
        )

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, xs):
        pass

    def to(self, device):
        self.sample_weights = self.sample_weights.to(device)
        self.mlp.to(device)

    def train_and_test(self, input_dict):
        # NOTE: using one mlp works better than ensembling several mlps
        split_mask = input_dict["split_masks"]
        device = input_dict["device"]
        y = input_dict["y"]
        results = torch.zeros(y.size(0), self.num_classes)
        self.to(device)
        for i in range(self.num_layers):
            print(f"\n------ training weak learner with hop {i} ------")
            self.train_weak_learner(i, input_dict)
            self.mlp.load_state_dict(
                torch.load(f"./.cache/{self.type_model}_{self.dataset}_mlp.pt")
            )
            # adaboosting (SAMME.R)
            out = self.mlp.inference(self.xs[i], input_dict["device"])

            # `results` is the y_soft
            adj_t = self.adj_t
            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
            DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t

            y_soft = torch.softmax(out, dim=1)
            print("\n --- Correcting and Smoothing --- ")
            t1 = time.time()
            y_soft = self.post.correct(
                y_soft, y[split_mask["train"]], split_mask["train"], DAD
            )
            y_soft = self.post.smooth(
                y_soft, y[split_mask["train"]], split_mask["train"], DA
            )
            t2 = time.time()
            print(" --- C&S takes %.4f sec --- " % (t2 - t1))

            out_logp = torch.log(y_soft)
            h = (self.num_classes - 1) * (
                out_logp - torch.mean(out_logp, dim=1).view(-1, 1)
            ).cpu()
            results += h
            # results += out
            # adjust weights
            loss = F.nll_loss(
                out_logp[split_mask["train"]].to("cpu"),
                y[split_mask["train"]].long(),
                reduction="none",
            )
            # update weights
            weight = self.sample_weights.cpu() * torch.exp(
                (1 - (self.num_classes - 1)) / (self.num_classes - 1) * loss
            )
            weight = weight / weight.sum()
            self.sample_weights = weight.to(input_dict["device"])
        out, acc = self.evaluate(results, y, split_mask)
        print(
            f"Final train acc: {acc['train']*100:.4f}, "
            f"Final valid acc: {acc['valid']*100:.4f}, "
            f"Dianl test acc: {acc['test']*100:.4f}"
        )
        return acc["train"], acc["valid"], acc["test"]

    def evaluate(self, out, y, split_mask):
        acc = {}
        if self.evaluator:
            y_true = y.unsqueeze(-1)
            y_pred = out.argmax(dim=-1, keepdim=True)
            for phase in ["train", "valid", "test"]:
                acc[phase] = self.evaluator.eval(
                    {
                        "y_true": y_true[split_mask[phase]],
                        "y_pred": y_pred[split_mask[phase]],
                    }
                )["acc"]
        else:
            pred = out.argmax(dim=1).to("cpu")
            y_true = y
            correct = pred.eq(y_true)
            for phase in ["train", "valid", "test"]:
                acc[phase] = (
                    correct[split_mask[phase]].sum().item()
                    / split_mask[phase].sum().item()
                )
        return out, acc

    def train_weak_learner(self, hop: int, input_dict):
        # load input_dict
        split_mask = input_dict["split_masks"]
        device = input_dict["device"]
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]

        x, y = self.xs[hop], input_dict["y"]
        # load self.xs[hop] to train self.mlps[hop]
        x_train = x[split_mask["train"]]
        y_train = y[split_mask["train"]]
        train_set = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, num_workers=8, pin_memory=True
        )
        best_valid_acc = 0.0
        for epoch in range(self.epochs):
            loss, train_acc = self.mlp.train_net(
                train_loader, loss_op, self.sample_weights, device
            )
            # print(
            #     f"Epoch: {epoch:02d}, "
            #     f"Loss: {float(loss):.4f}, "
            #     f"Approx Train Acc: {train_acc:.4f}"
            # )
            if (epoch + 1) % self.interval == 0:
                out = self.mlp.inference(self.xs[hop], device)

                out, acc = self.evaluate(
                    out, input_dict["y"], input_dict["split_masks"]
                )
                print(
                    f"Model: {hop:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Train acc: {acc['train']*100:.4f}, "
                    f"Valid acc: {acc['valid']*100:.4f}, "
                    f"Test acc: {acc['test']*100:.4f}"
                )
                if acc["valid"] > best_valid_acc:
                    best_valid_acc = acc["valid"]
                    if not os.path.exists(".cache/"):
                        os.mkdir(".cache/")
                    torch.save(
                        self.mlp.state_dict(),
                        f"./.cache/{self.type_model}_{self.dataset}_mlp.pt",
                    )
