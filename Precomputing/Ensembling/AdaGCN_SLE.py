import os
import time

import torch
import torch.nn.functional as F
from Precomputing.base import PrecomputingBase
from torch_sparse import SparseTensor

from .WeakLearners import MLP_SLE


class AdaGCN_SLE(PrecomputingBase):
    def __init__(self, args, data, train_idx, processed_dir, evaluator):
        super(AdaGCN_SLE, self).__init__(args, data, train_idx, processed_dir)
        # first try multiple weak learners
        self.model = MLP_SLE(args)

        sample_weights = torch.ones(data.num_nodes)
        sample_weights = sample_weights[train_idx]
        self.sample_weights = sample_weights / sample_weights.sum()
        self.evaluator = evaluator
        self.SLE_threshold = args.SLE_threshold

        row, col = data.edge_index
        adj_t = SparseTensor(
            row=col, col=row, sparse_sizes=(data.num_nodes, data.num_nodes)
        )

        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        self.adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    def forward(self, x):
        pass

    # def propagate(self, K, data):
    #     assert data.edge_index is not None
    #     row, col = data.edge_index
    #     adj_t = SparseTensor(
    #         row=col, col=row, sparse_sizes=(data.num_nodes, data.num_nodes)
    #     )

    #     deg = adj_t.sum(dim=1).to(torch.float)
    #     deg_inv_sqrt = deg.pow(-0.5)
    #     deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    #     adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    #     assert data.x is not None
    #     xs = [data.x]
    #     data.y_emb = F.one_hot(data.y.view(-1), num_classes=self.num_classes)
    #     y_embs = [data.y_emb]
    #     for i in range(1, K + 1):
    #         xs += [adj_t @ xs[-1]]
    #         data[f"x{i}"] = xs[-1]
    #         y_embs += [adj_t @ y_embs[-1]]
    #         data[f"y_embs{i}"] = y_embs[-1]
    #     return data

    # def precompute(self, data, processed_dir):
    #     print("precomputing features, may take a while.")
    #     t1 = time.time()
    #     data = self.propagate(self.num_layers, data)
    #     self.xs = [data.x] + [data[f"x{i}"] for i in range(1, self.num_layers + 1)]
    #     self.y_embs = [data.y_emb] + [
    #         data[f"y_embs{i}"] for i in range(1, self.num_layers + 1)
    #     ]
    #     t2 = time.time()
    #     print("precomputing finished using %.4f s." % (t2 - t1))
    def propagate(self, x):
        return self.adj_t @ x

    def to(self, device):
        self.sample_weights = self.sample_weights.to(device)
        self.model.to(device)

    def train_and_test(self, input_dict):
        device, split_masks, y, loss_op = (
            input_dict["device"],
            input_dict["split_masks"],
            input_dict["y"],
            input_dict["loss_op"],
        )
        self.to(device)
        results = torch.zeros(y.size(0), self.num_classes)
        y_emb = torch.zeros(y.size(0), self.num_classes)
        y_emb[split_masks["train"]] = F.one_hot(
            y[split_masks["train"]], num_classes=self.num_classes
        ).to(torch.float)

        # for self training
        pseudo_labels = torch.zeros_like(y)
        pseudo_labels[split_masks["train"]] = y[split_masks["train"]]
        pseudo_split_masks = split_masks
        print(
            "------ pseudo labels inited, rate: {:.4f} ------".format(
                pseudo_split_masks["train"].sum() / len(y)
            )
        )

        for i in range(self.num_layers):
            # NOTE: here the num_layers should be the stages in original SAGN
            print(f"\n------ training weak learner with hop {i} ------")
            x = self.xs[i]
            self.train_weak_learner(
                i,
                x,
                y_emb,
                pseudo_labels,
                y,  # the ground truth
                pseudo_split_masks,  # ['train'] is pseudo, valide and test are not modified
                device,
                loss_op,
            )
            self.model.load_state_dict(
                torch.load(f"./.cache/{self.type_model}_{self.dataset}_MLP_SLE.pt")
            )

            # make prediction
            out = self.model.inference(x, y_emb, device)
            # # self training: add hard labels
            # val, pred = torch.max(F.softmax(out, dim=1).to("cpu"), dim=1)
            # SLE_mask = val >= self.SLE_threshold
            # SLE_pred = pred[SLE_mask]
            # # SLE_pred U y
            # pseudo_split_masks["train"] = pseudo_split_masks["train"].logical_or(
            #     SLE_mask
            # )
            # pseudo_labels[SLE_mask] = SLE_pred
            # pseudo_labels[split_masks["train"]] = y[split_masks["train"]]
            # # update y_emb
            # y_emb[pseudo_split_masks["train"]] = F.one_hot(
            #     pseudo_labels[pseudo_split_masks["train"]], num_classes=self.num_classes
            # ).to(torch.float)
            print(
                "------ pseudo labels updated, rate: {:.4f} ------".format(
                    pseudo_split_masks["train"].sum() / len(y)
                )
            )
            y_emb = self.propagate(y_emb)

            # NOTE: adaboosting (SAMME.R)
            out_logp = F.log_softmax(out, dim=1)
            h = (self.num_classes - 1) * (
                out_logp - torch.mean(out_logp, dim=1).view(-1, 1)
            ).cpu()
            results += h
            # adjust weights
            loss = F.nll_loss(
                out_logp[split_masks["train"]].to("cpu"),
                y[split_masks["train"]].long(),
                reduction="none",
            )
            # update weights
            # weight = self.sample_weights.cpu() * torch.exp(
            #     (1 - (self.num_classes - 1)) / (self.num_classes - 1) * loss
            # )
            # weight = weight / weight.sum()
            # self.sample_weights = weight.to(input_dict["device"])
        out, acc = self.evaluate(results, y, split_masks)
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

    def train_weak_learner(
        self, hop, x, y_emb, pseudo_labels, origin_labels, split_mask, device, loss_op
    ):
        # load self.xs[hop] to train self.mlps[hop]
        x_train = x[split_mask["train"]]
        pesudo_labels_train = pseudo_labels[split_mask["train"]]
        y_emb_train = y_emb[split_mask["train"]]
        train_set = torch.utils.data.TensorDataset(
            x_train, y_emb_train, pesudo_labels_train
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, num_workers=8, pin_memory=True
        )
        best_valid_acc = 0.0
        # TODO: tomorrow finish model.train_net (for MLP_SLE)
        for epoch in range(self.epochs):
            _loss, _train_acc = self.model.train_net(
                train_loader, loss_op, self.sample_weights, device
            )
            if (epoch + 1) % self.interval == 0:
                out = self.model.inference(x, y_emb, device)
                out, acc = self.evaluate(out, origin_labels, split_mask)
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
                        self.model.state_dict(),
                        f"./.cache/{self.type_model}_{self.dataset}_MLP_SLE.pt",
                    )
