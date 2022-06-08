import os

import torch
import torch.nn.functional as F
from Precomputing.base import PrecomputingBase

from .WeakLearners import MLP


class GBGCN(PrecomputingBase):
    def __init__(self, args, data, train_idx, processed_dir, evaluator):
        super(GBGCN, self).__init__(args, data, train_idx, processed_dir)
        self.mlp = MLP(args)

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, xs):
        pass

    def to(self, device):
        self.mlp.to(device)

    def train_and_test(self, input_dict):
        # NOTE: using one mlp works better than ensembling several mlps
        split_mask = input_dict["split_masks"]
        y = input_dict["y"]
        results = torch.zeros(y.size(0), self.num_classes)
        self.to(input_dict["device"])
        for i in range(self.num_layers):
            print(f"\n------ training weak learner with hop {i} ------")
            self.train_weak_leaner(i, input_dict)
            self.mlp.load_state_dict(
                torch.load(f"./.cache/{self.type_model}_{self.dataset}_mlp.pt")
            )
            # adaboosting (SAMME.R)
            out = self.mlp.inference(self.xs[i], input_dict["device"])
            out_logp = F.log_softmax(out, dim=1)
            h = (self.num_classes - 1) * (
                out_logp - torch.mean(out_logp, dim=1).view(-1, 1)
            ).cpu()
            results += h
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

    def train_weak_leaner(self, hop: int, input_dict):
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
            loss, train_acc = self.mlp.train(
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
                        weak_learner.state_dict(),
                        f"./.cache/{self.type_model}_{self.dataset}_mlp.pt",
                    )
