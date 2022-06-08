import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Precomputing.base import PrecomputingBase


class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(args.num_feats, args.dim_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(args.dim_hidden, args.num_classes),
        )
        self.optimizer = torch.optim.Adam(
            self.mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        if args.multi_label:
            self.loss_op = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_op = torch.nn.NLLLoss()

    def forward(self, x):
        return self.mlp(x)

    def reset_parameters(self):
        for lin in self.mlp:
            if isinstance(lin, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(lin.weight)
                torch.nn.init.zeros_(lin.bias)

    def train(self, x, y):
        self.optimizer.zero_grad()
        out = self.forward(x)
        if isinstance(self.loss_op, torch.nn.NLLLoss):
            out = F.log_softmax(out, dim=-1)
            y = y.long()
        elif isinstance(self.loss_op, torch.nn.BCEWithLogitsLoss):
            y = y.float()
        loss = self.loss_op(out, y)
        loss.backward()
        self.optimizer.step()
        return loss.item(), out


class Bagging(PrecomputingBase):
    def __init__(self, args, data, train_idx, processed_dir):
        super(Bagging, self).__init__(args, data, train_idx, processed_dir)

        self.mlps = []
        for _ in range(self.num_layers + 1):
            self.mlps.append(MLP(args))

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, xs):
        outs = []
        for i, (x, mlp) in enumerate(zip(xs, self.mlps)):
            out = mlp(x)
            outs.append(out)
        return outs

    def to(self, device):
        for mlp in self.mlps:
            mlp.to(device)

    def train_net(self, input_dict):
        split_idx = input_dict["split_masks"]
        device = input_dict["device"]

        xs_train = torch.cat([x[split_idx["train"]] for x in self.xs], -1)
        y_train = input_dict["y"][split_idx["train"]]
        dim_feat = self.xs[0].shape[-1]

        train_set = torch.utils.data.TensorDataset(xs_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, num_workers=8, pin_memory=True
        )

        total_correct = 0
        y_true, y_preds = [], []

        for xs, y in train_loader:

            xs = [x.to(device) for x in torch.split(xs, dim_feat, -1)]
            y = y.to(device)

            loss, out = 0, 0
            for x, mlp in zip(xs, self.mlps):
                loss_tem, out_tem = mlp.train(x, y)
                loss += loss_tem
                out += out_tem

            if not self.multi_label:
                y_preds.append(out.argmax(dim=-1).detach().cpu())
                y_true.append(y.detach().cpu())
            else:
                y_preds.append(out.detach().cpu())
                y_true.append(y.detach().cpu())

        y_true = torch.cat(y_true, 0)
        y_preds = torch.cat(y_preds, 0)
        if not self.multi_label:
            total_correct = y_preds.eq(y_true).sum().item()
            train_acc = float(total_correct / y_train.size(0))
        else:
            y_preds = (y_preds > 0).float().numpy()
            train_acc = f1_score(y_true, y_preds, average="micro")

        return float(loss), train_acc

    @torch.no_grad()
    def inference(self, input_dict):
        x_all = input_dict["x"]
        device = input_dict["device"]
        y_preds = []
        loader = DataLoader(range(x_all.size(0)), batch_size=100000)
        for perm in loader:
            y_pred = self([x[perm].to(device) for x in self.xs])
            out = 0
            for out_i in y_pred:
                out_logp = F.log_softmax(out_i, dim=1)
                h = (self.num_classes - 1) * (
                        out_logp - torch.mean(out_logp, dim=1).view(-1, 1)
                    )
                out += h
            y_pred = out
            y_preds.append(y_pred.cpu())
        y_preds = torch.cat(y_preds, dim=0)

        return y_preds
