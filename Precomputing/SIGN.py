import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_sparse import SparseTensor

from Precomputing.base import PrecomputingBase

class SIGN(PrecomputingBase):
    def __init__(self, args, data, train_idx, processed_dir):
        super(SIGN, self).__init__(args, data, train_idx, processed_dir)

        self.lins = torch.nn.ModuleList()
        for _ in range(self.num_layers + 1):
            self.lins.append(torch.nn.Linear(self.num_feats, self.dim_hidden))
        self.out_lin = torch.nn.Linear((self.num_layers + 1) * self.dim_hidden, self.num_classes)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

        self.out_lin.reset_parameters()

    def forward(self, xs):
        outs = []
        for i, (x, lin) in enumerate(zip(xs, self.lins)):
            out = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        x = self.out_lin(x)
        return x

class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x


class SIGN_v2(PrecomputingBase):
    def __init__(self, args, data, train_idx, ffn_layers=2, input_drop=0.3):
        super(SIGN_v2, self).__init__(args, data, train_idx)

        in_feats = self.num_feats
        out_feats = self.num_classes
        hidden = self.dim_hidden
        num_hops = self.num_layers + 1
        dropout = self.dropout

        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = nn.Dropout(input_drop)
        for hop in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, ffn_layers, dropout))
        self.project = FeedForwardNet(num_hops * hidden, hidden, out_feats,
                                      ffn_layers, dropout)

    def forward(self, feats):
        feats = [self.input_drop(feat) for feat in feats]
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            hidden.append(ff(feat))
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        return out

    def reset_parameters(self):
        for ff in self.inception_ffs:
            ff.reset_parameters()
        self.project.reset_parameters()

# class SIGN_MLP(torch.nn.Module):
#     def __init__(self, args, data, train_idx):
#         super(SIGN_MLP, self).__init__()

#         self.num_layers = args.num_layers
#         self.dim_hidden = args.dim_hidden
#         self.num_classes = args.num_classes
#         self.num_feats = args.num_feats
#         self.batch_size = args.batch_size
#         self.dropout = args.dropout
#         self.norm = args.norm

#         print('precomputing features, may take a while')
#         data = SIGN(self.num_layers)(data)
#         self.xs = [data.x] + [data[f'x{i}'] for i in range(1, args.num_layers + 1)]

#         # self.norms = torch.nn.ModuleList()
#         # for _ in range(self.num_layers + 1):
#         #     if args.norm == 'BatchNorm':
#         #         self.norms.append(torch.nn.BatchNorm1d(self.num_feats))
#         #     else:
#         #         pass

#         self.lins = torch.nn.ModuleList()
#         for _ in range(self.num_layers + 1):
#             self.lins.append(torch.nn.Linear(self.num_feats, self.dim_hidden))
#         self.out_lin = torch.nn.Linear((self.num_layers + 1) * self.dim_hidden, self.num_classes)

#     def reset_parameters(self):
#         for lin in self.lins:
#             lin.reset_parameters()

#         self.out_lin.reset_parameters()

#     def forward(self, xs):
#         outs = []
#         for i, (x, lin) in enumerate(zip(xs, self.lins)):
#             # if len(self.norms) != 0:
#                 # x = self.norms[i](x)

#             out = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
#             outs.append(out)
#         x = torch.cat(outs, dim=-1)
#         x = self.out_lin(x)
#         return x

#     def train_net(self, input_dict):
#         split_idx = input_dict['split_masks']
#         device = input_dict['device']
#         y = input_dict['y']
#         optimizer = input_dict['optimizer']
#         loss_op = input_dict['loss_op']
#         xs_train = [x[split_idx['train']].to(device) for x in self.xs]
#         y_train_true = y[split_idx['train']].to(device)

#         optimizer.zero_grad()
#         out = self.forward(xs_train)
#         if isinstance(loss_op, torch.nn.NLLLoss):
#             out = F.log_softmax(out, dim=-1)
#         loss = loss_op(out, y_train_true)

#         loss = loss.mean()
#         loss.backward()
#         optimizer.step()

#         if isinstance(loss_op, torch.nn.NLLLoss):
#             total_correct = int(out.argmax(dim=-1).eq(y_train_true).sum())
#             train_acc = float(total_correct / y_train_true.size(0))
#         else:
#             total_correct = int(out.eq(y_train_true).sum())
#             train_acc = float(total_correct / (y_train_true.size(0) * self.num_classes))

#         return float(loss.item()), train_acc

#     def inference(self, input_dict):
#         x_all = input_dict['x']
#         device = input_dict['device']
#         y_preds = []
#         loader = DataLoader(range(x_all.size(0)), batch_size=100000)
#         for perm in loader:
#             y_pred = self.forward([x[perm].to(device) for x in self.xs])
#             y_preds.append(y_pred.cpu())
#         y_preds = torch.cat(y_preds, dim=0)

#         return y_preds
