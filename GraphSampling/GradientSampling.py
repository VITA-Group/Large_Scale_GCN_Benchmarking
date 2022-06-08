import math
import seaborn as sns
import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.typing import OptPairTensor
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul

import GraphSampling.cpp_extension.sample as sample
from GraphSampling._GraphSampling import _GraphSampling


class GSConv(SAGEConv):
    def __init__(self, *args, **kwargs):  # yapf: disable
        kwargs.setdefault("aggr", "mean")
        super(GSConv, self).__init__(*args, **kwargs)

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        out = matmul(adj_t, x[0], reduce=self.aggr)
        return out


class SubGraph:
    def __init__(self, data, adj: SparseTensor, node_idx: Tensor):
        self.ADJ = adj
        self.node_idx = node_idx
        self.adj, self.edge_idx = self.__subgraph__()
        self.x, self.y = data.x[node_idx], data.y[node_idx]
        self.train_mask = data.train_mask[node_idx]
        del self.ADJ

    def __call__(self):
        return self

    def __subgraph__(self):
        row, col, value = self.ADJ.coo()
        rowptr = self.ADJ.storage.rowptr()

        data = torch.ops.torch_sparse.saint_subgraph(self.node_idx, rowptr, row, col)
        row, col, edge_index = data

        if value is not None:
            value = value[edge_index]

        num_nodes = self.node_idx.size(0)
        out = SparseTensor(
            row=row,
            rowptr=None,
            col=col,
            value=value,
            sparse_sizes=(num_nodes, num_nodes),
            is_sorted=True,
        )
        return out, edge_index

    def __repr__(self):
        repr = "(\n"
        for k, v in vars(self).items():
            if isinstance(v, Tensor):
                repr += f"{k}:{v.shape},\n"
            elif isinstance(v, SparseTensor):
                repr += f"{k}:{type(v)},\n"
            else:
                repr += f"{k}:{v},\n"
        return repr + ")"


class GradientSampler:
    def __init__(self, data, batch_size: int):
        super(GradientSampler, self).__init__()

        self.batch_size = batch_size
        self.data = data
        self.N, self.E = data.num_nodes, data.num_edges
        self.prob, self.grads = None, None
        self.__reset_params__()

        row, col = data.edge_index[0], data.edge_index[1]
        edge_attr = torch.ones(row.size(0), requires_grad=True).to(data.x.device)
        # edge_attr = torch.range(0, self.E, dtype=torch.long, requires_grad=True).to(data.x.device)
        self.adj = SparseTensor(
            row=row, col=col, value=edge_attr, sparse_sizes=(self.N, self.N)
        )

    def __reset_params__(self):
        raise NotImplementedError

    def __zero_grad__(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        storage = self.adj.storage
        row, col, grads = storage.row(), storage.col(), storage.value().grad
        # sample nodes with self.prob
        p_cumsum = torch.cumsum(self.prob, 0)
        edge_sample = sample.edge_sample2(p_cumsum, self.batch_size, None)
        source_node_sample = col[edge_sample]
        target_node_sample = row[edge_sample]
        node_idx = torch.cat([source_node_sample, target_node_sample], -1).unique()
        # self.__reset_params__()
        return SubGraph(self.data, self.adj, node_idx)()

    def record(self, grads, idx):
        # self.grads = self.grads.scatter_(0, idx, grads, reduce='add')
        # self.grads = scatter(grads, idx, out=torch.abs(self.grads), reduce='mean')
        # grads = torch.abs(grads)
        self.grads = scatter(grads, idx, out=self.grads, reduce="mean")

    def update(self, rate):
        # self.grads = F.relu(self.grads)
        # self.grads = torch.abs(self.grads)
        # prob = self.prob - lr * self.grads
        # self.prob = prob / torch.sum(prob)
        # self.prob = self.grads / torch.sum(self.grads)
        # indicator = -torch.abs(self.grads)  # NOTE: ±grads or ±abs(grads)
        # _, idx = torch.topk(indicator, k=int(rate * self.N))
        # self.prob[idx] = 0
        # self.prob = self.prob / torch.sum(self.prob)
        # self.__zero_grad__()
        pass


class EdgeSampler(GradientSampler):
    def __reset_params__(self):
        self.prob = torch.tensor([1.0 / self.E] * self.E)
        self.__zero_grad__()

    def __zero_grad__(self):
        self.grads = torch.tensor([0.0] * self.E)


class NodeSampler(GradientSampler):
    def __reset_params__(self):
        self.prob = torch.tensor([1.0 / self.N] * self.N)
        self.__zero_grad__()

    def __zero_grad__(self):
        self.grads = torch.tensor([0.0] * self.N)


class GradientSampling(_GraphSampling):
    def __init__(self, args, data, train_idx, processed_dir):
        super(GradientSampling, self).__init__(args, data, train_idx, processed_dir)

        self.num_layers = args.num_layers
        self.dim_hidden = args.dim_hidden
        self.num_classes = args.num_classes
        self.num_feats = args.num_feats
        self.batch_size = args.batch_size
        self.train_size = train_idx.size(0)
        self.dropout = args.dropout
        self.update_rate = args.dst_update_rate
        self.num_steps = args.num_steps
        self.update_interval = args.dst_update_interval
        self.T_end = args.dst_T_end

        self.convs = torch.nn.ModuleList()
        self.convs.append(GSConv(self.num_feats, self.dim_hidden))
        for _ in range(self.num_layers - 2):
            self.convs.append(GSConv(self.dim_hidden, self.dim_hidden))
        self.convs.append(GSConv(self.dim_hidden, self.num_classes))

        self.update_interval = self.num_steps = 5

        # self.train_loader = iter(EdgeSampler(data, batch_size=self.batch_size))
        self.train_loader = iter(NodeSampler(data, batch_size=self.batch_size))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def train_net(self, input_dict):

        device = input_dict["device"]
        x = input_dict["x"].to(device)
        y = input_dict["y"].to(device)
        optimizer = input_dict["optimizer"]
        epoch = input_dict["epoch"]
        loss_op = input_dict["loss_op"]
        if isinstance(input_dict["loss_op"], torch.nn.NLLLoss):
            loss_op = torch.nn.NLLLoss(reduction="none")
        elif isinstance(input_dict["loss_op"], torch.nn.BCEWithLogitsLoss):
            loss_op = torch.nn.BCEWithLogitsLoss(reduction="none")

        total_loss = total_correct = 0
        for i in range(self.num_steps):
            optimizer.zero_grad()
            batch = next(self.train_loader)
            batch.y, batch.train_mask = batch.y.to(device), batch.train_mask.to(device)
            out = self(batch.x.to(device), batch.adj.to(device))
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
                loss = loss_op(out[batch.train_mask], batch.y[batch.train_mask])
            else:
                loss = loss_op(
                    out[batch.train_mask], batch.y[batch.train_mask].type_as(out)
                )

            weight = 1.0 / self.train_loader.prob[batch.node_idx[batch.train_mask]]
            weight = weight.to(device)
            weight = weight / torch.sum(weight)
            # weight = self.train_loader.prob[batch.node_idx[batch.train_mask]]
            # weight = weight.to(device)
            # weight = weight / torch.sum(weight)
            loss = torch.sum(loss * weight)
            batch.adj.storage.value().retain_grad()
            # out.retain_grad()
            loss.backward()
            storage = batch.adj.storage
            row, col, grads = storage.row(), storage.col(), storage.value().grad

            optimizer.step()
            if isinstance(self.train_loader, EdgeSampler):
                self.train_loader.record(grads.detach().to("cpu"), batch.edge_idx)
            elif isinstance(self.train_loader, NodeSampler):
                self.train_loader.record(grads.detach().to("cpu"), row)
                # self.train_loader.record(torch.norm(out.grad, p=1, dim=1).detach().to('cpu'), batch.node_idx)
            total_loss += float(loss.item())
            if isinstance(loss_op, torch.nn.NLLLoss):
                total_correct += int(out.argmax(dim=-1).eq(batch.y).sum())
            else:
                total_correct += int(out.eq(batch.y).sum())
        train_size = (
            self.train_size
            if isinstance(loss_op, torch.nn.NLLLoss)
            else self.train_size * self.num_classes
        )

        # cosine annealing
        lr = (self.update_rate / 2) * (1 + math.cos(epoch * math.pi / self.T_end))
        # or Inverse Power
        # k = 1
        # lr = self.update_rate * (1 - epoch / self.T_end) ** k
        # constant lr
        # lr = self.lr
        self.train_loader.update(rate=lr)
        # if epoch % 5 == 1:
        #     if not os.path.exists("./figs"):
        #         os.mkdir("./figs")
        #     plot = sns.histplot(self.train_loader.grads)
        #     fig = plot.get_figure()
        #     fig.savefig("./figs/grads_%d.png" % epoch)
            # plot.clf()

        return total_loss / self.num_steps, total_correct / train_size, self.train_loader.grads

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
