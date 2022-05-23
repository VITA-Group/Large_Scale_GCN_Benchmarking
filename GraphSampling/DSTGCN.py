import json
import math
import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import GraphSAINTRandomWalkSampler as RWSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.typing import OptPairTensor
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul

from utils import GB, MB, compute_tensor_bytes, get_memory_usage

from ._GraphSampling import _GraphSampling

# TODO: review and pune the code of dstgcn


class GSConv(SAGEConv):
    def __init__(self, *args, **kwargs):  # yapf: disable
        kwargs.setdefault("aggr", "mean")
        super(GSConv, self).__init__(*args, **kwargs)

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        out = matmul(adj_t, x[0], reduce=self.aggr)
        return out


def from_edge_index(edge_index, edge_attr=None, num_nodes=None, requires_grad=True):
    assert isinstance(edge_index, torch.Tensor)
    row, col = edge_index[0], edge_index[1]
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    if edge_attr is None:
        edge_attr = torch.ones(row.size(0), requires_grad=requires_grad).cpu()
    else:
        edge_attr = edge_attr.view(-1).copy(requires_grad=requires_grad).cpu()
        assert edge_attr.size(0) == row.size(0)
    adj = SparseTensor(
        row=row, col=col, value=edge_attr, sparse_sizes=(num_nodes, num_nodes)
    )
    return adj


def subgraph(
    src: SparseTensor, node_idx: torch.Tensor, num_nodes: int = None
) -> Tuple[SparseTensor, torch.Tensor]:
    row, col, value = src.coo()
    rowptr = src.storage.rowptr()

    data = torch.ops.torch_sparse.saint_subgraph(node_idx, rowptr, row, col)
    row, col, edge_index = data

    if value is not None:
        value = value[edge_index]

    if num_nodes is None:
        num_nodes = node_idx.size(0)
    out = SparseTensor(
        row=row,
        rowptr=None,
        col=col,
        value=value,
        sparse_sizes=(num_nodes, num_nodes),
        is_sorted=True,
    )

    return out


class DSTGCN(_GraphSampling):
    """
    a pseudo code for dynamic sparse training for large-scale graphs.
    **Edge sampler**:
    1. Initialization:
        sampling a subgraph with _edge_sampler(adj, rate)
    2. update:
        for each iteration:
            forward and backward pass
            get the grads of all edges

    """

    def __init__(self, args, data, train_idx, processed_dir):
        super(DSTGCN, self).__init__(args, data, train_idx, processed_dir)
        self.num_steps = args.num_steps
        self.dst_sample_coverage = args.dst_sample_coverage
        self.dst_update_rate = args.dst_update_rate
        self.dst_update_interval = args.dst_update_interval
        self.dst_T_end = args.dst_T_end
        self.walk_length = args.dst_walk_length
        self.dst_update_decay = args.dst_update_decay
        self.dst_update_scheme = args.dst_update_scheme
        self.dst_grads_scheme = args.dst_grads_scheme
        self.dst_agg = "mean"
        self.batch_size = args.batch_size
        # set dst_sample_coverage consistent with batch_size
        self.dst_sample_coverage = self.batch_size / data.num_nodes
        self.epochs = args.epochs
        self.N = data.num_nodes
        self.E = data.num_edges
        # Network Architecture
        self.convs = torch.nn.ModuleList()
        self.convs.append(GSConv(self.num_feats, self.dim_hidden))
        for _ in range(self.num_layers - 2):
            self.convs.append(GSConv(self.dim_hidden, self.dim_hidden))
        self.convs.append(GSConv(self.dim_hidden, self.num_classes))

        self.ADJ = from_edge_index(
            data.edge_index, num_nodes=self.N, edge_attr=None
        )  # static variable
        # randomly initialization
        self.node_idx = torch.arange(0, self.N)
        self.node_idx = self._random_walk_sampler(
            self.node_idx, self.dst_sample_coverage
        )
        # if self.dst_update_scheme == 'edge':
        #     self.node_idx = self._edge_sampler(self.ADJ, int(self.batch_size / 2))
        # else:
        #     self.node_idx = self._random_walk_sampler(self.node_idx, self.dst_sample_coverage)
        self.sampled_adj = subgraph(self.ADJ, self.node_idx)
        self.approx_batch_size = self.dst_sample_coverage * self.N
        # properties for logging
        self.log_sampled_node_idx = self.node_idx
        self.log_num_nodes = len(self.node_idx)
        self.update_rate = self.dst_update_rate
        self._reset_parameters()
        self.saved_args = vars(args)

    def logging(self):
        # get num of nodes that have been sampled
        sampled_N = len(self.log_sampled_node_idx)
        # print('---> coverage rate : {:.2f}, current_update_rate : {:.2f}, num_current_nodes：{}'.format(
        #     float(sampled_N / self.N), self.update_rate, len(self.node_idx)))

    def _random_walk_sampler(self, node_idx, rate):
        # randomly sample start nodes according to the rate
        num_start_nodes = int(rate * len(self.node_idx) / self.walk_length)
        start_idx = torch.randperm(len(node_idx), dtype=torch.long)[:num_start_nodes]
        start = node_idx[start_idx]
        node_idx = (
            self.ADJ.random_walk(start.flatten(), self.walk_length).view(-1).unique()
        )
        return node_idx

    def _edge_sampler(self, adj: SparseTensor, num_samples: int):
        row, col = adj.storage.row(), adj.storage.col()
        num_edges = len(row)
        idx = torch.randperm(num_edges, dtype=torch.long)[:num_samples]
        sampled_node_idx = torch.cat((row[idx], col[idx]), dim=0).view(-1).unique()
        return sampled_node_idx

    @property
    def the_other_node_idx(self):
        node_idx = torch.arange(0, self.N)
        mask = torch.ones((self.N,), dtype=torch.bool)
        mask[self.node_idx] = False
        return node_idx[mask]

    def _new_random_walk_sampler(self, rate):
        # !! start nodes should exclude sampled nodes
        # random walk can not approach very far from the start nodes
        node_idx = self.the_other_node_idx
        num_start_nodes = int(rate * self.approx_batch_size / self.walk_length)
        start_idx = torch.randperm(len(node_idx), dtype=torch.long)[:num_start_nodes]
        start = node_idx[start_idx]
        node_idx = (
            self.ADJ.random_walk(start.flatten(), self.walk_length).view(-1).unique()
        )
        return node_idx

    def _reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def _update_with_edge_grad(self, rate):  # update self.sampled_adj
        storage = self.sampled_adj.storage
        grad, row, col = storage.value().grad, storage.row(), storage.col()
        grad = torch.abs(grad)
        num_edges = len(self.sampled_adj.storage.row())
        num_nodes = len(self.node_idx)
        # remove the topk edges with least gradient
        val, idx = torch.topk(-grad, k=int(rate * num_edges))
        node_idx = torch.cat((row[idx], col[idx]), dim=0).view(-1).unique()
        mask = torch.ones((num_nodes,), dtype=torch.bool)
        mask[node_idx] = False
        num_removed_nodes = len(node_idx)
        node_idx = self.node_idx[mask]
        # add new edges by sampling edges
        # !!! difficult to control the number of sampled nodes
        sampled_node_idx = self._edge_sampler(self.ADJ, int(num_removed_nodes / 1.5))
        node_idx = torch.cat((sampled_node_idx, node_idx), dim=0).view(-1).unique()
        # record the sampled_node_idx for computing the coverage rate
        self.log_sampled_node_idx = (
            torch.cat((self.log_sampled_node_idx, node_idx), dim=0).view(-1).unique()
        )
        self.log_num_nodes = len(node_idx)

        self.sampled_adj = subgraph(self.ADJ, node_idx)
        self.node_idx = node_idx

    def _update_with_edge_grad_2(self, rate):
        # update with edge grad and random_walk_sampler
        storage = self.sampled_adj.storage
        grad, row, col = storage.value().grad, storage.row(), storage.col()
        grad = torch.abs(grad)
        num_edges = len(self.sampled_adj.storage.row())
        num_nodes = len(self.node_idx)
        # remove the topk edges with least gradient
        val, idx = torch.topk(-grad, k=int(rate * num_edges))
        node_idx = torch.cat((row[idx], col[idx]), dim=0).view(-1).unique()
        mask = torch.ones((num_nodes,), dtype=torch.bool)
        mask[node_idx] = False
        num_removed_nodes = len(node_idx)
        node_idx = self.node_idx[mask]
        # add new edges by sampling edges
        sampled_node_idx = self._new_random_walk_sampler(rate)
        node_idx = torch.cat((sampled_node_idx, node_idx), dim=0).view(-1).unique()
        # record the sampled_node_idx for computing the coverage rate
        self.log_sampled_node_idx = (
            torch.cat((self.log_sampled_node_idx, node_idx), dim=0).view(-1).unique()
        )
        self.log_num_nodes = len(node_idx)

        self.sampled_adj = subgraph(self.ADJ, node_idx)
        self.node_idx = node_idx

    def _update_with_node_grad_2(self, rate):
        storage = self.sampled_adj.storage
        grad, row, col = storage.value().grad, storage.row(), storage.col()
        grad = torch.abs(grad)
        node_grad = scatter(grad, row, reduce="mean")
        # node_grad += scatter(grad, col, reduce='mean')
        # grad_row = scatter(grad, row, reduce='mean')
        # grad_col = scatter(grad, col, reduce='mean')
        # node_grad = grad_row + grad_col
        # len(self.node_idx) != len(node_grad) as there are some isolated nodes
        num_nodes = len(self.node_idx)
        k = int(rate * num_nodes / self.walk_length)
        val, idx = torch.topk(-node_grad, k=k)
        # perform random walk based on idx
        idx = (
            self.sampled_adj.random_walk(idx.flatten(), self.walk_length)
            .view(-1)
            .unique()
        )
        mask = torch.ones((num_nodes,), dtype=torch.bool)
        mask[idx] = False
        sampled_node_idx = self.node_idx[mask]
        new_node_idx = self._new_random_walk_sampler(rate)
        node_idx = torch.cat((sampled_node_idx, new_node_idx), dim=0).view(-1).unique()
        # record the sampled_node_idx for computing the coverage rate
        self.log_sampled_node_idx = (
            torch.cat((self.log_sampled_node_idx, node_idx), dim=0).view(-1).unique()
        )
        self.log_num_nodes = len(self.log_sampled_node_idx)

        self.sampled_adj = subgraph(self.ADJ, node_idx)
        self.node_idx = node_idx

    def _update_with_node_grad_3(self, rate):
        storage = self.sampled_adj.storage
        grad, row, col = storage.value().grad, storage.row(), storage.col()
        grad = torch.abs(grad)
        node_grad = scatter(grad, row, reduce="mean")
        num_nodes = len(self.node_idx)
        k = int(rate * num_nodes)
        val, idx = torch.topk(torch.abs(node_grad), k=k)
        # preserve the remain node_idx as preserved_node_idx
        mask = torch.ones((num_nodes), dtype=torch.bool)
        mask[idx] = False
        preserved_node_idx = self.node_idx[mask]
        node_idx = (
            self.ADJ.random_walk(preserved_node_idx.flatten(), self.walk_length)
            .view(-1)
            .unique()[:num_nodes]
        )
        # record the sampled_node_idx for computing the coverage rate
        self.log_sampled_node_idx = (
            torch.cat((self.log_sampled_node_idx, node_idx), dim=0).view(-1).unique()
        )
        self.log_num_nodes = len(self.log_sampled_node_idx)

        self.sampled_adj = subgraph(self.ADJ, node_idx)
        self.node_idx = node_idx

    def _update_with_node_grad(self, rate):
        storage = self.sampled_adj.storage
        grad, row, col = storage.value().grad, storage.row(), storage.col()
        if self.dst_grads_scheme == 0:
            grad = torch.abs(grad)
        elif self.dst_grads_scheme == 1:
            grad = -torch.abs(grad)
        elif self.dst_grads_scheme == 2:
            grad = grad
        elif self.dst_grads_scheme == 3:
            grad = -grad
        else:
            raise ValueError("dst_grads_scheme: value not found")
        node_grad = scatter(grad, row, reduce=self.dst_agg)
        # node_grad += scatter(grad, col, reduce='mean')
        num_nodes = len(self.node_idx)
        # the inverse operation of dropping topk least gradient nodes
        val, idx = torch.topk(node_grad, k=int(((1.0 - rate) * num_nodes)))
        sampled_node_idx = self.node_idx[idx]
        # add new edges by random walk or node sampler
        # new_node_idx = self._node_sampler(rate)
        new_node_idx = self._new_random_walk_sampler(rate)
        node_idx = torch.cat((sampled_node_idx, new_node_idx), dim=0).view(-1).unique()
        # record the sampled_node_idx for computing the coverage rate
        self.log_sampled_node_idx = (
            torch.cat((self.log_sampled_node_idx, node_idx), dim=0).view(-1).unique()
        )
        self.log_num_nodes = len(self.log_sampled_node_idx)

        self.sampled_adj = subgraph(self.ADJ, node_idx)
        self.node_idx = node_idx

    def _random_update(self, rate):
        num_nodes = len(self.node_idx)
        idx = torch.randperm(num_nodes, dtype=torch.long)[
            : int((1.0 - rate) * num_nodes)
        ]
        sampled_node_idx = self.node_idx[idx]
        # add new edges by random walk or node sampler
        # new_node_idx = self._node_sampler(rate)
        new_node_idx = self._new_random_walk_sampler(rate)
        node_idx = torch.cat((sampled_node_idx, new_node_idx), dim=0).view(-1).unique()
        # record the sampled_node_idx for computing the coverage rate
        self.log_sampled_node_idx = (
            torch.cat((self.log_sampled_node_idx, node_idx), dim=0).view(-1).unique()
        )
        self.log_num_nodes = len(self.log_sampled_node_idx)

        self.sampled_adj = subgraph(self.ADJ, node_idx)
        self.node_idx = node_idx

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def train_net(self, input_dict):
        device = input_dict["device"]
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]
        epoch = input_dict["epoch"]
        total_loss = total_correct = 0

        x, y = input_dict["x"], input_dict["y"]
        # x, y = x[self.log_sampled_node_idx], y[self.log_sampled_node_idx]
        x, y = x.to(device), y.to(device)

        for i in range(self.num_steps):
            optimizer.zero_grad()
            batch_x = x[self.node_idx]
            batch_y = y[self.node_idx]
            batch_train_idx = self.train_idx[self.node_idx]
            # forword propagation
            out = self(batch_x, self.sampled_adj.to(device))
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
                loss = loss_op(out[batch_train_idx], batch_y[batch_train_idx])
            else:
                loss = loss_op(out[batch_train_idx], batch_y[batch_train_idx])

            t = epoch * self.num_steps + i
            if t % self.dst_update_interval == 0 and t <= self.dst_T_end:
                self.sampled_adj.storage._value.retain_grad()
                loss.backward()
            else:
                loss.backward(retain_graph=True)
            # update
            self.logging()
            if t % self.dst_update_interval == 0 and t <= self.dst_T_end:
                rate = (self.dst_update_rate / 2) * (
                    1 + math.cos(t * math.pi / self.dst_T_end)
                )
                self.update_rate = rate
                if self.dst_update_scheme == "edge":
                    self._update_with_edge_grad_2(rate)
                elif self.dst_update_scheme == "node":
                    self._update_with_node_grad(rate)
                elif self.dst_update_scheme == "node2":
                    self._update_with_node_grad_2(rate)
                elif self.dst_update_scheme == "node3":
                    self._update_with_node_grad_3(rate)
                else:
                    self._random_update(rate)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            optimizer.step()

            total_loss += float(loss.item())
            if isinstance(loss_op, torch.nn.NLLLoss):
                total_correct += int(out.argmax(dim=-1).eq(batch_y).sum())
            else:
                total_correct += int(out.eq(batch_y).sum())

        train_size = (
            self.train_size
            if isinstance(loss_op, torch.nn.NLLLoss)
            else self.train_size * self.num_classes
        )
        return total_loss, total_correct / train_size

    def mem_speed_bench(self, input_dict):
        device = input_dict["device"]
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]
        epoch = input_dict["epoch"]
        model_opt_usage = get_memory_usage(0, False)
        usage_dict = {
            "model_opt_usage": model_opt_usage,
            "data_mem": [],
            "act_mem": [],
            "peak_mem": [],
            "duration": [],
        }
        print(
            "model + optimizer only, mem: %.2f MB"
            % (usage_dict["model_opt_usage"] / MB)
        )
        x, y = input_dict["x"], input_dict["y"]
        x, y = x.to(device), y.to(device)
        init_mem = get_memory_usage(0, False)
        data_mem = init_mem - usage_dict["model_opt_usage"]
        usage_dict["data_mem"].append(data_mem)
        print("data mem: %.2f MB" % (data_mem / MB))
        torch.cuda.empty_cache()
        epoch_start_time = time.time()
        torch.cuda.synchronize()
        for i in range(self.num_steps):
            iter_start_time = time.time()
            torch.cuda.synchronize()
            optimizer.zero_grad()
            batch_x = x[self.node_idx]
            batch_y = y[self.node_idx]
            batch_train_idx = self.train_idx[self.node_idx]
            # forword propagation
            out = self(batch_x, self.sampled_adj.to(device))
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
                loss = loss_op(out[batch_train_idx], batch_y[batch_train_idx])
            else:
                loss = loss_op(out[batch_train_idx], batch_y[batch_train_idx])
            before_backward = get_memory_usage(0, False)
            act_mem = before_backward - init_mem - compute_tensor_bytes([loss, out])
            usage_dict["act_mem"].append(act_mem)
            print("act mem: %.2f MB" % (act_mem / MB))
            t = epoch * self.num_steps + i
            if t % self.dst_update_interval == 0 and t <= self.dst_T_end:
                self.sampled_adj.storage._value.retain_grad()
                loss.backward()
            else:
                loss.backward(retain_graph=True)
            # update
            self.logging()
            if t % self.dst_update_interval == 0 and t <= self.dst_T_end:
                rate = (self.dst_update_rate / 2) * (
                    1 + math.cos(t * math.pi / self.dst_T_end)
                )
                self.update_rate = rate
                # self._update_with_node_grad(rate)
                self._random_update(rate)
            optimizer.step()
            torch.cuda.synchronize()
            iter_end_time = time.time()
            duration = iter_end_time - iter_start_time
            print("duration: %.4f sec" % duration)
            usage_dict["duration"].append(duration)
            peak_usage = torch.cuda.max_memory_allocated(0)
            usage_dict["peak_mem"].append(peak_usage)
            print(f"peak mem usage: {peak_usage / MB}")
            torch.cuda.empty_cache()
            del out, loss, batch_x, batch_y
        with open("./dstgcn_mem_speed_log.json", "w") as fp:
            info_dict = {**self.saved_args, **usage_dict}
            del info_dict["device"]
            # import pdb; pdb.set_trace()
            json.dump(info_dict, fp)
        exit()
