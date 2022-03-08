import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

import torch_sparse

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.utils.convert import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes

import numba
import numpy as np
import scipy.sparse as sp

import math

@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)

    return list(p.keys()), list(p.values())


@numba.njit(cache=True)
def calc_ppr(indptr, indices, deg, alpha, epsilon, nodes):
    js = []
    vals = []
    for i, node in enumerate(nodes):
        j, val = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        js.append(j)
        vals.append(val)
    return js, vals

@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk):
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        j_np, val_np = np.array(j), np.array(val)
        idx_topk = np.argsort(val_np)[-topk:]
        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]
    return js, vals

def ppr_topk(adj_matrix, alpha, epsilon, nodes, topk):
    """Calculate the PPR matrix approximately using Anderson."""

    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]

    neighbors, weights = calc_ppr_topk_parallel(adj_matrix.indptr, adj_matrix.indices, out_degree,
                                                numba.float32(alpha), numba.float32(epsilon), nodes, topk)

    return construct_sparse(neighbors, weights, (len(nodes), nnodes))

def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)

class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        value_dropped = F.dropout(input.storage.value(), self.p, self.training)
        return torch_sparse.SparseTensor(
                row=input.storage.row(), rowptr=input.storage.rowptr(), col=input.storage.col(),
                value=value_dropped, sparse_sizes=input.sparse_sizes(), is_sorted=True)


class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if isinstance(input, torch_sparse.SparseTensor):
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if isinstance(input, torch_sparse.SparseTensor):
            res = input.matmul(self.weight)
            if self.bias:
                res += self.bias[None, :]
        else:
            if self.bias:
                res = torch.addmm(self.bias, input, self.weight)
            else:
                res = input.matmul(self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)

def matrix_to_torch(X):
    if sp.issparse(X):
        return torch_sparse.SparseTensor.from_scipy(X)
    else:
        return torch.FloatTensor(X)

class TopkPPR(nn.Module):

    def __init__(self, alpha, eps, k, normalization):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.topk = k
        self.normalization = normalization

    def forward(self, data, idx=None):
        """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""
        
        adj_matrix = to_scipy_sparse_matrix(data['edge_index'], edge_attr=data['edge_weight']).tocsr()
        if idx is None:
            N = maybe_num_nodes(data['edge_index'])
            idx = np.arange(N)

        topk_matrix = ppr_topk(adj_matrix, self.alpha, self.eps, idx, self.topk).tocsr()

        if self.normalization == 'sym':
            # Assume undirected (symmetric) adjacency matrix
            deg = adj_matrix.sum(1).A1
            deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
            deg_inv_sqrt = 1. / deg_sqrt

            row, col = topk_matrix.nonzero()
            # assert np.all(deg[idx[row]] > 0)
            # assert np.all(deg[col] > 0)
            topk_matrix.data = deg_sqrt[idx[row]] * topk_matrix.data * deg_inv_sqrt[col]
        elif self.normalization == 'col':
            # Assume undirected (symmetric) adjacency matrix
            deg = adj_matrix.sum(1).A1
            deg_inv = 1. / np.maximum(deg, 1e-12)

            row, col = topk_matrix.nonzero()
            # assert np.all(deg[idx[row]] > 0)
            # assert np.all(deg[col] > 0)
            topk_matrix.data = deg[idx[row]] * topk_matrix.data * deg_inv[col]
        elif self.normalization == 'row':
            pass
        else:
            raise ValueError(f"Unknown PPR normalization: {normalization}")

        data['edge_index'], data['edge_weight'] = from_scipy_sparse_matrix(topk_matrix)

        return data

class PPRGoMLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, nlayers, dropout):
        super().__init__()

        fcs = [MixedLinear(num_features, hidden_size, bias=False)]
        for i in range(nlayers - 2):
            fcs.append(nn.Linear(hidden_size, hidden_size, bias=False))
        fcs.append(nn.Linear(hidden_size, num_classes, bias=False))
        self.fcs = nn.ModuleList(fcs)

        self.drop = MixedDropout(dropout)

    def forward(self, X):
        embs = self.drop(X)
        embs = self.fcs[0](embs)
        for fc in self.fcs[1:]:
            embs = fc(self.drop(F.relu(embs)))
        return embs

class PPRGo(MessagePassing):
    def __init__(self, args, data, train_idx):
        super().__init__()
        
        self.num_layers = args.num_layers
        self.dim_hidden = args.dim_hidden
        self.num_classes = args.num_classes
        self.num_feats = args.num_feats
        self.batch_size = args.batch_size
        self.dropout = args.dropout

        # self.data = TopkPPR(args.alpha, args.eps, args.topk, normalization='row')(data)
        self.data = TopkPPR(0.5, 1e-4, 32, normalization='row')(data)

        self.mlp = PPRGoMLP(self.num_feats, self.num_classes, self.dim_hidden, self.num_layers, self.dropout)

    def forward(self, x, edge_index, edge_weight):
        logits = self.mlp(x)
        # print(edge_index.shape, edge_index.max(), edge_index.min(), edge_weight.shape, x.shape)
        # raise
        propagated_logits = self.propagate(edge_index, x=logits, edge_weight=edge_weight, size=None)
        return propagated_logits
    
    def train_net(self, input_dict):
        split_idx = input_dict['split_masks']
        device = input_dict['device']
        y = input_dict['y']
        optimizer = input_dict['optimizer']
        loss_op = input_dict['loss_op']
        x_train = self.data['x'].to(device)
        y_train_true = y[split_idx['train']].to(device)

        optimizer.zero_grad()
        out = self.forward(x_train, self.data['edge_index'].to(device), self.data['edge_weight'].to(device))
        out = out[split_idx['train']]
        if isinstance(loss_op, torch.nn.NLLLoss):
            out = F.log_softmax(out, dim=-1)
        loss = loss_op(out, y_train_true)

        loss = loss.mean()
        loss.backward()
        optimizer.step()

        if isinstance(loss_op, torch.nn.NLLLoss):
            total_correct = int(out.argmax(dim=-1).eq(y_train_true).sum())
            train_acc = float(total_correct / y_train_true.size(0))
        else:
            total_correct = int(out.eq(y_train_true).sum())
            train_acc = float(total_correct / (y_train_true.size(0) * self.num_classes))

        return float(loss.item()), train_acc

    def inference(self, input_dict):
        x_all = input_dict['x']
        device = input_dict['device']
        y_preds = []
        loader = torch.utils.data.DataLoader(range(x_all.size(0)), batch_size=100000)
        for perm in loader:
            y_pred = self.forward([x[perm].to(device) for x in x_all])
            y_preds.append(y_pred.cpu())
        y_preds = torch.cat(y_preds, dim=0)

        return y_preds
