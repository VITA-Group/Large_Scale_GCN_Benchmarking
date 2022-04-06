import os
from copy import deepcopy

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj, to_undirected
from torch_scatter import scatter
from torch_sparse import SparseTensor
from tqdm import tqdm

np.random.seed(0)


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        relu_first=True,
    ):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.relu_first = relu_first

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.relu_first:
                x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            if not self.relu_first:
                x = F.relu(x, inplace=True)

            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)


class MLPLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPLinear, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        return F.log_softmax(self.lin(x), dim=-1)


def sgc(x, adj, num_propagations):
    for _ in tqdm(range(num_propagations)):
        x = adj @ x
    return torch.from_numpy(x).to(torch.float)


def lp(adj, train_idx, labels, num_propagations, p, alpha, preprocess):
    if p is None:
        p = 0.6
    if alpha is None:
        alpha = 0.4

    c = labels.max() + 1
    idx = train_idx
    y = np.zeros((labels.shape[0], c))
    y[idx] = F.one_hot(labels[idx], c).numpy().squeeze(1)
    result = deepcopy(y)
    for i in tqdm(range(num_propagations)):
        result = y + alpha * adj @ (result ** p)
        result = np.clip(result, 0, 1)
    return torch.from_numpy(result).to(torch.float)


def diffusion(x, adj, num_propagations, p, alpha):
    if p is None:
        p = 1.0
    if alpha is None:
        alpha = 0.5

    inital_features = deepcopy(x)
    x = x ** p
    for i in tqdm(range(num_propagations)):
        #         x = (1-args.alpha)* inital_features + args.alpha * adj @ x
        x = x - alpha * (sparse.eye(adj.shape[0]) - adj) @ x
        x = x ** p
    return torch.from_numpy(x).to(torch.float)


def community(data, post_fix):
    print("Setting up community detection feature")
    np_edge_index = np.array(data.edge_index)

    G = nx.Graph()
    G.add_edges_from(np_edge_index.T)

    partition = community_louvain.best_partition(G)
    np_partition = np.zeros(data.num_nodes)
    for k, v in partition.items():
        np_partition[k] = v

    np_partition = np_partition.astype(np.int)

    n_values = int(np.max(np_partition) + 1)
    one_hot = np.eye(n_values)[np_partition]

    result = torch.from_numpy(one_hot).float()

    torch.save(result, f"LP/embeddings/community{post_fix}.pt")

    return result


def spectral(data, post_fix):
    from julia.api import Julia

    jl = Julia(compiled_modules=False)
    from julia import Main

    Main.include("LP/norm_spec.jl")
    print("Setting up spectral embedding")
    data.edge_index = to_undirected(data.edge_index)
    np_edge_index = np.array(data.edge_index.T)

    N = data.num_nodes
    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.to_scipy(layout="csr")
    result = torch.tensor(Main.main(adj, 128)).float()
    torch.save(result, f"LP/embeddings/spectral{post_fix}.pt")

    return result


def preprocess(
    data,
    preprocess="diffusion",
    num_propagations=10,
    p=None,
    alpha=None,
    use_cache=True,
    post_fix="",
):
    # use_cache = 0
    if use_cache:
        try:
            x = torch.load(f"LP/embeddings/{preprocess}{post_fix}.pt")
            print("Using cache")
            return x
        except:
            print(
                f"LP/embeddings/{preprocess}{post_fix}.pt not found or not enough iterations! Regenerating it now"
            )

    if preprocess == "community":
        return community(data, post_fix)

    if preprocess == "spectral":
        return spectral(data, post_fix)

    print("Computing adj...")
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    adj = adj.to_scipy(layout="csr")

    sgc_dict = {}

    print(f"Start {preprocess} processing")

    if preprocess == "sgc":
        result = sgc(data.x.numpy(), adj, num_propagations)
    #     if preprocess == "lp":
    #         result = lp(adj, data.y.data, num_propagations, p = p, alpha = alpha, preprocess = preprocess)
    if preprocess == "diffusion":
        result = diffusion(data.x.numpy(), adj, num_propagations, p=p, alpha=alpha)

    os.makedirs("LP/embeddings", exist_ok=1)
    torch.save(result, f"LP/embeddings/{preprocess}{post_fix}.pt")

    return result
