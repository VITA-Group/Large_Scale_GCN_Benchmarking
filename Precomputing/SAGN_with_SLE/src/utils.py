import gc
import os
import random
from operator import mul

import dgl
import dgl.function as fn
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

eps = 1e-9

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def clear_memory(device):
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

def entropy(probs):
    res = - probs * torch.log(probs + eps) - (1 - probs) * torch.log(1 - probs + eps)
    return res

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def to_scipy(tensor):
    """Convert a sparse tensor to scipy matrix"""
    values = tensor._values()
    indices = tensor._indices()
    return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)

def from_scipy(sparse_mx):
    """Convert a scipy sparse matrix to sparse tensor"""
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def compute_spectral_emb(adj, K):
    A = to_scipy(adj.to("cpu"))
    L = from_scipy(sp.csgraph.laplacian(A, normed=True))
    _, spectral_emb = torch.lobpcg(L, K)
    return spectral_emb.to(adj.device)

def outer_distance(y1, y2, train_mask):
    y1[y1 == 0] = eps
    y2[y2 == 0] = eps
    y1 = F.normalize(y1, p=1, dim=1)
    y2 = F.normalize(y2, p=1, dim=1)
    d = (y1[train_mask] * torch.log(y1[train_mask]) - y1[train_mask] * torch.log(y2[train_mask])).sum(dim=-1).mean(0)
    return d

def inner_distance(y, train_mask):
    y[y == 0] = eps
    y = F.normalize(y, p=1, dim=1)
    d = (y[train_mask] * torch.log(y[train_mask])).sum(dim=-1).mean(0) 
    if (~train_mask).sum()>0:
        d = d - (y[~train_mask] * torch.log(y[~train_mask])).sum(dim=-1).mean(0)
    return d

def calculate_homophily(g, labels, K=1, method="edge", multilabels=False, heterograph=False):
    assert method in ["edge", "node"]
    if multilabels:
        assert len(labels.shape) == 2
    else:
        if (labels.max() == 1) and len(labels.shape) > 1:
            labels = labels.argmax(dim=1)
    if heterograph:
        target_mask = g.ndata['target_mask']
        target_ids = g.ndata[dgl.NID][target_mask]
        num_target = target_mask.sum().item()
        g = g.subgraph(np.arange(g.number_of_nodes())[target_mask])
    g = dgl.khop_graph(g, K)
    src, dst = g.edges()
    # if multilabels:
    #     out = 0
    #     for c in labels.shape[1]:
    #         mask = (labels[src, c])
    mask = (labels[src] == labels[dst]).float()
    if method == "edge":
        out = mask.mean(dim=0)
    elif method == "node":
        g.edata["mask"] = mask
        g.update_all(fn.copy_e("mask", "m"), fn.mean("m", "out"))
        out = g.ndata.pop("out").mean(dim=0)
    # for multilabels, we average homophily across labels

    return out.mean(0).item()

def calculate_customized_homophily(g, labels, K, multilabels=False):
    if (not multilabels) and labels.max() > 1:
        y = torch.zeros(size=(len(labels), labels.max()+1))
        y[labels] = 1
    else:
        y = labels
    g.ndata['y'] = y.clone()
    for k in range(K):
        g.update_all(fn.copy_u('y', 'm'), fn.mean('m', 'y'))
    
    y_new = g.ndata.pop('y')
    y_new = F.normalize(y_new, dim=1, p=1)
    out = y_new[labels.long()].mean(0)
    return out.mean(0)

def read_subset_list(name, dir):
    print("Reading Relation Subsets:")
    if name == "ogbn-mag":
        name = "mag"
    fname = os.path.join(dir, name)

    rel_subsets = []
    with open(fname) as f:
        for line in f:
            relations = tuple(line.strip().split(','))
            rel_subsets.append(relations)
            print(relations)
    return rel_subsets

def generate_subset_list(g, num_subsets, target_ntype="paper"):
    edges = {e:(u,v) for u,v,e in g.metagraph().edges}
    print(edges)
    all_relations = list(edges.keys())
    subset_list = []
    while len(subset_list) < num_subsets:
        touched = False
        candidate = []
        for relation in all_relations:
            p = np.random.rand()
            if p >= 0.5:
                candidate.append(relation)
                if target_ntype in edges[relation]:
                    touched = True
        if touched:
            candidate = tuple(candidate)
            if candidate not in subset_list:
                subset_list.append(candidate)
    return subset_list
