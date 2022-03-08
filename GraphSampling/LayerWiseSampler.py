from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import WeightedRandomSampler
from torch_sparse import SparseTensor


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class LayerWiseSampler(torch.utils.data.DataLoader):
    def __init__(
        self,
        edge_index: Tensor,
        sizes: List[int],
        num_steps: int,
        node_idx: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        return_e_id: bool = False,
        transform: Callable = None,
        **kwargs
    ):

        edge_index = edge_index.to("cpu")

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]
        if "dataset" in kwargs:
            del kwargs["dataset"]

        # Save for Pytorch Lightning...
        self.edge_index = edge_index
        self.node_idx = node_idx
        self.num_nodes = num_nodes
        self.num_steps = num_steps

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = False
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if (
                num_nodes is None
                and node_idx is not None
                and node_idx.dtype == torch.bool
            ):
                num_nodes = node_idx.size(0)
            if (
                num_nodes is None
                and node_idx is not None
                and node_idx.dtype == torch.long
            ):
                num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1

            # NOTE: add self-loop here
            self.adj_t = SparseTensor(
                row=torch.cat([edge_index[0], torch.arange(num_nodes)], dim=0),
                col=torch.cat([edge_index[1], torch.arange(num_nodes)], dim=0),
                value=None,
                sparse_sizes=(num_nodes, num_nodes),
            ).t()

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = torch.nonzero(node_idx, as_tuple=False).view(-1)

        #! here we use 1/D(v) as the results of l2 norm of $\hat{A}(:, v)$ since we uniformly use GraphSAGE (where $\hat{A}(:, v)$ = D^{-1}A{:, v})
        col_norm = 1.0 / self.adj_t.sum(dim=0)
        self.probs = col_norm / col_norm.sum()

        super(LayerWiseSampler, self).__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs
        )

    def __len__(self):
        return self.num_steps

    def sample(self, batch: Tensor):

        batch_size: int = len(batch)

        adjs, n_ids = [], []
        n_id = batch
        for size in self.sizes:
            n_ids.append(n_id)
            adj_t, n_id = self._one_layer_sampling(n_id, size)
            #  adj_t = adj.t()
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout="coo")

            adjs.append(Adj(adj_t, e_id, len(n_id)))
        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        # n_ids = n_ids[0] if len(n_ids) == 1 else n_ids[::-1]
        out = (n_id, batch, adjs)  # (input_idx, output_idx, adjs)
        out = self.transform(*out) if self.transform is not None else out
        return out

    def _one_layer_sampling(self, n_id, size):
        raise NotImplementedError

    def __repr__(self):
        return "{}(sizes={})".format(self.__class__.__name__, self.sizes)


class FastGCNSampler(LayerWiseSampler):
    def __init__(self, *args, **kwargs):
        super(FastGCNSampler, self).__init__(*args, **kwargs)

    def _one_layer_sampling(self, n_id, size):
        """
        Implemented based on https://github.com/Gkunnan97/FastGCN_pytorch/blob/gpu/sampler.py
        """
        adj = self.adj_t[n_id, :]
        nidx = torch.nonzero(adj.sum(dim=0)).squeeze()
        p = self.probs[nidx]
        p = p / p.sum()
        idx = list(WeightedRandomSampler(p, size, replacement=False))
        sampled_nidx = nidx[idx]
        # preserve n_id
        # sampled_nidx = torch.cat([n_id, sampled_nidx], dim=0).unique()
        adj = adj[:, sampled_nidx]
        return adj, sampled_nidx


class LADIESSampler(LayerWiseSampler):
    def __init__(self, *args, **kwargs):
        super(LADIESSampler, self).__init__(*args, **kwargs)

    def _one_layer_sampling(self, n_id, size):
        adj = self.adj_t[n_id, :]
        nidx = torch.nonzero(adj.sum(dim=0)).squeeze()

        p = self.probs[nidx]
        p = p / p.sum()
        idx = list(WeightedRandomSampler(p, size, replacement=False))
        sampled_nidx = nidx[idx]
        # preserve n_id
        if isinstance(n_id, list):
            n_id = torch.tensor(n_id)
        sampled_nidx = torch.cat([n_id, sampled_nidx], dim=0).unique()
        adj = adj[:, sampled_nidx]
        return adj, sampled_nidx
