import torch
import torch.nn.functional as F
from torch_geometric.transforms import SIGN
from torch.utils.data import DataLoader

import torch
from torch_sparse import SparseTensor

from Precomputing.base import PrecomputingBase

class SGC(PrecomputingBase):
    def __init__(self, args, data, train_idx, processed_dir):
        super(SGC, self).__init__(args, data, train_idx, processed_dir)

        self.lin = torch.nn.Linear(self.num_feats, self.num_classes)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, xs):
        return self.lin(xs[-1])
