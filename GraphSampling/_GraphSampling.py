import os

import torch
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler


class _GraphSampling(torch.nn.Module):
    # Implemented base on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/graph_saint.py
    def __init__(self, args, data, train_idx, processed_dir):
        super(_GraphSampling, self).__init__()

        self.type_model = args.type_model
        self.num_layers = args.num_layers
        self.dim_hidden = args.dim_hidden
        self.num_classes = args.num_classes
        self.num_feats = args.num_feats
        self.batch_size = args.batch_size
        self.train_size = train_idx.size(0)
        self.dropout = args.dropout
        self.train_idx = train_idx
        self.save_dir = processed_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.debug_mem_speed = args.debug_mem_speed
        self.test_loader = NeighborSampler(
            data.edge_index, sizes=[-1], batch_size=1024, shuffle=False
        )

    def inference(self, input_dict):
        device = input_dict["device"]
        x_all = input_dict["x"]
        for i, conv in enumerate(self.convs):
            xs = []
            for _, n_id, adj in self.test_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[: size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all
