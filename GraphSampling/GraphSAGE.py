import json
import time

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

from utils import GB, MB, compute_tensor_bytes, get_memory_usage

from ._GraphSampling import _GraphSampling


class GraphSAGE(_GraphSampling):
    def __init__(self, args, data, train_idx, processed_dir):
        super(GraphSAGE, self).__init__(args, data, train_idx, processed_dir)

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(self.num_feats, self.dim_hidden))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(self.dim_hidden, self.dim_hidden))
        self.convs.append(SAGEConv(self.dim_hidden, self.num_classes))
        num_neighbors = [25, 10, 5, 5, 5, 5, 5, 5, 5]
        self.train_loader = NeighborSampler(
            data.edge_index,
            node_idx=train_idx,
            sizes=num_neighbors[: self.num_layers],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12,
        )

        self.saved_args = vars(args)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def train_net(self, input_dict):

        device = input_dict["device"]
        x = input_dict["x"].to(device)
        y = input_dict["y"].to(device)
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]

        total_loss = total_correct = 0
        for batch_size, n_id, adjs in self.train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()
            out = self(x[n_id], adjs)
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
            loss = loss_op(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            if isinstance(loss_op, torch.nn.NLLLoss):
                total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
            else:
                total_correct += int(out.eq(y[n_id[:batch_size]]).sum())

        train_size = (
            self.train_size
            if isinstance(loss_op, torch.nn.NLLLoss)
            else self.train_size * self.num_classes
        )
        return total_loss / len(self.train_loader), total_correct / train_size

    def mem_speed_bench(self, input_dict):
        torch.cuda.empty_cache()
        device = input_dict["device"]

        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]
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

        x = input_dict["x"].to(device)
        y = input_dict["y"].to(device)
        init_mem = get_memory_usage(0, False)
        data_mem = init_mem - usage_dict["model_opt_usage"]
        usage_dict["data_mem"].append(data_mem)
        print("data mem: %.2f MB" % (data_mem / MB))

        torch.cuda.empty_cache()
        epoch_start_time = time.time()
        torch.cuda.synchronize()
        for i, (batch_size, n_id, adjs) in enumerate(self.train_loader):
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            iter_start_time = time.time()
            torch.cuda.synchronize()
            adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()
            out = self(x[n_id], adjs)
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
            print(f'num_sampled_nodes: {out.shape[0]}')
            loss = loss_op(out, y[n_id[:batch_size]])
            before_backward = get_memory_usage(0, False)
            act_mem = before_backward - init_mem - compute_tensor_bytes([loss, out])
            usage_dict["act_mem"].append(act_mem)
            print("act mem: %.2f MB" % (act_mem / MB))
            loss.backward()
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
            del adjs, batch_size, n_id, loss, out

        with open(
            "./%s_graphsage_mem_speed_log.json" % (self.saved_args["dataset"]), "w"
        ) as fp:
            info_dict = {**self.saved_args, **usage_dict}
            del info_dict["device"]
            json.dump(info_dict, fp)
        exit()
