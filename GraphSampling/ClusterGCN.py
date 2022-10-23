import json
import time

import torch
import torch.nn.functional as F
from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.nn import SAGEConv

from utils import GB, MB, compute_tensor_bytes, get_memory_usage

from ._GraphSampling import _GraphSampling



class ClusterGCN(_GraphSampling):
    # Implemented base on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/graph_saint.py

    def __init__(self, args, data, train_idx, processed_dir):
        super(ClusterGCN, self).__init__(args, data, train_idx, processed_dir)
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(self.num_feats, self.dim_hidden))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(self.dim_hidden, self.dim_hidden))
        self.convs.append(SAGEConv(self.dim_hidden, self.num_classes))
        self.reset_parameters()

        sample_size = max(1, int(args.batch_size / (data.num_nodes / args.num_parts)))
        cluster_data = ClusterData(
            data, num_parts=args.num_parts, recursive=False, save_dir=self.save_dir
        )
        self.train_loader = ClusterLoader(
            cluster_data, batch_size=sample_size, shuffle=True
        )
        self.saved_args = vars(args)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def train_net(self, input_dict):

        device = input_dict["device"]
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]
        total_loss = total_correct = 0
        for batch in self.train_loader:
            batch = batch.to(device)
            if batch.train_mask.sum() == 0:
                continue
            optimizer.zero_grad()
            out = self(batch.x, batch.edge_index)
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
                loss = loss_op(out[batch.train_mask], batch.y[batch.train_mask])
            else:
                loss = loss_op(
                    out[batch.train_mask], batch.y[batch.train_mask].type_as(out)
                )
            loss.backward()
            optimizer.step()
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
        return total_loss / len(self.train_loader), total_correct / train_size

    def mem_speed_bench(self, input_dict):
        torch.cuda.empty_cache()
        device = input_dict["device"]
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]
        torch.cuda.empty_cache()
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
        epoch_start_time = time.time()
        torch.cuda.synchronize()
        for batch in self.train_loader:
            iter_start_time = time.time()
            torch.cuda.synchronize()
            optimizer.zero_grad()
            batch = batch.to(device)
            init_mem = get_memory_usage(0, False)
            data_mem = init_mem - usage_dict["model_opt_usage"]
            usage_dict["data_mem"].append(data_mem)
            print("---> num_sampled_nodes: {}".format(batch.x.shape[0]))
            print("data mem: %.2f MB" % (data_mem / MB))
            out = self(batch.x, batch.edge_index)
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
            loss = loss_op(out[batch.train_mask], batch.y[batch.train_mask])
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
            del out, loss, batch
        with open(
            "./%s_clustergcn_mem_speed_log.json" % (self.saved_args["dataset"]), "w"
        ) as fp:
            info_dict = {**self.saved_args, **usage_dict}
            del info_dict["device"]
            json.dump(info_dict, fp)
        exit()
