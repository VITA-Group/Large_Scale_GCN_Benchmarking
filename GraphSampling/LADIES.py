#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
import time
from re import L
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch._C import _init_names
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader.utils import filter_data, to_csc
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import EdgeType, NodeType
from torch_sparse import SparseTensor, matmul

from utils import GB, MB, compute_tensor_bytes, get_memory_usage

from ._GraphSampling import _GraphSampling
from .LayerWiseSampler import LADIESSampler


class FASTConv(MessagePassing):
    """NOTE: Implemented from torch_geometric.nn.SAGEConv"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: SparseTensor):
        out = self.propagate(edge_index, x=x)
        out = self.lin(out)
        return out

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce="mean")


class LADIES(_GraphSampling):
    def __init__(self, args, data, train_idx, processed_dir):
        super(LADIES, self).__init__(args, data, train_idx, processed_dir)

        self.num_steps = args.num_steps
        self.convs = torch.nn.ModuleList()
        self.convs.append(FASTConv(self.num_feats, self.dim_hidden))
        for _ in range(self.num_layers - 2):
            self.convs.append(FASTConv(self.dim_hidden, self.dim_hidden))
        self.convs.append(FASTConv(self.dim_hidden, self.num_classes))
        num_neighbors = [self.batch_size] * self.num_layers
        #  num_neighbors = [self.batch_size, int(self.batch_size/2)]
        self.train_loader = LADIESSampler(
            data.edge_index,
            node_idx=train_idx,
            sizes=num_neighbors,
            num_steps=args.num_steps,
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
        for i, (edge_index, _, size) in enumerate(adjs):
            # x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i](x, edge_index)
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
        for i, (input_idx, output_idx, adjs) in enumerate(self.train_loader):
            if i >= self.num_steps:
                break
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]

            optimizer.zero_grad()
            out = self(x[input_idx], adjs)
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
            loss = loss_op(out, y[output_idx])
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            if isinstance(loss_op, torch.nn.NLLLoss):
                total_correct += int(out.argmax(dim=-1).eq(y[output_idx]).sum())
            else:
                total_correct += int(out.eq(y[output_idx]).sum())

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
        usage_dict = {'model_opt_usage': model_opt_usage, 'data_mem': [], 'act_mem': [], 'peak_mem': [], 'duration': []}
        print('model + optimizer only, mem: %.2f MB' % (usage_dict['model_opt_usage'] / MB))

        x = input_dict["x"].to(device)
        y = input_dict["y"].to(device)
        init_mem = get_memory_usage(0, False)
        data_mem = init_mem - usage_dict['model_opt_usage']
        usage_dict['data_mem'].append(data_mem)
        print('data mem: %.2f MB' % (data_mem / MB))

        torch.cuda.empty_cache()
        epoch_start_time = time.time()
        torch.cuda.synchronize()
        for i, (input_idx, output_idx, adjs) in enumerate(self.train_loader):
            if i >= self.num_steps:
                break
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            iter_start_time = time.time()
            torch.cuda.synchronize()
            adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()
            out = self(x[input_idx], adjs)
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
            if out.shape[0] != len(output_idx):
                raise ValueError("shape not matched")
            print(f'num_sampled_nodes: {out.shape[0]}')
            loss = loss_op(out, y[output_idx])
            before_backward = get_memory_usage(0, False)
            act_mem = before_backward - init_mem - compute_tensor_bytes([loss, out])
            usage_dict['act_mem'].append(act_mem)
            print('act mem: %.2f MB' % (act_mem / MB))
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            iter_end_time = time.time()
            duration = iter_end_time - iter_start_time
            print('duration: %.4f sec' % duration)
            usage_dict['duration'].append(duration)
            peak_usage = torch.cuda.max_memory_allocated(0)
            usage_dict['peak_mem'].append(peak_usage)
            print(f'peak mem usage: {peak_usage / MB}')
            torch.cuda.empty_cache()
            del adjs, input_idx, output_idx, loss, out

        with open('./%s_ladies_mem_speed_log.json' % (self.saved_args['dataset']), 'w') as fp:
            info_dict = {**self.saved_args, **usage_dict}
            del info_dict['device']
            json.dump(info_dict, fp)
        exit()
