import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP

class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, args):
        super(GPRGNN, self).__init__()

        self.num_layers = args.num_layers
        self.dim_hidden = args.dim_hidden
        self.num_classes = args.num_classes
        self.num_feats = args.num_feats
        self.batch_size = args.batch_size
        self.dprate = args.dropout

        self.lin1 = Linear(self.num_feats, self.dim_hidden)
        self.lin2 = Linear(self.dim_hidden, self.num_classes)

        Init = args.GPR_init
        Gamma = None
        self.prop1 = GPR_prop(self.num_layers, args.GPR_alpha, Init, Gamma)

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dprate, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.lin2(x)

        if self.dprate != 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)

        x = self.prop1(x, edge_index)
        return x

    def train_net(self, input_dict):
        device = input_dict['device']
        split_idx = input_dict['split_masks']['train'].to(device)
        data = input_dict['data'].to(device)
        y = input_dict['y'].to(device)
        optimizer = input_dict['optimizer']
        loss_op = input_dict['loss_op']
        # xs_train = self.x[split_idx['train']].to(device)
        # y_train_true = y[split_idx['train']].to(device)

        optimizer.zero_grad()
        out = self.forward(data)[split_idx]
        if isinstance(loss_op, torch.nn.NLLLoss):
            out = F.log_softmax(out, dim=-1)
        loss = loss_op(out, y[[split_idx]])

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
        device = input_dict['device']
        split_idx = input_dict['split_masks']['test'].to(device)
        data = input_dict['data'].to(device)
        y_preds = self.forward(data)[split_idx].cpu()
        # if isinstance(loss_op, torch.nn.NLLLoss):
        #     out = F.log_softmax(out, dim=-1)

        # y_preds = []
        # loader = DataLoader(range(x_all.size(0)), batch_size=100000)
        # for perm in loader:
        #     y_pred = self.forward([x[perm].to(device) for x in self.xs])
        #     y_preds.append(y_pred.cpu())
        # y_preds = torch.cat(y_preds, dim=0)

        return y_preds
