
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.transforms import SIGN
from torch.utils.data import DataLoader

from Precomputing.base import PrecomputingBase

# adapted from https://github.com/chennnM/GBP
class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias='bn'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.bias(output)
        #output = F.normalize(output,p=2,dim=1)
        if self.in_features == self.out_features:
            output = output + input
        return output


# MLP apply initial residual
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.alpha=0.5
        self.reset_parameters()
        self.bias = nn.BatchNorm1d(out_features)
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input ,h0):
        support = (1-self.alpha)*input+self.alpha*h0
        output = torch.mm(support, self.weight)
        output=self.bias(output)
        #output=F.normalize(output,p=2,dim=1)
        if self.in_features==self.out_features:
            output = output+input
        return output


# adapted from dgl sign
class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        #self.norm=bns
        self.reset_parameters()
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers -1:
                #if self.norm:
                x = self.dropout(self.prelu(self.bns[layer_id](x)))
                #else:
                #    x = self.dropout(self.prelu(x))
        return x


class FeedForwardNetII(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNetII, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.in_feats=in_feats
        self.hidden=hidden
        self.out_feats=out_feats
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats,bias=False))
        else:
            self.layers.append(Dense(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(GraphConvolution(hidden, hidden))
            self.layers.append(Dense(hidden, out_feats))

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    def forward(self, x):
        x=self.layers[0](x)
        h0=x
        for layer_id, layer in enumerate(self.layers):
            if layer_id==0:
                continue
            elif layer_id== self.n_layers - 1:
                x = self.dropout(self.prelu(x))
                x = layer(x)
            else:
                x = self.dropout(self.prelu(x))
                x = layer(x,h0)
                #x = self.dropout(self.prelu(x))
        return x

class JK_GAMLP(PrecomputingBase):
    def __init__(self, args, data, train_idx, input_drop=0.0, att_dropout=0.0, label_drop=0.0, alpha=0.5, n_layers_1=2, n_layers_2=2, n_layers_3=2, n_layers_4=2, act='relu', pre_process=False, residual=False,use_label=False):
        super(JK_GAMLP, self).__init__(args, data, train_idx)

        num_hops = self.num_layers + 1
        label_num_hops = self.num_layers + 1
        nfeat = args.num_feats
        hidden = self.dim_hidden
        nclass = args.num_classes
        dropout = args.dropout

        self.num_hops = num_hops
        self.label_num_hops=label_num_hops
        self.use_label=use_label
        self.residual = residual
        self.prelu = nn.ReLU()
        self.alpha=alpha
        self.res_fc = nn.Linear(nfeat, hidden, bias=False)
        if pre_process:
            self.lr_jk_ref = FeedForwardNetII(
                num_hops*hidden, hidden, hidden, n_layers_1, dropout)
            self.lr_att = nn.Linear(hidden + hidden, 1)
            self.lr_output = FeedForwardNetII(
                hidden, hidden, nclass, n_layers_2, dropout)
            self.process = nn.ModuleList(
                [FeedForwardNet(nfeat, hidden, hidden, 2, dropout) for i in range(num_hops)])
        else:
            self.lr_jk_ref = FeedForwardNetII(
                num_hops*nfeat, hidden, hidden, n_layers_1, dropout)
            self.lr_att = nn.Linear(nfeat + hidden, 1)
            self.lr_output = FeedForwardNetII(
                nfeat, hidden, nclass, n_layers_2, dropout)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.att_drop = nn.Dropout(att_dropout)
        self.label_drop = nn.Dropout(label_drop)
        self.pre_process = pre_process
        if use_label:
            self.label_transform=nn.Parameter(torch.FloatTensor(nclass, nclass))
            self.label_jk_ref=FeedForwardNet(
                nclass*label_num_hops, hidden, nclass,n_layers_3, dropout)
            self.label_output = FeedForwardNet(
                nclass, hidden, nclass, n_layers_4, dropout)
            self.label_att=nn.Linear(nclass+nclass,1)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)


    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        self.lr_jk_ref.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, feature_list, label_list=[]):
        num_node = feature_list[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in feature_list]
        input_list = []
        if self.pre_process:
            for i in range(len(feature_list)):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list=feature_list
        concat_features = torch.cat(input_list, dim=1)
        jk_ref = self.dropout(self.prelu(self.lr_jk_ref(concat_features)))
        attention_scores = [self.act(self.lr_att(torch.cat((jk_ref, x), dim=1))).view(num_node, 1) for x in
                            input_list]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            W[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops):
            right_1 = right_1 + \
                torch.mul(input_list[i], self.att_drop(
                    W[:, i].view(num_node, 1)))
        if self.residual:
            right_1 += self.res_fc(feature_list[0])
            right_1 = self.dropout(self.prelu(right_1))
        right_1 = self.lr_output(right_1)
        if self.use_label:
            label_list = [self.label_drop(feature) for feature in label_list]
            #print(label_list[0])
            for i in range(1,len(label_list)):
                #label_list=(1-self.alpha)*label_list[i]+self.alpha*F.softmax(torch.mm(label_list[i],self.label_transform),dim=1)
                alpha=np.cos(i*np.pi/(self.label_num_hops*2))
                label_list[i]=(1-alpha)*label_list[i]+alpha*label_list[-1]
            input_list = label_list
            concat_features = torch.cat(input_list, dim=1)
            jk_ref = self.dropout(self.prelu(self.label_jk_ref(concat_features)))
            attention_scores = [self.act(self.label_att(torch.cat((jk_ref, x), dim=1))).view(num_node, 1) for x in
                                input_list]
            W = torch.cat(attention_scores, dim=1)
            W = F.softmax(W, 1)
            right_2 = torch.mul(input_list[0], self.att_drop(
                W[:, 0].view(num_node, 1)))
            for i in range(1, self.label_num_hops):
                right_2 = right_2 + \
                    torch.mul(input_list[i], self.att_drop(
                        W[:, i].view(num_node, 1)))
            right_2 = self.label_output(right_2)
            right_1 = right_1+right_2
        return right_1
