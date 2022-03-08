import math
import os
import random
import time

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from layers import (MLP, FeedForwardNet, GroupMLP, MultiHeadBatchNorm,
                    MultiHeadMLP)


################################################################
# DGL's implementation of SIGN
class SIGN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, num_hops, n_layers,
                 dropout, input_drop):
        super(SIGN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = nn.Dropout(input_drop)
        for hop in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, n_layers, dropout))
        self.project = FeedForwardNet(num_hops * hidden, hidden, out_feats,
                                      n_layers, dropout)

    def forward(self, feats):
        feats = [self.input_drop(feat) for feat in feats]
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            hidden.append(ff(feat))
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        return out

    def reset_parameters(self):
        for ff in self.inception_ffs:
            ff.reset_parameters()
        self.project.reset_parameters()

################################################################
# SAGN model
class SAGN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, num_hops, n_layers, num_heads, weight_style="attention", alpha=0.5, focal="first",
                 hop_norm="softmax", dropout=0.5, input_drop=0.0, attn_drop=0.0, negative_slope=0.2, zero_inits=False, position_emb=False):
        super(SAGN, self).__init__()
        self._num_heads = num_heads
        self._hidden = hidden
        self._out_feats = out_feats
        self._weight_style = weight_style
        self._alpha = alpha
        self._hop_norm = hop_norm
        self._zero_inits = zero_inits
        self._focal = focal
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_drop)
        # self.bn = nn.BatchNorm1d(hidden * num_heads)
        self.bn = MultiHeadBatchNorm(num_heads, hidden * num_heads)
        self.relu = nn.ReLU()
        self.input_drop = nn.Dropout(input_drop)
        self.multihop_encoders = nn.ModuleList([GroupMLP(in_feats, hidden, hidden, num_heads, n_layers, dropout) for i in range(num_hops)])
        self.res_fc = nn.Linear(in_feats, hidden * num_heads, bias=False)
        
        if weight_style == "attention":
            self.hop_attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
            self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
            self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        if position_emb:
            self.pos_emb = nn.Parameter(torch.FloatTensor(size=(num_hops, in_feats)))
        else:
            self.pos_emb = None
        
        self.post_encoder = GroupMLP(hidden, hidden, out_feats, num_heads, n_layers, dropout)
        # self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for encoder in self.multihop_encoders:
            encoder.reset_parameters()
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        if self._weight_style == "attention":
            if self._zero_inits:
                nn.init.zeros_(self.hop_attn_l)
                nn.init.zeros_(self.hop_attn_r)
            else:
                nn.init.xavier_normal_(self.hop_attn_l, gain=gain)
                nn.init.xavier_normal_(self.hop_attn_r, gain=gain)
        if self.pos_emb is not None:
            nn.init.xavier_normal_(self.pos_emb, gain=gain)
        self.post_encoder.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, feats):
        out = 0
        feats = [self.input_drop(feat) for feat in feats]
        if self.pos_emb is not None:
            feats = [f +self.pos_emb[[i]] for i, f in enumerate(feats)]
        hidden = []
        for i in range(len(feats)):
            hidden.append(self.multihop_encoders[i](feats[i]).view(-1, self._num_heads, self._hidden))
        
        a = None
        if self._weight_style == "attention":
            if self._focal == "first":
                focal_feat = hidden[0]
            if self._focal == "last":
                focal_feat = hidden[-1]
            if self._focal == "average":
                focal_feat = 0
                for h in hidden:
                    focal_feat += h
                focal_feat /= len(hidden)
                
            astack_l = [(h * self.hop_attn_l).sum(dim=-1).unsqueeze(-1) for h in hidden]
            a_r = (focal_feat * self.hop_attn_r).sum(dim=-1).unsqueeze(-1)
            astack = torch.stack([(a_l + a_r) for a_l in astack_l], dim=-1)
            if self._hop_norm == "softmax":
                a = self.leaky_relu(astack)
                a = F.softmax(a, dim=-1)
            if self._hop_norm == "sigmoid":
                a = torch.sigmoid(astack)
            if self._hop_norm == "tanh":
                a = torch.tanh(astack)
            a = self.attn_dropout(a)
            
            for i in range(a.shape[-1]):
                out += hidden[i] * a[:, :, :, i]

        if self._weight_style == "uniform":
            for h in hidden:
                out += h / len(hidden)
        
        if self._weight_style == "exponent":
            for k, h in enumerate(hidden):
                out += self._alpha ** k * h

        out += self.res_fc(feats[0]).view(-1, self._num_heads, self._hidden)
        out = out.flatten(1, -1)
        out = self.dropout(self.relu(self.bn(out)))
        out = out.view(-1, self._num_heads, self._hidden)
        out = self.post_encoder(out)
        out = out.mean(1)
        
        return out, a.mean(1) if a is not None else None

# a simplified version of SAGN
class PlainSAGN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, num_heads, residual=True, pre_norm=False, hop_norm="softmax",
                 dropout=0.5, input_drop=0.0, attn_drop=0.0, negative_slope=0.2, zero_inits=False):
        super(PlainSAGN, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._hidden = hidden
        self._out_feats = out_feats
        self._pre_norm = pre_norm
        self._hop_norm = hop_norm
        self._zero_inits = zero_inits
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.bn = nn.BatchNorm1d(in_feats * num_heads)
        self.relu = nn.ReLU()
        self.input_drop = nn.Dropout(input_drop)
        self.hop_attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, in_feats)))
        self.hop_attn_0 = nn.Parameter(torch.FloatTensor(size=(1, num_heads, in_feats)))
        self.hop_attn_K = nn.Parameter(torch.FloatTensor(size=(1, num_heads, in_feats)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.post_encoder = GroupMLP(in_feats * num_heads, hidden, out_feats, num_heads, n_layers, dropout, residual=residual)

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        if self._zero_inits:
            nn.init.zeros_(self.hop_attn)
            nn.init.zeros_(self.hop_attn_0)
            nn.init.zeros_(self.hop_attn_K)
        else:
            nn.init.xavier_normal_(self.hop_attn, gain=gain)
            nn.init.xavier_normal_(self.hop_attn_0, gain=gain)
            nn.init.xavier_normal_(self.hop_attn_K, gain=gain)

        self.post_encoder.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, feats):
        out = 0
        hidden = []
        for i in range(len(feats)):
            hidden.append(feats[i].view(-1, 1, self._in_feats))
        if self._pre_norm:
            hidden = [F.normalize(feat, 2, 2) for feat in hidden]
        # norms = [torch.norm(feat, 2, 2).clamp(min=1e-9) for feat in hidden]
        # norm_attn = torch.norm(self.hop_attn).clamp(min=1e-9)

        astack = [(feat * self.hop_attn).sum(-1) for feat in hidden]
        # a_0 = (hidden[0] * self.hop_attn_0).sum(dim=-1)
        # a_K = (hidden[-1] * self.hop_attn_K).sum(dim=-1)
        astack = torch.stack([a for a in astack], dim=-1)
        
        if self._hop_norm == "softmax":
            a = self.leaky_relu(astack)
            a = F.softmax(a, dim=-1)
        if self._hop_norm == "sigmoid":
            a = torch.sigmoid(a)
        if self._hop_norm == "tanh":
            a = torch.tanh(a)

        a = self.attn_dropout(a)
        
        for i in range(a.shape[-1]):
            out += hidden[i] * a[:, :, [i]]
        out = out.flatten(1, -1)
        out = self.bn(out)
        # out = self.dropout(self.relu(self.bn(out)))
        out = self.post_encoder(out)
        out = out.mean(1)

        return out, a

################################################################
# NARS aggregator across subgraphs
class WeightedAggregator(nn.Module):
    def __init__(self, subset_list, in_feats, num_hops):
        super(WeightedAggregator, self).__init__()
        self.num_hops = num_hops
        self.subset_list =subset_list
        self.agg_feats = nn.ParameterList()
        for _ in range(num_hops):
            self.agg_feats.append(nn.Parameter(torch.FloatTensor(len(subset_list), in_feats)))
            
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for weight in self.agg_feats:
            nn.init.xavier_uniform_(weight, gain=gain)

    def forward(self, feats_dict):
        new_feats = []
        for k in range(self.num_hops):
            feats = torch.cat([feats_dict[rel_subset][k].unsqueeze(1) for rel_subset in self.subset_list], dim=1)
            new_feats.append((feats * self.agg_feats[k].unsqueeze(0)).sum(dim=1))

        return new_feats

# NARS model wrapper with homogeneous model (MLP, SIGN, SAGN...) as input
# This can also be used as the base model in SLEModel
class NARS(nn.Module):
    def __init__(self, in_feats, num_hops, homo_model, subset_list):
        super(NARS, self).__init__()
        self.aggregator = WeightedAggregator(subset_list, in_feats, num_hops)
        self.clf = homo_model
        self.reset_parameters()
    
    def reset_parameters(self):
        self.aggregator.reset_parameters()
        self.clf.reset_parameters()

    def forward(self, feats_dict):
        feats = self.aggregator(feats_dict)
        out = self.clf(feats)
        return out

################################################################
# Enhanced model with a label model in SLE
class SLEModel(nn.Module):
    def __init__(self, base_model, label_model, reproduce_previous=True):
        super().__init__()
        self._reproduce_previous = reproduce_previous
        self.base_model = base_model
        self.label_model = label_model
        self.reset_parameters()

    def reset_parameters(self):
        if self._reproduce_previous:
            self.previous_reset_parameters()
        else:
            if self.base_model is not None:
                self.base_model.reset_parameters()
            if self.label_model is not None:
                self.label_model.reset_parameters()

    def previous_reset_parameters(self):
        # To ensure the reproducibility of results from 
        # previous (before clean up) version, we reserve
        # the old order of initialization.
        gain = nn.init.calculate_gain("relu")
        if self.base_model is not None:
            if hasattr(self.base_model, "multihop_encoders"):
                for encoder in self.base_model.multihop_encoders:
                    encoder.reset_parameters()
            if hasattr(self.base_model, "res_fc"):
                nn.init.xavier_normal_(self.base_model.res_fc.weight, gain=gain)
            if hasattr(self.base_model, "hop_attn_l"):
                if self.base_model._weight_style == "attention":
                    if self.base_model._zero_inits:
                        nn.init.zeros_(self.base_model.hop_attn_l)
                        nn.init.zeros_(self.base_model.hop_attn_r)
                    else:
                        nn.init.xavier_normal_(self.base_model.hop_attn_l, gain=gain)
                        nn.init.xavier_normal_(self.base_model.hop_attn_r, gain=gain)
            if self.label_model is not None:
                self.label_model.reset_parameters()
            if hasattr(self.base_model, "pos_emb"):
                if self.base_model.pos_emb is not None:
                    nn.init.xavier_normal_(self.base_model.pos_emb, gain=gain)
            if hasattr(self.base_model, "post_encoder"):
                self.base_model.post_encoder.reset_parameters()
            if hasattr(self.base_model, "bn"):
                self.base_model.bn.reset_parameters()
                    
        else:
            if self.label_model is not None:
                self.label_model.reset_parameters()

    def forward(self, feats, label_emb):
        out = 0
        if self.base_model is not None:
            out = self.base_model(feats)

        if self.label_model is not None:
            label_out = self.label_model(label_emb).mean(1)
            if isinstance(out, tuple):
                out = (out[0] + label_out, out[1])
            else:
                out = out + label_out

        return out


################################################################
# DGL's implementation of Correct&Smooth (C&S)

class LabelPropagation(nn.Module):
    r"""
    Description
    -----------
    Introduced in `Learning from Labeled and Unlabeled Data with Label Propagation <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf>`_
    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},
    where unlabeled data is inferred by labeled data via propagation.
    Parameters
    ----------
        num_layers: int
            The number of propagations.
        alpha: float
            The :math:`\alpha` coefficient.
        adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
    """
    def __init__(self, num_layers, alpha, adj='DAD', display=False):
        super(LabelPropagation, self).__init__()

        self.num_layers = num_layers
        self.alpha = alpha
        self.adj = adj
        self.display = display
    
    @torch.no_grad()
    def forward(self, g, labels, mask=None, post_step=lambda y: y.clamp_(0., 1.)):
        with g.local_scope():
            if labels.dtype == torch.long:
                labels = F.one_hot(labels.view(-1)).to(torch.float32)
            
            y = labels
            if mask is not None:
                y = torch.zeros_like(labels)
                y[mask] = labels[mask]
            
            last = (1 - self.alpha) * y
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5 if self.adj == 'DAD' else -1).to(labels.device).unsqueeze(1)

            for _ in tqdm(range(self.num_layers)) if self.display else range(self.num_layers):
                # Assume the graphs to be undirected
                if self.adj in ['DAD', 'AD']:
                    y = norm * y
                
                g.ndata['h'] = y
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                y = self.alpha * g.ndata.pop('h')

                if self.adj in ['DAD', 'DA']:
                    y = y * norm
                
                y = post_step(last + y)
            
            return y


class CorrectAndSmooth(nn.Module):
    r"""
    Description
    -----------
    Introduced in `Combining Label Propagation and Simple Models Out-performs Graph Neural Networks <https://arxiv.org/abs/2010.13993>`_
    Parameters
    ----------
        num_correction_layers: int
            The number of correct propagations.
        correction_alpha: float
            The coefficient of correction.
        correction_adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
        num_smoothing_layers: int
            The number of smooth propagations.
        smoothing_alpha: float
            The coefficient of smoothing.
        smoothing_adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
        autoscale: bool, optional
            If set to True, will automatically determine the scaling factor :math:`\sigma`. Default is True.
        scale: float, optional
            The scaling factor :math:`\sigma`, in case :obj:`autoscale = False`. Default is 1.
    """
    def __init__(self,
                 num_correction_layers,
                 correction_alpha,
                 correction_adj,
                 num_smoothing_layers,
                 smoothing_alpha,
                 smoothing_adj,
                 autoscale=True,
                 scale=1.):
        super(CorrectAndSmooth, self).__init__()
        
        self.autoscale = autoscale
        self.scale = scale

        self.prop1 = LabelPropagation(num_correction_layers,
                                      correction_alpha,
                                      correction_adj)
        # We correct typo "correction_adj" in smoothing label propagation
        self.prop2 = LabelPropagation(num_smoothing_layers,
                                      smoothing_alpha,
                                      smoothing_adj)

    def correct(self, g, y_soft, y_true, train_nid, val_nid, test_nid, n_nodes):
        # We suppose y_soft=[(train_nodes, val_nodes, test_nodes), n_classes]
        # Expected output result=[(train_nodes, val_nodes, test_nodes), n_classes]
        # while labels=[all_nodes, n_classes]
        with g.local_scope():
            assert abs(float(y_soft.sum()) / y_soft.size(0) - 1.0) < 1e-2
            numel = train_nid.size(0)
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)
            
            error = torch.zeros(size=(n_nodes, y_soft.size(1))).to(y_soft.device)
            error[train_nid] = y_true - y_soft[:len(train_nid)]
            y_all = torch.zeros(size=(n_nodes, y_soft.size(1))).to(y_soft.device)
            y_all[torch.cat([train_nid, val_nid, test_nid], dim=0)] = y_soft

            if self.autoscale:
                smoothed_error = self.prop1(g, error, post_step=lambda x: x.clamp_(-1., 1.))
                sigma = error[train_nid].abs().sum() / numel
                scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
                scale[scale.isinf() | (scale > 1000)] = 1.0

                result = y_soft + scale * smoothed_error[torch.cat([train_nid, val_nid, test_nid], dim=0)]
                result[result.isnan()] = y_soft[result.isnan()]
                return result
            else:
                def fix_input(x):
                    x[train_nid] = error[train_nid]
                    return x
                
                smoothed_error = self.prop1(g, error, post_step=fix_input)

                result = y_all + self.scale * smoothed_error[torch.cat([train_nid, val_nid, test_nid], dim=0)]
                result[result.isnan()] = y_soft[result.isnan()]
                return result

    def smooth(self, g, y_soft, y_true, train_nid, val_nid, test_nid, n_nodes):
        with g.local_scope():
            numel = train_nid.size(0)
            assert y_true.size(0) == numel

            if len(y_true.shape) == 1 or y_true.shape[1] == 1:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)
            
            y_all = torch.zeros(size=(n_nodes, y_soft.size(1))).to(y_soft.device)
            y_all[torch.cat([train_nid, val_nid, test_nid], dim=0)] = y_soft
            y_all[train_nid] = y_true
            
            return self.prop2(g, y_all)[torch.cat([train_nid, val_nid, test_nid], dim=0)]

    def forward(self, g, y_soft, y_true, operations, train_nid, val_nid, test_nid, n_nodes):
        for operation in operations:
            if operation == "correction":
                print("Performing correction...")
                y_soft = self.correct(g, y_soft, y_true, train_nid, val_nid, test_nid, n_nodes)
            if operation == "smoothing":
                print("Performing smoothing...")
                y_soft = self.smooth(g, y_soft, y_true, train_nid, val_nid, test_nid, n_nodes)
        return y_soft
