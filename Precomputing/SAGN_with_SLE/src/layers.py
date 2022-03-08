import torch
from torch import nn
import math
from typing import Optional
from torch import Tensor

eps=1e-5

################################################################
# DGL's implementation of FeedForwardNet (MLP) for SIGN
class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x


################################################################
# More general MLP layer
class MLP(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout, input_drop=0., residual=False, normalization="batch"):
        super(MLP, self).__init__()
        self._residual = residual
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.n_layers = n_layers
        
        self.input_drop = nn.Dropout(input_drop)

        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            if normalization == "batch":
                self.norms.append(nn.BatchNorm1d(hidden))
            if normalization == "layer":
                self.norms.append(nn.LayerNorm(hidden))
            if normalization == "none":
                self.norms.append(nn.Identity())
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
                if normalization == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden))
                if normalization == "layer":
                    self.norms.append(nn.LayerNorm(hidden))
                if normalization == "none":
                    self.norms.append(nn.Identity())
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

        for norm in self.norms:
            norm.reset_parameters()
        # print(self.layers[0].weight)

    def forward(self, x):
        x = self.input_drop(x)
        if self._residual:
            prev_x = x
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.relu(self.norms[layer_id](x)))
            if self._residual:
                if x.shape[1] == prev_x.shape[1]:
                    x += prev_x
                prev_x = x

        return x

# Multi-head (ensemble) MLP, note that different heads are processed
# sequentially
class MultiHeadMLP(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_heads, n_layers, dropout, input_drop=0., concat=False, residual=False, normalization="batch"):
        super().__init__()
        self._concat = concat
        self.mlp_list = nn.ModuleList([MLP(in_feats, hidden, out_feats, n_layers, dropout, input_drop=input_drop, residual=residual, normalization=normalization) for _ in range(n_heads)])
        # self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.mlp_list:
            mlp.reset_parameters()
            
    def forward(self, x):
        # x size:
        # [N, d_in] or [N, H, d_in]
        if len(x.shape) == 3:
            out = [mlp(x[:, i, :]) for i, mlp in enumerate(self.mlp_list)]
        if len(x.shape) == 2:
            out = [mlp(x) for mlp in self.mlp_list]
        out = torch.stack(out, dim=1)
        if self._concat:
            out = out.flatten(1, -1)
        return out

class ParallelMLP(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_heads, n_layers, dropout, input_drop=0., residual=False, normalization="batch"):
        super(ParallelMLP, self).__init__()
        self._residual = residual
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self._n_heads = n_heads
        self._n_layers = n_layers
        
        self.input_drop = nn.Dropout(input_drop)

        if self._n_layers == 1:
            # self.layers.append(MultiHeadLinear(in_feats, out_feats, n_heads))
            self.layers.append(nn.Conv1d(in_feats * n_heads, out_feats * n_heads, kernel_size=1, groups=n_heads))
        else:
            # self.layers.append(MultiHeadLinear(in_feats, hidden, n_heads))
            self.layers.append(nn.Conv1d(in_feats * n_heads, hidden * n_heads, kernel_size=1, groups=n_heads))
            if normalization == "batch":
                # self.norms.append(MultiHeadBatchNorm(n_heads, hidden * n_heads))
                self.norms.append(nn.BatchNorm1d(hidden * n_heads))
            if normalization == "layer":
                self.norms.append(nn.GroupNorm(n_heads, hidden * n_heads))
            if normalization == "none":
                self.norms.append(nn.Identity())
            for i in range(self._n_layers - 2):
                # self.layers.append(MultiHeadLinear(hidden, hidden, n_heads))
                self.layers.append(nn.Conv1d(hidden * n_heads, hidden * n_heads, kernel_size=1, groups=n_heads))
                if normalization == "batch":
                    # self.norms.append(MultiHeadBatchNorm(n_heads, hidden * n_heads))
                    self.norms.append(nn.BatchNorm1d(hidden * n_heads))
                if normalization == "layer":
                    self.norms.append(nn.GroupNorm(n_heads, hidden * n_heads))
                if normalization == "none":
                    self.norms.append(nn.Identity())
            # self.layers.append(MultiHeadLinear(hidden, out_feats, n_heads))
            self.layers.append(nn.Conv1d(hidden * n_heads, out_feats * n_heads, kernel_size=1, groups=n_heads))
        if self._n_layers > 1:
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        # for head in range(self._n_heads):
            
        #     for layer in self.layers:

        #         nn.init.kaiming_uniform_(layer.weight[head], a=math.sqrt(5))
        #         if layer.bias is not None:
        #             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight[head])
        #             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #             nn.init.uniform_(layer.bias[head], -bound, bound)
        self.reset_parameters()

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")
    
        for head in range(self._n_heads):
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight[head], gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias[head])
        for norm in self.norms:
            norm.reset_parameters()
            # for norm in self.norms:
            #     norm.moving_mean[head].zero_()
            #     norm.moving_var[head].fill_(1)
            #     if norm._affine:
            #         nn.init.ones_(norm.scale[head])
            #         nn.init.zeros_(norm.offset[head])
        # print(self.layers[0].weight[0])

    def forward(self, x):
        x = self.input_drop(x)
        if len(x.shape) == 2:
            x = x.view(-1, 1, x.shape[1])
            x = x.repeat(1, self._n_heads, 1)
            # x = x.repeat(1, self._n_heads).unsqueeze(-1)
        if len(x.shape) == 3:
            x = x.flatten(1, -1).unsqueeze(-1)
        if self._residual:
            prev_x = x
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            # x = x.flatten(1, -1)
            if layer_id < self._n_layers - 1:
                shape = x.shape
                
                x = self.dropout(self.relu(self.norms[layer_id](x)))
                # x = x.reshape(shape=shape)

            if self._residual:
                if x.shape[2] == prev_x.shape[2]:
                    x += prev_x
                prev_x = x
        x = x.view(-1, self._n_heads, x.shape[1] // self._n_heads)

        return x

################################################################
# Modified multi-head Linear layer
class MultiHeadLinear(nn.Module):
    def __init__(self, in_feats, out_feats, n_heads, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(size=(n_heads, in_feats, out_feats)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(n_heads, 1, out_feats)))
        else:
            self.bias = None

    def reset_parameters(self) -> None:
        for weight, bias in zip(self.weight, self.bias):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            if bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(bias, -bound, bound)

    # def reset_parameters(self):
    #     gain = nn.init.calculate_gain("relu")
    #     for weight in self.weight:
    #         nn.init.xavier_uniform_(weight, gain=gain)
    #     if self.bias is not None:
    #         nn.init.zeros_(self.bias)

    def forward(self, x):
        # input size: [N, d_in] or [H, N, d_in]
        # output size: [H, N, d_out]
        if len(x.shape) == 3:
            x = x.transpose(0, 1)

        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x += self.bias
        return x.transpose(0, 1)

# Modified multi-head BatchNorm1d layer
class MultiHeadBatchNorm(nn.Module):
    def __init__(self, n_heads, in_feats, momentum=0.1, affine=True, device=None,
        dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        assert in_feats % n_heads == 0
        self._in_feats = in_feats
        self._n_heads = n_heads
        self._momentum = momentum
        self._affine = affine
        if affine:
            self.weight = nn.Parameter(torch.empty(size=(n_heads, in_feats // n_heads)))
            self.bias = nn.Parameter(torch.empty(size=(n_heads, in_feats // n_heads)))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        
        self.register_buffer("running_mean", torch.zeros(size=(n_heads, in_feats // n_heads)))
        self.register_buffer("running_var", torch.ones(size=(n_heads, in_feats // n_heads)))
        self.running_mean: Optional[Tensor]
        self.running_var: Optional[Tensor]
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()  # type: ignore[union-attr]
        self.running_var.fill_(1)  # type: ignore[union-attr]
        if self._affine:
            nn.init.zeros_(self.bias)
            for weight in self.weight:
                nn.init.ones_(weight)

    def forward(self, x):
        assert x.shape[1] == self._in_feats
        x = x.view(-1, self._n_heads, self._in_feats // self._n_heads)
        
        self.running_mean = self.running_mean.to(x.device)
        self.running_var = self.running_var.to(x.device)
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        if bn_training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, unbiased=False, keepdim=True)
            out = (x-mean) * torch.rsqrt(var + eps)
            self.running_mean = (1 - self._momentum) * self.running_mean + self._momentum * mean.detach()
            self.running_var = (1 - self._momentum) * self.running_var + self._momentum * var.detach()
        else:
            out = (x - self.running_mean)  * torch.rsqrt(self.running_var + eps)
        if self._affine:
            out = out * self.weight + self.bias
        return out

# Another multi-head MLP defined from scratch
class GroupMLP(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_heads, n_layers, dropout, input_drop=0., residual=False, normalization="batch"):
        super(GroupMLP, self).__init__()
        self._residual = residual
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self._n_heads = n_heads
        self._n_layers = n_layers
        
        self.input_drop = nn.Dropout(input_drop)

        if self._n_layers == 1:
            self.layers.append(MultiHeadLinear(in_feats, out_feats, n_heads))
        else:
            self.layers.append(MultiHeadLinear(in_feats, hidden, n_heads))
            if normalization == "batch":
                self.norms.append(MultiHeadBatchNorm(n_heads, hidden * n_heads))
                # self.norms.append(nn.BatchNorm1d(hidden * n_heads))
            if normalization == "layer":
                self.norms.append(nn.GroupNorm(n_heads, hidden * n_heads))
            if normalization == "none":
                self.norms.append(nn.Identity())
            for i in range(self._n_layers - 2):
                self.layers.append(MultiHeadLinear(hidden, hidden, n_heads))
                if normalization == "batch":
                    self.norms.append(MultiHeadBatchNorm(n_heads, hidden * n_heads))
                    # self.norms.append(nn.BatchNorm1d(hidden * n_heads))
                if normalization == "layer":
                    self.norms.append(nn.GroupNorm(n_heads, hidden * n_heads))
                if normalization == "none":
                    self.norms.append(nn.Identity())
            self.layers.append(MultiHeadLinear(hidden, out_feats, n_heads))
        if self._n_layers > 1:
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        for head in range(self._n_heads):
            
            for layer in self.layers:

                nn.init.kaiming_uniform_(layer.weight[head], a=math.sqrt(5))
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight[head])
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(layer.bias[head], -bound, bound)
        self.reset_parameters()

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")
    
        for head in range(self._n_heads):
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight[head], gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias[head])
        for norm in self.norms:
            norm.reset_parameters()
            # for norm in self.norms:
            #     norm.moving_mean[head].zero_()
            #     norm.moving_var[head].fill_(1)
            #     if norm._affine:
            #         nn.init.ones_(norm.scale[head])
            #         nn.init.zeros_(norm.offset[head])
        # print(self.layers[0].weight[0])

    def forward(self, x):
        x = self.input_drop(x)
        if len(x.shape) == 2:
            x = x.view(-1, 1, x.shape[1])
        if self._residual:
            prev_x = x
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            
            if layer_id < self._n_layers - 1:
                shape = x.shape
                x = x.flatten(1, -1)
                x = self.dropout(self.relu(self.norms[layer_id](x)))
                x = x.reshape(shape=shape)

            if self._residual:
                if x.shape[2] == prev_x.shape[2]:
                    x += prev_x
                prev_x = x

        return x




