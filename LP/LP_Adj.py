import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
# from w import *
from sklearn.metrics import f1_score
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul

from .diffusion_feature import *
from .outcome_correlation import *


class LabelPropagation_Adj(nn.Module):
    def __init__(self, args, data, train_mask):
        super().__init__()
        self.train_cnt = 0
        self.args = args
        self.num_layers = args.num_layers
        self.alpha = args.lpStep.alpha

        self.num_classes = args.num_classes
        self.num_nodes = data.num_nodes
        self.edge_index = data.edge_index
        self.train_mask = train_mask

        self.preStep = PreStep(args)
        # self.midStep = MidStep(args)
        self.midStep = None
        self.lpStep = None
        self.embs_step1 = None
        self.x_after_step2 = None
        self.data_cpu = copy.deepcopy(data).to("cpu")
        self.data = data

    def train_net(self, input_dict):
        # only complete ONE-TIME backprop/update for all nodes

        # input_dict = {'x': self.x, 'y': self.y, 'optimizer': self.optimizer, 'loss_op': self.loss_op, 'device':self.device}
        self.train_cnt += 1
        device, split_masks = input_dict["device"], input_dict["split_masks"]

        if self.embs_step1 is None:  # only preprocess ONCE; has to be on cpu
            self.embs_step1 = self.preStep(self.data_cpu).to(device)

        data = self.data_cpu.to(device)
        x, y = data.x, data.y
        # self.edge_index = self.edge_index.to(device)
        loss_op = input_dict["loss_op"]
        train_mask = split_masks["train"]

        if self.midStep is None:
            self.midStep = MidStep(self.args, self.embs_step1, self.data).to(device)
            self.optimizer = torch.optim.Adam(
                self.midStep.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )

        if self.lpStep is None:
            self.lpStep = LPStep(self.args, data, split_masks)

        self.x_after_step2, train_loss = self.midStep.train_forward(
            self.embs_step1, y, self.optimizer, loss_op, split_masks
        )  # only place that require opt

        # if self.train_cnt>20:
        #     print()
        #     acc = cal_acc_logits(self.x_after_step2[split_masks['test']], data.y[split_masks['test']])

        # ---- below 2 lines only used for printing test acc ----
        self.lpStep.split_masks = split_masks
        self.lpStep.y = y

        self.out = self.lpStep(self.x_after_step2, data)
        self.out, y, train_mask = to_device([self.out, y, train_mask], "cpu")

        train_acc = cal_acc_logits(self.out[train_mask], y[train_mask])
        # total_correct = int(self.out[train_mask].argmax(dim=-1).eq(y[train_mask]).sum())
        # train_acc = total_correct / int(train_mask.sum())
        return train_loss, train_acc

    def inference(self, input_dict):

        # return self.x_after_step2
        return self.out

    @torch.no_grad()
    def forward_backup(
        self,
        y: Tensor,
        edge_index: Adj,
        mask: Optional[Tensor] = None,
        edge_weight: OptTensor = None,
        post_step: Callable = lambda y: y.clamp_(0.0, 1.0),
    ) -> Tensor:
        """"""

        if y.dtype == torch.long:
            y = F.one_hot(y.view(-1)).to(torch.float)

        out = y
        if mask is not None:
            out = torch.zeros_like(y)
            out[mask] = y[mask]

        if isinstance(edge_index, SparseTensor) and not edge_index.has_value():
            edge_index = gcn_norm(edge_index, add_self_loops=False)
        elif isinstance(edge_index, Tensor) and edge_weight is None:
            edge_index, edge_weight = gcn_norm(
                edge_index, num_nodes=y.size(0), add_self_loops=False
            )

        res = (1 - self.alpha) * out
        for _ in range(self.num_layers):
            # propagate_type: (y: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight, size=None)
            out.mul_(self.alpha).add_(res)
            out = post_step(out)

        return out

    # def inference(self, input_dict):
    #     label = input_dict['y'].data
    #     Y = torch.zeros((self.num_nodes, self.num_classes))
    #     Y[self.train_mask] = F.one_hot(label[self.train_mask], self.num_classes).float()

    #     Y_soft = self.forward(Y, self.edge_index)
    #     return Y_soft

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return "{}(num_layers={}, alpha={})".format(
            self.__class__.__name__, self.num_layers, self.alpha
        )


class LPStep(nn.Module):
    """two papers:
    http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf
    https://github.com/CUAI/CorrectAndSmooth
    """

    def __init__(self, args, data, split_masks):
        super().__init__()
        self.train_cnt = 0
        self.args = args
        self.train_idx = torch.where(split_masks["train"] == True)[0].to(args.device)
        self.valid_idx = torch.where(split_masks["valid"] == True)[0].to(args.device)
        self.test_idx = torch.where(split_masks["test"] == True)[0].to(args.device)
        self.split_idx = {
            "train": self.train_idx,
            "valid": self.valid_idx,
            "test": self.test_idx,
        }
        self.no_prep = args.lpStep.no_prep
        self.best_test_acc = -9999.0

        adj, D_isqrt = process_adj(data)
        DAD, DA, AD = gen_normalized_adjs(adj, D_isqrt)

        self.lp_dict = {
            "train_only": True,
            "alpha1": args.lpStep.alpha1,
            "alpha2": args.lpStep.alpha2,
            "A1": eval(args.lpStep.A1),
            "A2": eval(args.lpStep.A2),
            "num_propagations1": args.lpStep.num_propagations1,
            "num_propagations2": args.lpStep.num_propagations2,
            "display": False,
            "device": args.device,
            # below: lp only
            "idxs": ["train"],
            "alpha": args.lpStep.alpha,
            "num_propagations": args.lpStep.num_propagations,
            "A": eval(args.lpStep.A),
            # below: gat
            "labels": ["train"],
        }
        self.fn = eval(self.args.lpStep.fn)
        return

    def forward(self, model_out, data):
        # need to pass 'data.y' through 'data'
        self.train_cnt += 1

        if self.args.lpStep.lp_force_on_cpu:
            self.split_idx, data, model_out = to_device(
                [self.split_idx, data, model_out], "cpu"
            )
        else:
            self.split_idx, data, model_out = to_device(
                [self.split_idx, data, model_out], self.args.device
            )

        if self.no_prep:
            out = label_propagation(data, self.split_idx, **self.lp_dict)
        else:
            # print('cnt=', self.train_cnt)
            # if self.train_cnt>20:
            #     print()

            # from .cs.run_experiments  import evaluate_params, eval_test, mlp_dict, mlp_fn, datafixed
            # files = ['models/arxiv_mlp/6.pt', 'models/arxiv_mlp/2.pt', 'models/arxiv_mlp/0.pt', 'models/arxiv_mlp/8.pt', 'models/arxiv_mlp/4.pt', 'models/arxiv_mlp/7.pt', 'models/arxiv_mlp/3.pt', 'models/arxiv_mlp/1.pt', 'models/arxiv_mlp/9.pt', 'models/arxiv_mlp/5.pt']
            # file = files[2]
            # f2 = '../CorrectAndSmooth/'+file
            # model_out1, run = model_load(f2)

            # for file in files:
            #     f2 = '../CorrectAndSmooth/'+file
            #     model_out1, run = model_load(f2)

            #     _, out = self.fn(data, model_out1, self.split_idx, **self.lp_dict)
            #     acc = cal_acc_logits(out[self.split_idx['test']], data.y[self.split_idx['test']])

            #     # acc = cal_acc_logits(out, data.y)
            #     _, out = self.fn(data, model_out, self.split_idx, **self.lp_dict)
            #     acc = cal_acc_logits(out[self.split_idx['test']], data.y[self.split_idx['test']])

            #     acc = cal_acc_logits(model_out1[self.split_idx['test']], data.y[self.split_idx['test']])
            #     acc = cal_acc_logits(model_out[self.split_idx['test']], data.y[self.split_idx['test']])

            #     print(acc)
            # raise

            _, out = self.fn(data, model_out.exp(), self.split_idx, **self.lp_dict)

            # from .cs.run_experiments  import evaluate_params, eval_test, mlp_dict, mlp_fn, datafixed
            # out = evaluate_params(datafixed, eval_test, None, self.split_idx, mlp_dict, fn = mlp_fn)

        test_mask = self.split_masks["test"]
        test_acc = cal_acc_logits(out[test_mask], self.y[test_mask])
        self.best_test_acc = max([test_acc, self.best_test_acc])
        print(
            f"\n-----------------\n in LPStep, best test acc  =  {self.best_test_acc*100:.3f}"
        )

        self.split_idx, data, model_out = to_device(
            [self.split_idx, data, model_out], self.args.device
        )
        return out


class PreStep(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.data, self.split_idx = load_ogbn_1()
        return

    def forward(self, data):

        # data = self.data
        embs = []
        if "diffusion" in self.args.preStep.pre_methods:
            embs.append(
                preprocess(
                    data,
                    "diffusion",
                    self.args.preStep.num_propagations,
                    post_fix=self.args.dataset,
                )
            )
        if "spectral" in self.args.preStep.pre_methods:
            embs.append(
                preprocess(
                    data,
                    "spectral",
                    self.args.preStep.num_propagations,
                    post_fix=self.args.dataset,
                )
            )
        if "community" in self.args.preStep.pre_methods:
            embs.append(
                preprocess(
                    data,
                    "community",
                    self.args.preStep.num_propagations,
                    post_fix=self.args.dataset,
                )
            )

        embeddings = torch.cat(embs, dim=-1)
        # x = torch.cat([data.x, embeddings], dim=-1)

        return embeddings


class MidStep(nn.Module):
    def __init__(self, args, embs, data):
        super().__init__()

        self.args = args
        self.train_cnt = 0
        self.best_valid = 0.0

        self.data = data
        # self.data, self.split_idx = load_ogbn_2()

        if args.midStep.model == "mlp":
            self.model = MLP(
                embs.size(-1) + args.num_feats,
                args.midStep.hidden_channels,
                args.num_classes,
                args.midStep.num_layers,
                0.5,
                args.dataset == "Products",
            ).to(args.device)
        elif args.midStep.model == "linear":
            self.model = MLPLinear(embs.size(-1) + args.num_feats, args.num_classes).to(
                args.device
            )
        elif args.midStep.model == "plain":
            self.model = MLPLinear(embs.size(-1) + args.num_feats, args.num_classes).to(
                args.device
            )
        return

    def forward(self, x):
        return self.model(x)

    def train_forward(self, embs, y, optimizer, loss_op, split_masks):
        self.train_cnt += 1

        # e1 = torch.load(f'../CorrectAndSmooth/embeddings/diffusionarxiv.pt')
        # e2 = torch.load(f'../CorrectAndSmooth/embeddings/spectralarxiv.pt')
        # self.data.x, e1, e2 = to_device([self.data.x, e1,e2], self.args.device)
        # x = torch.cat([self.data.x, e1,e2], dim=-1)

        x = torch.cat(to_device([self.data.x, embs], self.args.device), dim=-1)

        # if self.train_cnt%10==0:
        #     os.makedirs('step2_x_emb',exist_ok=1)
        #     torch.save( x.to('cpu'), f'step2_x_emb/{self.train_cnt}.pt')

        train_mask = split_masks["train"]
        valid_mask = split_masks["valid"]
        test_mask = split_masks["test"]

        optimizer.zero_grad()
        out = self.model(x)
        if isinstance(loss_op, torch.nn.NLLLoss):
            out = F.log_softmax(out, dim=-1)
        else:
            y = self.data.y.to(self.args.device).float()

        loss = loss_op(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        train_acc = cal_acc_logits(out[train_mask], y[train_mask])
        valid_acc = cal_acc_logits(out[valid_mask], y[valid_mask])
        test_acc = cal_acc_logits(out[test_mask], y[test_mask])

        if valid_acc > self.best_valid:
            self.best_valid = valid_acc
            # self.best_out = out.exp()
            self.best_out = out
            print("!!! best val")
        print(
            f"in mid step, cnt = {self.train_cnt}, train/val/test acc = {train_acc:.3f}, {valid_acc:.3f}, {test_acc:.3f}"
        )

        loss = float(loss.item())

        return self.best_out, loss


def cal_acc_logits(output, labels):
    # work with model-output, which can be either logits (0~1) or real-valued pre-exp output.
    assert len(output.shape) == 2 and output.shape[1] > 1
    if (
        len(labels.shape) == 2 and labels.shape[1] > 1
    ):  # the case of multi label classification
        if output.min() >= 0:  # has exp() operation
            output = output.log()

        pred = (output > 0).float()
        y = labels
        correct = int(pred.eq(y).sum()) / (y.size(0) * y.size(1))
    else:
        labels = labels.reshape(-1).to("cpu")
        indices = torch.max(output, dim=1)[1].to("cpu")
        correct = float(torch.sum(indices == labels) / len(labels))
    return correct


# def cal_acc_indices(output, labels):
#     assert (len(output.shape)==2 and output.shape[1]==1) or len(output.shape)==1
#     labels = labels.reshape(-1).to('cpu')
#     output = output.reshape(-1).to('cpu')
#     correct = float(torch.sum(output == labels)/len(labels))
#     return correct


def to_device(list1d, device):
    newl = []
    for x in list1d:
        if type(x) is dict:
            for k, v in x.items():
                x[k] = v.to(device)
        else:
            x = x.to(device)
        newl.append(x)
    return newl


# def load_ogbn_2(dataset='ogbn-arxiv'):
#     from torch_geometric.utils import to_undirected
#     import torch_geometric.transforms as T
#     dataset = PygNodePropPredDataset(name=dataset,transform=T.ToSparseTensor())
#     split_idx = dataset.get_idx_split()
#     data = dataset[0]
#     data.adj_t = data.adj_t.to_symmetric()
#     # data.edge_index = to_undirected(data.edge_index, data.num_nodes)
#     data.y = data.y.squeeze(1)
#     return data, split_idx

# def load_ogbn_3(dataset='ogbn-arxiv'):
#     from torch_geometric.utils import to_undirected
#     import torch_geometric.transforms as T
#     dataset = PygNodePropPredDataset(name=dataset)
#     split_idx = dataset.get_idx_split()
#     data = dataset[0]
#     # data.adj_t = data.adj_t.to_symmetric()
#     # data.edge_index = to_undirected(data.edge_index, data.num_nodes)
#     data.y = data.y.squeeze(1)
#     return data, split_idx
