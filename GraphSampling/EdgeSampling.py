import torch
import time
import numpy as np
from torch_geometric.loader import (
    NeighborSampler,
    GraphSAINTSampler,
    GraphSAINTEdgeSampler,
    GraphSAINTRandomWalkSampler,
)
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import pandas as pd
import torch_geometric.transforms as T
# import GraphSampling.cpp_extension.sample as sample


class EdgeSampler(GraphSAINTSampler):
    r"""The GraphSAINT edge sampler class (see
    :class:`~torch_geometric.loader.GraphSAINTSampler`).
    """

    def __init__(
        self,
        data,
        batch_size: int,
        V_path=None,
        num_steps=1,
        sample_coverage=0,
        save_dir=None,
        log=True,
        **kwargs,
    ):
        super().__init__(
            data,
            batch_size,
            num_steps=num_steps,
            sample_coverage=sample_coverage,
            save_dir=save_dir,
            log=log,
            **kwargs,
        )
        self.V = torch.tensor(pd.read_csv(V_path, header=None).values).float()
        row = self.adj.storage.row()
        col = self.adj.storage.col()
        self.edge_imp = torch.norm(self.V[row, :] - self.V[col, :], dim=1) ** 2
        self.prob = self.edge_imp / torch.sum(self.edge_imp)

        # self.prob = torch.Tensor([1. / data.num_edges] * data.num_edges)
        self.p_cumsum = torch.cumsum(self.prob, 0)

        # self.edge_imp = (torch.norm(self.V[row, :] - self.V[col, :], dim=1) ** 2).numpy()
        # self.prob = self.edge_imp / np.sum(self.edge_imp)

    def __sample_nodes__(self, batch_size):
        row, col, _ = self.adj.coo()

        # # # # t = time.time()
        # edge_sample = torch.multinomial(self.prob, batch_size, replacement=True)
        # # import pdb; pdb.set_trace()
        # # # # print(f'sample time: {time.time() - t}')
        # source_node_sample = col[edge_sample]
        # target_node_sample = row[edge_sample]
        # return torch.cat([source_node_sample, target_node_sample], -1)

        edge_sample = sample.edge_sample2(self.p_cumsum, batch_size, None)
        # edge_sample = np.random.choice(range(len(self.prob)), batch_size, p=self.prob, replace=True)
        # print(edge_sample)
        source_node_sample = col[edge_sample]
        target_node_sample = row[edge_sample]
        return torch.cat([source_node_sample, target_node_sample], -1)


class EdgeSampling(torch.nn.Module):
    def __init__(self, args, data, train_idx, processed_dir):
        super(EdgeSampling, self).__init__()

        self.num_layers = args.num_layers
        self.dim_hidden = args.dim_hidden
        self.num_classes = args.num_classes
        self.num_feats = args.num_feats
        self.batch_size = args.batch_size
        self.train_size = train_idx.size(0)
        self.dropout = args.dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(self.num_feats, self.dim_hidden))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(self.dim_hidden, self.dim_hidden))
        self.convs.append(SAGEConv(self.dim_hidden, self.num_classes))
        edgesampler_num_steps = int(data.num_nodes / 2 / self.batch_size)
        print(f"edgesampler_num_steps: {edgesampler_num_steps}")
        self.train_loader = EdgeSampler(
            data,
            batch_size=self.batch_size,
            V_path=f"{processed_dir}/V_{args.dataset}_adj.csv",
            # num_steps= 30,
            num_workers=12,
            num_steps=edgesampler_num_steps,
        )
        # self.train_loader = GraphSAINTRandomWalkSampler(data, batch_size=self.batch_size,
        #                                 walk_length=4,
        #                                 num_steps=29)
        # num_steps=int(data.num_nodes / self.batch_size)+1)
        self.test_loader = NeighborSampler(
            data.edge_index,
            node_idx=None,
            sizes=[-1],
            batch_size=8192,
            shuffle=False,
            num_workers=12,
        )

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def train_net(self, input_dict):

        device = input_dict["device"]
        x = input_dict["x"].to(device)
        y = input_dict["y"].to(device)
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]

        # pbar = tqdm(total=self.train_size)
        # pbar.set_description('Training')

        total_loss = total_correct = 0
        for batch in self.train_loader:
            # torch.cuda.synchronize()
            # t = time.time()
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            optimizer.zero_grad()
            # batch = batch.to(device)
            batch = T.ToSparseTensor()(batch.to(device))
            out = self(batch.x, batch.adj_t)
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
            # torch.cuda.synchronize()
            # print(f'used {time.time() - t} sec')
            # pbar.update(self.batch_size)

        # pbar.close()
        train_size = (
            self.train_size
            if isinstance(loss_op, torch.nn.NLLLoss)
            else self.train_size * self.num_classes
        )
        return total_loss / len(self.train_loader), total_correct / train_size

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    #
    # def forward(self, x, adjs):
    #     # `train_loader` computes the k-hop neighborhood of a batch of nodes,
    #     # and returns, for each layer, a bipartite graph object, holding the
    #     # bipartite edges `edge_index`, the index `e_id` of the original edges,
    #     # and the size/shape `size` of the bipartite graph.
    #     # Target nodes are also included in the source nodes so that one can
    #     # easily apply skip-connections or add self-loops.
    #     for i, (edge_index, _, size) in enumerate(adjs):
    #         x_target = x[:size[1]]  # Target nodes are always placed first.
    #         x = self.convs[i]((x, x_target), edge_index)
    #         if i != self.num_layers - 1:
    #             x = F.relu(x)
    #             x = F.dropout(x, p=self.dropout, training=self.training)
    #     return x

    # def train_net(self, input_dict):

    #     device = input_dict['device']
    #     x = input_dict['x'].to(device)
    #     y = input_dict['y'].to(device)
    #     optimizer = input_dict['optimizer']
    #     loss_op = input_dict['loss_op']

    #     pbar = tqdm(total=self.train_size)
    #     pbar.set_description('Training')

    #     total_loss = total_correct = 0
    #     for batch_size, n_id, adjs in self.train_loader:
    #         # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
    #         adjs = [adj.to(device) for adj in adjs]
    #         optimizer.zero_grad()
    #         out = self.forward(x[n_id], adjs)
    #         if isinstance(loss_op, torch.nn.NLLLoss):
    #             out = F.log_softmax(out, dim=-1)
    #         loss = loss_op(out, y[n_id[:batch_size]])
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += float(loss.item())
    #         if isinstance(loss_op, torch.nn.NLLLoss):
    #             total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
    #         else:
    #             total_correct += int(out.eq(y[n_id[:batch_size]]).sum())
    #         pbar.update(batch_size)

    #     pbar.close()
    #     train_size = self.train_size if isinstance(loss_op, torch.nn.NLLLoss) \
    #         else self.train_size * self.num_classes
    #     return total_loss / len(self.train_loader), total_correct / train_size

    @torch.no_grad()
    def inference(self, input_dict):
        device = input_dict["device"]
        x_all = input_dict["x"]
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in self.test_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[: size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all
