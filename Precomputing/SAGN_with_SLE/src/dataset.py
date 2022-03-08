import os
from functools import namedtuple

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.data import PPIDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from sklearn.metrics import accuracy_score, f1_score
import scipy.sparse as sp
import json

from networkx.readwrite import json_graph


def get_ogb_evaluator(dataset):
    """
    Get evaluator from Open Graph Benchmark based on dataset
    """
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
        "y_true": labels.view(-1, 1),
        "y_pred": preds.view(-1, 1),
    })["acc"]

class ACCEvaluator(object):

    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):

        return accuracy_score(y_true.cpu(), y_pred.cpu())

class F1Evaluator(object):

    def __init__(self, average='micro'):
        self.average = average
        pass

    def __call__(self, y_pred, y_true):

        return f1_score(y_true.cpu(), y_pred.cpu(), average=self.average)

def get_evaluator(name):
    if name in ["cora"]:
        evaluator = ACCEvaluator()
    elif name in ["yelp", "ppi", "ppi_large", "reddit", "flickr"]:
        evaluator = F1Evaluator(average="micro")
    else:
        evaluator = get_ogb_evaluator(name)
    return evaluator

def load_dataset(device, args):
    """
    Load dataset and move graph and features to device
    """
    if args.dataset in ["reddit", "cora", "ppi", "ppi_large", "yelp", "flickr"]:
        # raise RuntimeError("Dataset {} is not supported".format(name))
        if args.dataset == "reddit":
            from dgl.data import RedditDataset
            data = RedditDataset(self_loop=True)
            g = data[0]
            g = dgl.add_self_loop(g)
            n_classes = data.num_classes
        elif args.dataset == "cora":
            from dgl.data import CitationGraphDataset
            data = CitationGraphDataset('cora', raw_dir=os.path.join(args.data_dir, 'cora'))
            g = data[0]
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
            n_classes = data.num_classes
        elif args.dataset == "ppi":
            data = load_ppi_data(args.data_dir)
            g = data.g
            n_classes = data.num_classes
        elif args.dataset == "ppi_large":
            data = load_ppi_large_data()
            g = data.g
            n_classes = data.num_classes
        elif args.dataset == "yelp":
            from torch_geometric.datasets import Yelp
            pyg_data = Yelp(os.path.join(args.data_dir, 'yelp'))[0]
            feat = pyg_data.x
            labels = pyg_data.y
            u, v = pyg_data.edge_index
            g = dgl.graph((u, v))
            g.ndata['feat'] = feat
            g.ndata['label'] = labels
            g.ndata['train_mask'] = pyg_data.train_mask
            g.ndata['val_mask'] = pyg_data.val_mask
            g.ndata['test_mask'] = pyg_data.test_mask
            n_classes = labels.size(1)
        elif args.dataset == "flickr":
            from torch_geometric.datasets import Flickr
            pyg_data = Flickr(os.path.join(args.data_dir, "flickr"))[0]
            feat = pyg_data.x
            labels = pyg_data.y
            # labels = torch.argmax(labels, dim=1)
            u, v = pyg_data.edge_index
            g = dgl.graph((u, v))
            g.ndata['feat'] = feat
            g.ndata['label'] = labels
            g.ndata['train_mask'] = pyg_data.train_mask
            g.ndata['val_mask'] = pyg_data.val_mask
            g.ndata['test_mask'] = pyg_data.test_mask
            n_classes = labels.max().item() + 1
        
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        train_nid = train_mask.nonzero().squeeze().long()
        val_nid = val_mask.nonzero().squeeze().long()
        test_nid = test_mask.nonzero().squeeze().long()
        g = g.to(device)
        labels = g.ndata['label']

    else:
        dataset = DglNodePropPredDataset(name=args.dataset, root=args.data_dir)
        splitted_idx = dataset.get_idx_split()
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]
        n_classes = dataset.num_classes
        g = g.to(device)

        if args.dataset == "ogbn-arxiv":
            g = dgl.add_reverse_edges(g, copy_ndata=True)
            g = dgl.add_self_loop(g)
            g.ndata['feat'] = g.ndata['feat'].float()

        elif args.dataset == "ogbn-papers100M":
            g = dgl.add_reverse_edges(g, copy_ndata=True)
            g.ndata['feat'] = g.ndata['feat'].float()
            labels = labels.long()

        elif args.dataset == "ogbn-mag":
            # MAG is a heterogeneous graph. The task is to make prediction for
            # paper nodes
            path = os.path.join(args.emb_path, f"{args.pretrain_model}_mag")
            labels = labels["paper"]
            train_nid = train_nid["paper"]
            val_nid = val_nid["paper"]
            test_nid = test_nid["paper"]
            features = g.nodes['paper'].data['feat']
            author_emb = torch.load(os.path.join(path, "author.pt"), map_location=torch.device("cpu")).float()
            topic_emb = torch.load(os.path.join(path, "field_of_study.pt"), map_location=torch.device("cpu")).float()
            institution_emb = torch.load(os.path.join(path, "institution.pt"), map_location=torch.device("cpu")).float()

            g.nodes["author"].data["feat"] = author_emb.to(device)
            g.nodes["institution"].data["feat"] = institution_emb.to(device)
            g.nodes["field_of_study"].data["feat"] = topic_emb.to(device)
            g.nodes["paper"].data["feat"] = features.to(device)
            paper_dim = g.nodes["paper"].data["feat"].shape[1]
            author_dim = g.nodes["author"].data["feat"].shape[1]
            if paper_dim != author_dim:
                paper_feat = g.nodes["paper"].data.pop("feat")
                rand_weight = torch.Tensor(paper_dim, author_dim).uniform_(-0.5, 0.5)
                g.nodes["paper"].data["feat"] = torch.matmul(paper_feat, rand_weight.to(device))
                print(f"Randomly project paper feature from dimension {paper_dim} to {author_dim}")

            labels = labels.to(device).squeeze()
            n_classes = int(labels.max() - labels.min()) + 1
        
        else:
            g.ndata['feat'] = g.ndata['feat'].float()

        labels = labels.squeeze()

    evaluator = get_evaluator(args.dataset)

    print(f"# Nodes: {g.number_of_nodes()}\n"
          f"# Edges: {g.number_of_edges()}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {n_classes}")

    return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator

def load_ppi_data(root):
    DataType = namedtuple('Dataset', ['num_classes', 'g'])
    adj_full = sp.load_npz(os.path.join(root, 'ppi', 'adj_full.npz'))
    G = dgl.from_scipy(adj_full)
    nodes_num = G.num_nodes()
    role = json.load(open(os.path.join(root, 'ppi','role.json'),'r'))
    tr = list(role['tr'])
    te = list(role['te'])
    va = list(role['va'])
    mask = np.zeros((nodes_num,), dtype=bool)
    train_mask = mask.copy()
    train_mask[tr] = True
    val_mask = mask.copy()
    val_mask[va] = True
    test_mask = mask.copy()
    test_mask[te] = True
    
    G.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
    G.ndata['val_mask'] = torch.tensor(val_mask, dtype=torch.bool)
    G.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)

    feats=np.load(os.path.join(root, 'ppi', 'feats.npy'))
    G.ndata['feat'] = torch.tensor(feats, dtype=torch.float)

    class_map = json.load(open(os.path.join(root, 'ppi', 'class_map.json'), 'r'))
    labels = np.array([class_map[str(i)] for i in range(nodes_num)])
    G.ndata['label'] = torch.tensor(labels, dtype=torch.float)
    data = DataType(g=G, num_classes=labels.shape[1])
    return data

def load_ppi_large_data():
    '''Wraps the dgl's load_data utility to handle ppi special case'''
    DataType = namedtuple('Dataset', ['num_classes', 'g'])
    train_dataset = PPIDataset('train')
    train_graph = dgl.batch([train_dataset[i] for i in range(len(train_dataset))], edge_attrs=None, node_attrs=None)
    val_dataset = PPIDataset('valid')
    val_graph = dgl.batch([val_dataset[i] for i in range(len(val_dataset))], edge_attrs=None, node_attrs=None)
    test_dataset = PPIDataset('test')
    test_graph = dgl.batch([test_dataset[i] for i in range(len(test_dataset))], edge_attrs=None, node_attrs=None)
    G = dgl.batch(
        [train_graph, val_graph, test_graph], edge_attrs=None, node_attrs=None)

    train_nodes_num = train_graph.number_of_nodes()
    test_nodes_num = test_graph.number_of_nodes()
    val_nodes_num = val_graph.number_of_nodes()
    nodes_num = G.number_of_nodes()
    assert(nodes_num == (train_nodes_num + test_nodes_num + val_nodes_num))
    # construct mask
    mask = np.zeros((nodes_num,), dtype=bool)
    train_mask = mask.copy()
    train_mask[:train_nodes_num] = True
    val_mask = mask.copy()
    val_mask[train_nodes_num:-test_nodes_num] = True
    test_mask = mask.copy()
    test_mask[-test_nodes_num:] = True

    G.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
    G.ndata['val_mask'] = torch.tensor(val_mask, dtype=torch.bool)
    G.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)

    data = DataType(g=G, num_classes=train_dataset.num_labels)
    return data
