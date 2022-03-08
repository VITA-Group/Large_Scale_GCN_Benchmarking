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
import tqdm
from sklearn.decomposition import PCA

from dataset import load_dataset
from utils import (calculate_homophily, clear_memory,
                   entropy, inner_distance, outer_distance)


def neighbor_average_features_by_chunks(g, feat, args, style="all", stats=False, memory_efficient=False, target_nid=None):
    """
    Compute multi-hop neighbor-averaged node features by chunks
    """
    if args.chunks == 1:
        return neighbor_average_features(g, feat, args, style=style, stats=stats, memory_efficient=memory_efficient, target_nid=target_nid)
    aggr_device = torch.device("cpu" if args.aggr_gpu < 0 else "cuda:{}".format(args.aggr_gpu))
    feat_size = feat.shape[1]
    chunk_size = int(math.ceil(feat_size / args.chunks))
    
    print("Saving temporary initial feature chunks……")
    tmp_dir = os.path.join(args.data_dir, "_".join(args.dataset.split("-")), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    for part, i in enumerate(tqdm.tqdm(range(0, feat_size, chunk_size))):
        chunk_path = os.path.join(tmp_dir, f"feat_{part}.npy")
        if os.path.exists(chunk_path):
            continue
        chunk = feat[:, i: min(i+chunk_size, feat_size)]
        np.save(chunk_path, chunk.cpu().numpy())

    del feat
    clear_memory(aggr_device)
    print("Perform feature propagation by chunks……")
    for part, i in enumerate(tqdm.tqdm(range(0, feat_size, chunk_size))):
        chunk = torch.from_numpy(np.load(os.path.join(tmp_dir, f"feat_{i}.npy"))).to(aggr_device)
        out_chunk = neighbor_average_features(g, chunk, args, style=style, stats=stats, memory_efficient=memory_efficient)
        if style=="all":
            chunk = [c.cpu().numpy() for c in chunk]
        else:
            chunk = chunk.cpu().numpy()
        np.save(os.path.join(tmp_dir, f"smoothed_feat_{part}.npy"), out_chunk)
    del chunk
    clear_memory(aggr_device)
    print("Loading aggregated chunks……")
    if style=="all":
        out_feat = [torch.empty_like(feat, device=aggr_device) for k in range(args.K+1)]  
    else:
        out_feat = torch.empty_like(feat, device=aggr_device)
    for part, i in enumerate(tqdm.tqdm(range(0, feat_size, chunk_size))):
        out_chunk = torch.from_numpy(np.load(os.path.join(tmp_dir, f"smoothed_feat_{part}.npy"))).to(aggr_device)
        if style == "all":
            for k in range(args.K+1):
                out_feat[k][:, i: min(i+chunk_size, feat_size)] = out_chunk[k]
        else:
            out_feat[:, i: min(i+chunk_size, feat_size)] = out_chunk
    print("Removing temporary files……")
    for part, i in enumerate(tqdm.tqdm(range(0, feat_size, chunk_size))):
        # os.remove(os.path.join(tmp_dir, f"feat_{i}.pt"))
        os.remove(os.path.join(tmp_dir, f"smoothed_feat_{part}.pt"))
    del out_chunk
    clear_memory(aggr_device)
    return out_feat


def neighbor_average_features(g, feat, args, style="all", stats=True, memory_efficient=False, target_nid=None):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats", style)
    
    aggr_device = torch.device("cpu" if args.aggr_gpu < 0 else "cuda:{}".format(args.aggr_gpu))
    g = g.to(aggr_device)
    feat = feat.to(aggr_device)
    tmp_dir = os.path.join(args.data_dir, "_".join(args.dataset.split("-")), "tmp")
    idx = target_nid if target_nid is not None else torch.arange(len(feat)).to(aggr_device)
    os.makedirs(tmp_dir, exist_ok=True)
    if style == "all":
        if memory_efficient:
            torch.save(feat[idx].clone(), os.path.join(tmp_dir, '0.pt'))
            res = []
        else:
            res = [feat[idx].clone()]
        
            
        # print(g.ndata["feat"].shape)
        # print(norm.shape)
        if args.use_norm:
            degs = g.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
        for hop in range(1, args.K + 1):
            g.ndata['f'] = feat
            if args.use_norm:
                g.ndata['f'] = g.ndata['f'] * norm
                g.update_all(fn.copy_src(src=f'f', out='msg'),
                            fn.sum(msg='msg', out=f'f'))
                g.ndata['f'] = g.ndata['f'] * norm
            else:
                g.update_all(fn.copy_src(src='f', out='msg'),
                            fn.mean(msg='msg', out='f'))
            feat = g.ndata.pop("f")
            if memory_efficient:
                torch.save(feat[idx].clone(), os.path.join(tmp_dir, f'{hop}.pt'))
            else:
                res.append(feat[idx].clone())
        
        del feat
        clear_memory(aggr_device)
        if memory_efficient:
            for hop in range(args.K+1):
                res.append(torch.load(os.path.join(tmp_dir, f'{hop}.pt')))
                os.remove(os.path.join(tmp_dir, f'{hop}.pt'))

            # if hop > 1:
            #     g.ndata['label_emb'] = 0.5 * g.ndata['pre_label_emb'] + \
            #                            0.5 * g.ndata['label_emb']

        clear_memory(aggr_device)

        if args.dataset == "ogbn-mag":
            # For MAG dataset, only return features for target node types (i.e.
            # paper nodes)
                target_mask = g.ndata['target_mask']
                target_ids = g.ndata[dgl.NID][target_mask]
                num_target = target_mask.sum().item()
                new_res = []
                for x in res:
                    feat = torch.zeros((num_target,) + x.shape[1:],
                                    dtype=x.dtype, device=x.device)
                    feat[target_ids] = x[target_mask]
                    new_res.append(feat)
                res = new_res

    # del g.ndata['pre_label_emb']
    elif style in ["last", "ppnp"]:
        if stats:
            feat_0 = feat.clone()
            train_mask = g.ndata["train_mask"]
            print(f"hop 0: outer distance {outer_distance(feat_0, feat_0, train_mask):.4f}, inner distance {inner_distance(feat_0, train_mask):.4f}")
        if style == "ppnp": init_feat = feat
        if args.use_norm:
            degs = g.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
        for hop in range(1, args.label_K+1):         
            # g.ndata["f_next"] = g.ndata["f"]
            if args.use_norm:
                feat = feat * norm
                g.ndata['f'] = feat
                g.update_all(fn.copy_src(src='f', out='msg'),
                            fn.sum(msg='msg', out='f'))
                feat = g.ndata.pop('f')
                # degs = g.in_degrees().float().clamp(min=1)
                # norm = torch.pow(degs, -0.5)
                # shp = norm.shape + (1,) * (g.ndata['f'].dim() - 1)
                # norm = torch.reshape(norm, shp)
                feat = feat * norm
            else:
                g.ndata['f'] = feat
                g.update_all(fn.copy_src(src='f', out='msg'),
                            fn.mean(msg='msg', out='f'))
                feat = g.ndata.pop('f')
            if style == "ppnp":
                feat = 0.5 * feat + 0.5 * init_feat
            if stats:
                print(f"hop {hop}: outer distance {outer_distance(feat_0, feat, train_mask):.4f}, inner distance {inner_distance(feat, train_mask):.4f}")
            
        res = feat[idx].clone()
        del feat
        clear_memory(aggr_device)

        if args.dataset == "ogbn-mag":
            # For MAG dataset, only return features for target node types (i.e.
            # paper nodes)
            target_mask = g.ndata['target_mask']
            target_ids = g.ndata[dgl.NID][target_mask]
            num_target = target_mask.sum().item()
            new_res = torch.zeros((num_target,) + feat.shape[1:],
                                    dtype=feat.dtype, device=feat.device)
            new_res[target_ids] = res[target_mask]
            res = new_res

    
    return res

def gen_rel_subset_feature(g, rel_subset, args, device):
    """
    Build relation subgraph given relation subset and generate multi-hop
    neighbor-averaged feature on this subgraph
    """

    new_edges = {}
    ntypes = set()
    for etype in rel_subset:
        stype, _, dtype = g.to_canonical_etype(etype)
        src, dst = g.all_edges(etype=etype)
        src = src.numpy()
        dst = dst.numpy()
        new_edges[(stype, etype, dtype)] = (src, dst)
        new_edges[(dtype, etype + "_r", stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)
    new_g = dgl.heterograph(new_edges)

    # set node feature and calc deg
    for ntype in ntypes:
        num_nodes = new_g.number_of_nodes(ntype)
        if num_nodes < g.nodes[ntype].data["feat"].shape[0]:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"][:num_nodes, :]
        else:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"]
        deg = 0
        for etype in new_g.etypes:
            _, _, dtype = new_g.to_canonical_etype(etype)
            if ntype == dtype:
                deg = deg + new_g.in_degrees(etype=etype)
        norm = 1.0 / deg.float()
        norm[torch.isinf(norm)] = 0
        new_g.nodes[ntype].data["norm"] = norm.view(-1, 1).to(device)

    res = []

    # compute k-hop feature
    for hop in range(1, args.K + 1):
        ntype2feat = {}
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)
            new_g[etype].update_all(fn.copy_u(f'hop_{hop-1}', 'm'), fn.sum('m', 'new_feat'))
            new_feat = new_g.nodes[dtype].data.pop("new_feat")
            assert("new_feat" not in new_g.nodes[stype].data)
            if dtype in ntype2feat:
                ntype2feat[dtype] += new_feat
            else:
                ntype2feat[dtype] = new_feat
        for ntype in new_g.ntypes:
            assert ntype in ntype2feat  # because subgraph is not directional
            feat_dict = new_g.nodes[ntype].data
            old_feat = feat_dict.pop(f"hop_{hop-1}")
            if ntype == "paper":
                res.append(old_feat.cpu())
            feat_dict[f"hop_{hop}"] = ntype2feat.pop(ntype).mul_(feat_dict["norm"])

    res.append(new_g.nodes["paper"].data.pop(f"hop_{args.K}").cpu())
    del new_g, feat_dict, new_edges
    clear_memory(device)
    return res

def prepare_data(device, args, probs_path, stage=0, load_embs=False, load_label_emb=False, subset_list=None):
    """
    Load dataset and compute neighbor-averaged node features used by scalable GNN model
    Note that we select only one integrated representation as node feature input for mlp 
    """
    aggr_device = torch.device("cpu" if args.aggr_gpu < 0 else "cuda:{}".format(args.aggr_gpu))
    emb_path = os.path.join(args.data_dir, '_'.join(args.dataset.split('-')), "embedding",
                f"smoothed_embs_K_{args.K}.pt")
    label_emb_path = os.path.join(args.data_dir, '_'.join(args.dataset.split('-')), "embedding", 
                f"smoothed_label_emb_K_{args.label_K}.pt")
    if not os.path.exists(os.path.dirname(emb_path)):
        os.makedirs(os.path.dirname(emb_path))

    data = load_dataset(aggr_device, args)
    t1 = time.time()
    
    g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
    # homophily = calculate_homophily(g, labels, 
    #                                 method="edge", 
    #                                 multilabels=args.dataset in ["ppi", "yelp"], 
    #                                 heterograph=args.dataset == "ogbn-mag")
    # print(f"initial homophily: {homophily}")
    tr_va_te_nid = torch.cat([train_nid, val_nid, test_nid], dim=0)
    
    feat_averaging_style = "all" if args.model != "mlp" else "ppnp"
    label_averaging_style = "last"
    if args.dataset == "ogbn-mag":
        train_mask = {"paper": torch.BoolTensor(np.isin(np.arange(len(labels)), train_nid)).to(aggr_device)}
        for node_type in ["author", "field_of_study", "institution"]:
            train_mask[node_type] = torch.zeros(size=(g.number_of_nodes(node_type),)).bool().to(aggr_device)
        g.ndata["train_mask"] = train_mask
        
        target_type_id = g.get_ntype_id("paper")
        homo_g = dgl.to_homogeneous(g, ndata=["feat", "train_mask"])
        homo_g = dgl.add_reverse_edges(homo_g, copy_ndata=True)
        # homo_g = dgl.add_self_loop(homo_g)
        homo_g.ndata["target_mask"] = homo_g.ndata[dgl.NTYPE] == target_type_id
        in_feats = g.ndata['feat']['paper'].shape[1]
        # n_classes = (labels.max() + 1).item() if labels.dim() == 1 else labels.size(1)
        print("in_feats:", in_feats)
        feat = g.ndata['feat']['paper']
    else:
        train_mask = torch.BoolTensor(np.isin(np.arange(len(labels)), train_nid))
        g.ndata["train_mask"] = train_mask.to(aggr_device)
        
        in_feats = g.ndata['feat'].shape[1]
        # n_classes = (labels.max() + 1).item() if labels.dim() == 1 else labels.size(1)
        print("in_feats:", in_feats)
        feat = g.ndata.pop('feat')

    
    if stage > 0:
        threshold = args.threshold[stage-1] if stage <= len(args.threshold) else args.threshold[-1]
        teacher_probs = torch.load(probs_path).to(aggr_device)
        

        # assert len(teacher_probs) == len(feat)
        if args.dataset in ['yelp', 'ppi', 'ppi_large']:
            threshold = - threshold * np.log(threshold) - (1-threshold) * np.log(1-threshold)
            entropy_distribution = entropy(teacher_probs)
            print(threshold)
            print(entropy_distribution.mean(1).max().item())
            
            confident_nid_inner = torch.arange(len(teacher_probs))[(entropy_distribution.mean(1) <= threshold)]
        else:
            # confident_nid_inner = []
            # entropy_distribution = (- teacher_probs * torch.log(teacher_probs + 1e-9)).sum(1)
            # confident_nid_inner = torch.argsort(entropy_distribution)[:2000000]
            # print(confident_nid_inner.shape)
            # for c in range(n_classes):
            #     print(f"class {c}: {teacher_probs[:, c].mean():.4f}, {teacher_probs[:, c].max():.4f}")
            #     if teacher_probs[:, c].max() < 0.9:
            #         continue
            #     confident_nid_inner.append(torch.argsort(teacher_probs[:, c], descending=True)[:70000])
            # confident_nid_inner = torch.cat(confident_nid_inner, dim=0).unique()
                
            confident_nid_inner = torch.arange(len(teacher_probs))[teacher_probs.max(1)[0] > threshold]
        extra_confident_nid_inner = confident_nid_inner[confident_nid_inner >= len(train_nid)]
        confident_nid = tr_va_te_nid[confident_nid_inner]
        extra_confident_nid = tr_va_te_nid[extra_confident_nid_inner]
        print(f"pseudo label number: {len(confident_nid)}")
        if args.dataset in ["yelp", "ppi", "ppi_large"]:
            pseudo_labels = teacher_probs
            # pseudo_labels[entropy_distribution > threshold] = 0.5
            pseudo_labels[pseudo_labels > 0.5] = 1
            pseudo_labels[pseudo_labels < 0.5] = 0
            labels_with_pseudos = torch.ones_like(labels) * 0.5
        else:
            pseudo_labels = torch.argmax(teacher_probs, dim=1).to(labels.device)
            labels_with_pseudos = torch.zeros_like(labels)
        train_nid_with_pseudos = np.union1d(train_nid, confident_nid)
        print(f"enhanced train set number: {len(train_nid_with_pseudos)}")
        labels_with_pseudos[train_nid] = labels[train_nid]
        labels_with_pseudos[extra_confident_nid] = pseudo_labels[extra_confident_nid_inner]
        # if args.dataset in ["yelp", "ppi", "ppi_large"]:
        #     # print((entropy_distribution[extra_confident_nid] > threshold).sum())
        #     labels_with_pseudos[extra_confident_nid][entropy_distribution[extra_confident_nid] > threshold] = 0.5
        
        # train_nid_with_pseudos = np.random.choice(train_nid_with_pseudos, size=int(0.5 * len(train_nid_with_pseudos)), replace=False)
    else:
        teacher_probs = None
        pseudo_labels = None
        labels_with_pseudos = labels.clone()
        confident_nid = train_nid
        train_nid_with_pseudos = train_nid
    
    if args.use_labels & ((not args.inductive) or stage > 0):
        print("using label information")
        if args.dataset in ["yelp", "ppi", "ppi_large"]:
            label_emb = 0.5 * torch.ones([feat.shape[0], n_classes]).to(labels.device)
            # label_emb = labels_with_pseudos.mean(0).repeat([feat.shape[0], 1])
            label_emb[train_nid_with_pseudos] = labels_with_pseudos.float()[train_nid_with_pseudos]

        else:
            label_emb = torch.zeros([feat.shape[0], n_classes]).to(labels.device)
            # label_emb = (1. / n_classes) * torch.ones([feat.shape[0], n_classes]).to(device)
            label_emb[train_nid_with_pseudos] = F.one_hot(labels_with_pseudos[train_nid_with_pseudos], num_classes=n_classes).float().to(labels.device)


        if args.dataset == "ogbn-mag":
            # rand_weight = torch.Tensor(n_classes, 128).uniform_(-0.5, 0.5)
            # label_emb = torch.matmul(label_emb, rand_weight.to(device))
            # pca = PCA(n_components=128)
            # label_emb = torch.FloatTensor(pca.fit_transform(label_emb.cpu())).to(device)
            target_mask = homo_g.ndata["target_mask"]
            target_ids = homo_g.ndata[dgl.NID][target_mask]
            num_target = target_mask.sum().item()
            new_label_emb = torch.zeros((len(homo_g.ndata["feat"]),) + label_emb.shape[1:],
                                dtype=label_emb.dtype, device=label_emb.device)
            new_label_emb[target_mask] = label_emb[target_ids]
            label_emb = new_label_emb
    else:
        label_emb = None
    
    if args.inductive:
        # This setting is not compatible with ogbn-mag!
        print("Inductive setting detected")
        if os.path.exists(os.path.join("../subgraphs",args.dataset, "subgraph_train.pt")):
            print("Load train subgraph")
            g_train = torch.load(os.path.join("../subgraphs",args.dataset, "subgraph_train.pt")).to(g.device)
        else:
            print("Extract train subgraph")
            g_train = dgl.node_subgraph(g, train_nid.to(g.device))
            if not os.path.exists(os.path.join("../subgraphs",args.dataset)):
                os.makedirs(os.path.join("../subgraphs",args.dataset))
            torch.save(g_train, os.path.join("../subgraphs",args.dataset, "subgraph_train.pt"))
        # print("get val/test subgraph")
        # g_val_test = dgl.node_subgraph(g, torch.cat([val_nid, test_nid],dim=0).to(g.device))
        
        train_mask = g_train.ndata[dgl.NID]
        if load_embs and os.path.exists(emb_path):
            pass
        else:
            feats = neighbor_average_features_by_chunks(g, feat, args, style=feat_averaging_style, stats=args.dataset not in ["ogbn-mag", "ogbn-papers100M"], memory_efficient=args.memory_efficient)
            feats_train = neighbor_average_features_by_chunks(g_train, feat[g_train.ndata[dgl.NID]], args, style=feat_averaging_style, stats=args.dataset not in ["ogbn-mag", "ogbn-papers100M"], memory_efficient=args.memory_efficient)
            if args.model in ["sagn", "simple_sagn", "sign"]:
                for i in range(args.K+1):
                    feats[i][train_mask] = feats_train[i]
            else:
                feats[train_mask] = feats_train
            if load_embs:
                if not os.path.exists(emb_path):
                    print("saving smoothed node features to " + emb_path)
                    torch.save(feats, emb_path)
                del feats, feat
                clear_memory(device)
                
        
        if (stage == 0) and load_label_emb and os.path.exists(label_emb_path):
            pass
        else:
            if label_emb is not None:
                if args.dataset in ["ppi", "yelp"]:
                    label_emb = torch.cat([label_emb, 1-label_emb], dim=1)
                label_emb_train = label_emb[g_train.ndata[dgl.NID]]
                # mask_rate = len(train_nid) * 1. / len(labels)
                # mask_rate = 0.5
                # inner_train_nid = torch.arange(len(train_nid))
                # mask = torch.rand(len(train_nid)) > mask_rate
                # if args.dataset in ["ppi", "yelp"]:
                #     label_emb_train[mask] = 0.5 * torch.ones_like(label_emb_train[mask])
                #     # label_emb_train[mask] = torch.rand_like(label_emb_train[mask])
                # else:
                #     label_emb_train[mask] = torch.zeros_like(label_emb_train[mask])
                label_emb_train = neighbor_average_features_by_chunks(g_train, label_emb_train, args, style=label_averaging_style,stats=args.dataset not in ["ogbn-mag", "ogbn-papers100M"], memory_efficient=args.memory_efficient)
                label_emb = neighbor_average_features_by_chunks(g, label_emb, args, style=label_averaging_style,stats=args.dataset not in ["ogbn-mag", "ogbn-papers100M"], memory_efficient=args.memory_efficient)
                if args.dataset in ["ppi", "yelp"]:
                    # scale = torch.norm(label_emb, p=2, dim=1).mean() / torch.norm(label_emb_train, p=2, dim=1).mean()
                    # label_emb_train = label_emb_train * scale

                    # label_emb_train = (label_emb_train - label_emb_train.mean(0)) / label_emb_train.std(0)
                    # label_emb = (label_emb - label_emb.mean(0)) / label_emb.std(0)
                    
                    # label_emb_train = label_emb_train[:,:n_classes]
                    # label_emb = label_emb[:, :n_classes]
                    
                    label_emb_train = F.normalize(label_emb_train.view(-1, 2, n_classes), p=1, dim=1)[:, 0, :]
                    label_emb = F.normalize(label_emb.view(-1, 2, n_classes), p=1, dim=1)[:, 0, :]
                    # label_emb_train = label_emb_train[:, [0]]
                    # label_emb = label_emb[:, [0]]
                    
                    # label_emb_train = F.normalize(label_emb_train, p=1, dim=1)
                    # label_emb = F.normalize(label_emb, p=1, dim=1)
                    # pass
                else:
                    label_emb_train = F.normalize(label_emb_train, p=1, dim=1)
                    label_emb = F.normalize(label_emb, p=1, dim=1)
                
                
                label_emb[train_mask] = label_emb_train
            if load_label_emb:
                if not os.path.exists(label_emb_path):
                    print("saving initial label embeddings to " + label_emb_path)
                    torch.save(label_emb, label_emb_path)
                del label_emb
                clear_memory(device)
        
    else:
        # for transductive setting
        
        if (stage == 0) and load_label_emb and os.path.exists(label_emb_path):
            pass
        else:
            if label_emb is not None:
                label_emb = neighbor_average_features_by_chunks(g if args.dataset != "ogbn-mag" else homo_g, label_emb, args, 
                                                                    style=label_averaging_style, 
                                                                    stats=args.dataset not in ["ogbn-mag", "ogbn-papers100M"],
                                                                    memory_efficient=args.memory_efficient,
                                                                    target_nid=tr_va_te_nid if args.dataset=="ogbn-papers100M" else None)
                if args.dataset == "ogbn-mag":
                    del homo_g
                    clear_memory(device)
                # if args.dataset == "ogbn-papers100M":            
                #     label_emb = label_emb[tr_va_te_nid]
            if load_label_emb and stage == 0: 
                if (not os.path.exists(label_emb_path)):
                    print("saving initial label embeddings to " + label_emb_path)
                    torch.save(label_emb, label_emb_path)
                del label_emb
                clear_memory(device)

        if load_embs and os.path.exists(emb_path):
            pass
        else:
            if args.dataset == "ogbn-mag":
                feats = {("raw",): [feat]}
                for rel_subset in subset_list:
                    print(f"Preprocessing subgraph of {rel_subset}...")
                    feats[rel_subset] = gen_rel_subset_feature(g, rel_subset, args, aggr_device)
            else:
                feats = neighbor_average_features_by_chunks(g, feat, args, 
                                                                style=feat_averaging_style, 
                                                                stats=args.dataset not in ["ogbn-mag", "ogbn-papers100M"],
                                                                memory_efficient=args.memory_efficient,
                                                                target_nid=tr_va_te_nid if args.dataset=="ogbn-papers100M" else None)
            # if args.dataset == "ogbn-papers100M":
                
            #     feats = [feat[tr_va_te_nid] for feat in feats]
            if load_embs:
                if not os.path.exists(emb_path):
                    print("saving smoothed node features to " + emb_path)
                    torch.save(feats, emb_path)
                del feats, feat
                clear_memory(device)
        del g
        clear_memory(device)
        
        # if args.save_temporal_emb:
        #     torch.save(feats, emb_path)
            
    
    # save smoothed node features and initial smoothed node label embeddings, 
    # if "load" is set true and they have not been saved
 
    if load_embs:
        print("load saved embeddings")
        feats = torch.load(emb_path)
    if load_label_emb and (stage == 0):
        print("load saved label embedding")
        label_emb = torch.load(label_emb_path)

    # label_emb = (label_emb - label_emb.mean(0)) / label_emb.std(0)
    # eval_feats = neighbor_average_features(g, eval_feat, args)
    labels = labels.to(device)
    labels_with_pseudos = labels_with_pseudos.to(device)
    # move to device

    train_nid = train_nid.to(device)
    train_nid_with_pseudos = torch.LongTensor(train_nid_with_pseudos).to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)
    t2 = time.time()

    return feats, label_emb, teacher_probs, labels, labels_with_pseudos, in_feats, n_classes, \
        train_nid, train_nid_with_pseudos, val_nid, test_nid, evaluator, t2 - t1
