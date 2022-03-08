import os
import time

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train(model, feats, label_emb, teacher_probs, labels, loss_fcn, optimizer, train_loader, args):
    model.train()
    device = labels.device
    for batch in train_loader:
        if len(batch) == 1:
            continue
        if args.dataset == "ogbn-mag":
            batch_feats = {rel_subset: [x[batch].to(device) for x in feat] for rel_subset, feat in feats.items()}
        else:
            batch_feats = [x[batch].to(device) for x in feats] if isinstance(feats, list) else feats[batch].to(device)
        if label_emb is not None:
            batch_label_emb = label_emb[batch].to(device)
        else:
            batch_label_emb = None
        
        if (args.model in ["sagn", "plain_sagn"]) and (not args.avoid_features):
            out, _ = model(batch_feats, batch_label_emb)
        else:
            out = model(batch_feats, batch_label_emb)
        loss = loss_fcn(out, labels[batch])
        # T = 0.5
        # alpha = 0.5
        # if teacher_probs is not None:
        #     loss = (1-alpha) * loss + alpha * F.kl_div(F.log_softmax(out, dim=-1) / T, teacher_probs[batch] / T) * (T * T)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, feats, label_emb, teacher_probs, labels, loss_fcn, val_loader, test_loader, evaluator,
         train_nid, val_nid, test_nid, args):
    model.eval()
    num_nodes = labels.shape[0]
    device = labels.device
    loss_list = []
    count_list = []
    preds = []
    for batch in val_loader:
        if args.dataset == "ogbn-mag":
            batch_feats = {rel_subset: [x[batch].to(device) for x in feat] for rel_subset, feat in feats.items()}
        else:
            batch_feats = [x[batch].to(device) for x in feats] if isinstance(feats, list) else feats[batch].to(device)
        if label_emb is not None:
            batch_label_emb = label_emb[batch].to(device)
        else:
            batch_label_emb = None
        # We can get attention scores from SAGN
        if (args.model in ["sagn", "plain_sagn"]) and (not args.avoid_features):
            out, _ = model(batch_feats, batch_label_emb)
        else:
            out = model(batch_feats, batch_label_emb)
        loss_list.append(loss_fcn(out, labels[batch]).cpu().item())
        count_list.append(len(batch))
    loss_list = np.array(loss_list)
    count_list = np.array(count_list)
    val_loss = (loss_list * count_list).sum() / count_list.sum()
    start = time.time()
    for batch in test_loader:
        if args.dataset == "ogbn-mag":
            batch_feats = {rel_subset: [x[batch].to(device) for x in feat] for rel_subset, feat in feats.items()}
        else:
            batch_feats = [x[batch].to(device) for x in feats] if isinstance(feats, list) else feats[batch].to(device)
        if label_emb is not None:
            batch_label_emb = label_emb[batch].to(device)
        else:
            batch_label_emb = None
        if (args.model in ["sagn", "plain_sagn"]) and (not args.avoid_features):
            out, _ = model(batch_feats, batch_label_emb)
        else:
            out = model(batch_feats, batch_label_emb)
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            preds.append((out > 0).float())
        else:
            preds.append(torch.argmax(out, dim=-1))

    # Concat mini-batch prediction results along node dimension
    preds = torch.cat(preds, dim=0)
    end = time.time()
    train_res = evaluator(preds[:len(train_nid)], labels[train_nid])
    val_res = evaluator(preds[len(train_nid):(len(train_nid)+len(val_nid))], labels[val_nid])
    test_res = evaluator(preds[(len(train_nid)+len(val_nid)):], labels[test_nid])
    return train_res, val_res, test_res, val_loss, end - start
