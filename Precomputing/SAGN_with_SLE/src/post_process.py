import argparse
import glob
import os

import numpy as np
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from tqdm import tqdm

from dataset import load_dataset
from models import CorrectAndSmooth
import gc

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_output_files(output_path):
    outputs = glob.glob(output_path)
    print(f"Detect {len(outputs)} model output files")
    assert len(outputs) > 0
    probs_list = []
    for out in outputs:
        # probs = torch.zeros(size=(n_nodes, n_classes), device="cpu")
        # probs[tr_va_te_nid] = torch.load(out, map_location="cpu")
        probs = torch.load(out, map_location="cpu")
        probs_list.append(probs)
        # mx_diff = (out_probs[-1].sum(dim=-1) - 1).abs().max()
        # if mx_diff > 1e-1:
        #     print(f'Max difference: {mx_diff}')
        #     print("model output doesn't seem to sum to 1. Did you remember to exp() if your model outputs log_softmax()?")
        #     raise Exception
    return probs_list

def generate_preds_path(args):
    path = os.path.join(args.probs_dir, args.dataset, 
                        args.model if (args.weight_style == "attention") \
                            else (args.model + "_" + args.weight_style),
                        f"use_labels_{args.use_labels}_use_feats_{not args.avoid_features}"+
                        f"_K_{args.K}_label_K_{args.label_K}_probs_seed_*_stage_{args.stage}.pt")
    return path

def calculate_metrics(probs_list, labels, train_nid, val_nid, test_nid, evaluator, args):
    train_results = []
    val_results = []
    test_results = []
    inner_train_nid = torch.arange(len(train_nid))
    inner_val_nid = torch.arange(len(train_nid), len(train_nid)+len(val_nid))
    inner_test_nid = torch.arange(len(train_nid)+len(val_nid), len(train_nid)+len(val_nid)+len(test_nid))
    for probs in probs_list:
        if args.dataset in ["ppi", "yelp"]:
            preds = (probs > 0).float()
        else:
            preds = torch.argmax(probs, dim=-1)
        train_res = evaluator(preds[inner_train_nid], labels[train_nid])
        val_res = evaluator(preds[inner_val_nid], labels[val_nid])
        test_res = evaluator(preds[inner_test_nid], labels[test_nid])
        train_results.append(train_res)
        val_results.append(val_res)
        test_results.append(test_res)
    print(f"Train score: {np.mean(train_results):.4f}±{np.std(train_results):.4f}\n"
          f"Valid score: {np.mean(val_results):.4f}±{np.std(val_results):.4f}\n"
          f"Test score: {np.mean(test_results):.4f}±{np.std(test_results):.4f}")
    return

def main(args):
    device = torch.device("cpu" if args.gpu < 0 else f"cuda:{args.gpu}")
    data = load_dataset(device, args)
    g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
    g.ndata.pop("feat")
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    labels = labels.to(device)
    tr_va_te_nid = torch.cat([train_nid, val_nid, test_nid], dim=0)
    preds_path = generate_preds_path(args)
    print(preds_path)
    probs_list = load_output_files(preds_path)
    print("-"*10 + " Before " + "-"*10)
    calculate_metrics(probs_list, labels, train_nid, val_nid, test_nid, evaluator, args)

    cs = CorrectAndSmooth(args.num_correction_layers, args.correction_alpha, args.correction_adj,
                          args.num_smoothing_layers, args.smoothing_alpha, args.smoothing_adj,
                          autoscale=args.autoscale, scale=args.scale)
    processed_preds_list = []
    for i, probs in enumerate(probs_list):
        print(f"Processing run: {i}")
        probs = probs.to(device)
        processed_preds_list.append(cs(g, probs, labels[train_nid], args.operations, train_nid, val_nid, test_nid, len(labels)))
    print("-"*10 + " Correct & Smooth " + "-"*10)
    calculate_metrics(processed_preds_list, labels, train_nid, val_nid, test_nid, evaluator, args)
    
    
def define_parser():
    parser = argparse.ArgumentParser(description="hyperparameters for Correct&Smooth postprocessing")
    parser.add_argument("--gpu", type=int, default=-1, 
                        help="Select which GPU device to process (-1 for CPU)")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv", 
                        help="Dataset name")
    parser.add_argument("--data-dir", type=str, default="/mnt/ssd/ssd/dataset",
                        help="Root directory for datasets")
    parser.add_argument("--probs_dir", type=str, default="../intermediate_outputs",
                        help="Directory of trained model output")
    parser.add_argument("--model", type=str, default="sagn", 
                        help="Model name")
    parser.add_argument("--weight_style", type=str, default="attention", 
                        help="Weight style for SAGN and PlainSAGN")
    parser.add_argument("--K", type=int, default=5, 
                        help="Maximum hop for feature propagation")
    parser.add_argument("--label-K", type=int, default=9, 
                        help="Maximum hop for label propagation (in SLE)")
    parser.add_argument("--stage", type=int, default=0, 
                        help="Which stage in SLE to postprocess")
    parser.add_argument("--use-labels", action="store_true", 
                        help="Whether to enhance base model with a label model")
    parser.add_argument("--avoid-features", action="store_true", 
                        help="Whether to ignore node features (only useful when using labels)")
    parser.add_argument("--num-correction-layers", type=int, default=50, 
                        help="Propagation number for Correction operation")
    parser.add_argument("--num-smoothing-layers", type=int, default=50, 
                        help="Propagation number for Smoothing operation")
    parser.add_argument("--correction-alpha", type=float, default=1.0, 
                        help="Alpha value for Correction operation")
    parser.add_argument("--smoothing-alpha", type=float, default=0.7,
                        help="Alpha value for Smoothing operation")
    parser.add_argument("--correction-adj", type=str, default="DAD", choices=["DA", "AD", "DAD"], 
                        help="Adjacency matrix for Correction operation")
    parser.add_argument("--smoothing-adj", type=str, default="DAD", choices=["DA", "AD", "DAD"], 
                        help="Adjacency matrix for Smoothing operation")
    parser.add_argument("--scale", type=float, default=20, 
                        help="Fixed scale for Correction operation (only useful when autoscale=False")
    parser.add_argument("--autoscale", action="store_true", 
                        help="Whether to use autoscale in Correction operation")
    parser.add_argument("--operations", type=str, nargs="+", default=["correction", "smoothing"], 
                        help="Select operations needed")
    return parser


if __name__ == "__main__":
    parser = define_parser()
    args = parser.parse_args()
    print(args)
    main(args)
