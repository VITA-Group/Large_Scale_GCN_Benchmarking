import argparse


class BaseOptions:
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""

    def initialize(self):
        parser = argparse.ArgumentParser(description="Constrained learing")

        parser.add_argument(
            "--debug_mem_speed",
            action="store_true",
            help="whether to get the memory usage and throughput",
        )
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--tosparse", action="store_true")
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            choices=[
                "Flickr",
                "Reddit",
                "ogbn-products",
                "ogbn-papers100M",
                "ogbn-arxiv",
            ],
        )

        parser.add_argument(
            "--type_model",
            type=str,
            required=True,
            choices=[
                "GraphSAGE",
                "FastGCN",
                "LADIES",
                "ClusterGCN",
                "GraphSAINT",
                "SGC",
                "SIGN",
                "SIGN_MLP",
                "LP_Adj",
                "SAGN",
                "GAMLP",
                "EnGCN",
            ],
        )
        parser.add_argument("--exp_name", type=str, default="")
        parser.add_argument("--N_exp", type=int, default=20)
        parser.add_argument("--resume", action="store_true", default=False)
        parser.add_argument("--cuda", type=bool, default=True, required=False, help="run in cuda mode")
        parser.add_argument("--cuda_num", type=int, default=0, help="GPU number")

        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument(
            "--epochs",
            type=int,
            default=50,
            help="number of training the one shot model",
        )
        parser.add_argument(
            "--eval_steps",
            type=int,
            default=5,
            help="interval steps to evaluate model performance",
        )

        parser.add_argument(
            "--multi_label",
            type=bool,
            default=False,
            help="multi_label or single_label task",
        )
        parser.add_argument("--dropout", type=float, default=0.2, help="input feature dropout")
        parser.add_argument("--norm", type=str, default="None")
        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")  # 5e-4
        parser.add_argument("--dim_hidden", type=int, default=128)
        parser.add_argument(
            "--batch_size",
            type=int,
            default=20000,
            help="batch size depending on methods, " "need to provide fair batch for different approaches",
        )
        # parameters for GraphSAINT
        parser.add_argument("--walk_length", type=int, default=2, help="walk length of RW sampler")
        parser.add_argument("--num_steps", type=int, default=5)
        parser.add_argument("--sample_coverage", type=int, default=0)
        parser.add_argument("--use_norm", type=bool, default=False)
        # parameters for ClusterGCN
        parser.add_argument("--num_parts", type=int, default=1500)
        # parameters for Greedy Gradient Sampling Selection
        parser.add_argument("--dst_sample_coverage", type=float, default=0.1, help="dst sampling rate")
        parser.add_argument("--dst_walk_length", type=int, default=2, help="random walk length")
        parser.add_argument(
            "--dst_update_rate",
            type=float,
            default=0.8,
            help="initialized dst update rate",
        )
        parser.add_argument("--dst_update_interval", type=int, default=1, help="dst update interval")
        parser.add_argument("--dst_T_end", type=int, default=250)
        parser.add_argument(
            "--dst_update_decay",
            type=bool,
            default=True,
            help="whether to decay update rate",
        )
        parser.add_argument("--dst_update_scheme", type=str, default="node3", help="update schemes")
        parser.add_argument(
            "--dst_grads_scheme",
            type=int,
            default=3,
            help="tem: search for updating scheme with grads",
        )

        parser.add_argument("--LP__no_prep", type=int, default=0)  # no change!!!
        parser.add_argument("--LP__pre_num_propagations", type=int, default=10)  # no change!!!
        parser.add_argument("--LP__A1", type=str, default="DA")  # ['DA' 'AD' 'DAD']
        parser.add_argument("--LP__A2", type=str, default="AD")  # ['DA' 'AD' 'DAD']
        parser.add_argument("--LP__prop_fn", type=int, default=1)  # [0,1]
        parser.add_argument("--LP__num_propagations1", type=int, default=50)
        parser.add_argument("--LP__num_propagations2", type=int, default=50)
        parser.add_argument("--LP__alpha1", type=float, default=0.9791632871592579)
        parser.add_argument("--LP__alpha2", type=float, default=0.7564990804200602)
        parser.add_argument("--LP__num_layers", type=int, default=3)  # [0,  1,2,3]

        parser.add_argument("--SLE_threshold", type=float, default=0.9)
        parser.add_argument("--num_mlp_layers", type=int, default=3)
        parser.add_argument("--use_batch_norm", type=bool, default=True)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--use_label_mlp", type=bool, default=True)

        parser.add_argument("--GAMLP_type", type=str, default="JK", choices=["JK", "R"])
        parser.add_argument("--GAMLP_alpha", type=float, default=0.5)
        parser.add_argument("--GPR_alpha", type=float, default=0.1)
        parser.add_argument(
            "--GPR_init",
            type=str,
            default="PPR",
            choices=["SGC", "PPR", "NPPR", "Random", "WS", "Null"],
        )  # [0,  1,2,3]
        # hyperparameters for gradient evaluation
        parser.add_argument("--type_run", type=str, default="filtered", choices=["complete", "filtered"])
        parser.add_argument("--filter_rate", type=float, default=0.2)

        args = parser.parse_args()
        args = self.reset_dataset_dependent_parameters(args)

        if args.type_model == "LP_Adj":
            set_labprop_configs(args)

        return args

    # setting the common hyperparameters used for comparing different methods of a trick
    def reset_dataset_dependent_parameters(self, args):
        if args.dataset == "Flickr":
            args.num_classes = 7
            args.num_feats = 500
            # args.dropout = 0.5
            # args.batch_size = 10000
            # args.num_layers = 4

        elif args.dataset == "Reddit":
            args.num_classes = 41
            args.num_feats = 602

        elif args.dataset == "ogbn-products":
            args.multi_label = False
            args.num_classes = 47
            args.num_feats = 100

        elif args.dataset == "AmazonProducts":
            args.multi_label = True
            args.num_classes = 107
            args.num_feats = 200

        elif args.dataset == "ogbn-arxiv":  # I added this for debug purpose. -Wenqing
            args.num_feats = 128
            args.num_classes = 40
            args.N_nodes = 169343
            # args.dim_hidden = 256

        elif args.dataset == "ogbn-papers100M":
            args.multi_label = False
            args.num_classes = 172
            args.num_feats = 128

        return args


def set_labprop_configs(args):
    class C:
        pass

    args.preStep = C()
    args.lpStep = C()
    args.midStep = C()

    args.lr = 0.01
    args.dropout = 0.5
    args.weight_decay = 0

    use_default_LP_settings = 0

    if use_default_LP_settings:
        print("\n\n  using default config!!! \n\n")
        # args.lp_has_prep = 1
        args.preStep.num_propagations = 10
        # args.preStep.p = 1
        # args.preStep.alpha = 0.5
        args.preStep.pre_methods = "diffusion+spectral"  # options: sgc , diffusion , spectral , community
        args.midStep.model = ["mlp", "linear", "plain", "gat"][0]
        args.midStep.hidden_channels = 256
        args.midStep.num_layers = 3
        args.lpStep.fn = [
            "double_correlation_fixed",
            "double_correlation_autoscale",
            "only_outcome_correlation",
        ][1]
        args.lpStep.A = "DAD"
        args.lpStep.A1 = "DA"
        args.lpStep.A2 = "AD"
        args.lpStep.alpha = 0.5
        args.lpStep.alpha1 = 0.9791632871592579
        args.lpStep.alpha2 = 0.7564990804200602
        args.lpStep.num_propagations = 50
        args.lpStep.num_propagations1 = 50
        args.lpStep.num_propagations2 = 50
        args.lpStep.lp_force_on_cpu = True  # fixed due to hard coding in C&S. please never change this.
        args.lpStep.no_prep = 0
        # if the above 'lpStep.no_prep' is set to 1, what will happen:
        # there will be no preprocessing (self.preStep);
        # no MLP (self.midStep);
        # the node features are never considered;
        # it will only take the label-propagation, with initialization of zero vectors at test nodes, and true labels at train nodes.
    else:

        # LP__pre_num_propagations 10
        # LP__num_layers [0,  1,2,3]
        # LP__prop_fn [0,1]
        # LP__A1 ['DA' 'AD' 'DAD']
        # LP__A2 ['DA' 'AD' 'DAD']
        # LP__alpha1 0.9791632871592579
        # LP__alpha2 0.7564990804200602
        # LP__num_propagations1 50
        # LP__num_propagations2 50
        # LP__no_prep 0

        # no_prep
        # midStep.model
        # fn (LP__prop_function)
        # num_propagations1
        # num_propagations2
        # A1
        # A2
        # alpha1
        # alpha2
        # num_layers
        args.preStep.num_propagations = args.LP__pre_num_propagations
        args.preStep.pre_methods = "diffusion+spectral"  # options: sgc , diffusion , spectral , community
        if args.LP__num_layers == 0:
            args.midStep.model = ["mlp", "linear", "plain", "gat"][1]
        else:
            args.midStep.model = ["mlp", "linear", "plain", "gat"][0]
        args.midStep.hidden_channels = 256
        args.midStep.num_layers = args.LP__num_layers
        args.lpStep.fn = [
            "double_correlation_fixed",
            "double_correlation_autoscale",
            "only_outcome_correlation",
        ][args.LP__prop_fn]
        args.lpStep.A = "DAD"
        args.lpStep.A1 = args.LP__A1
        args.lpStep.A2 = args.LP__A2
        args.lpStep.alpha = 0.5
        args.lpStep.alpha1 = args.LP__alpha1
        args.lpStep.alpha2 = args.LP__alpha1
        args.lpStep.num_propagations = 50
        args.lpStep.num_propagations1 = args.LP__num_propagations1
        args.lpStep.num_propagations2 = args.LP__num_propagations2
        args.lpStep.lp_force_on_cpu = True  # fixed due to hard coding in C&S. please never change this.
        args.lpStep.no_prep = args.LP__no_prep


# def bestGPU(gpu_verbose=True, is_bestMem=True, **w):
#     import GPUtil
#     Gpus = GPUtil.getGPUs()
#     Ngpu = 4
#     mems, loads = [], []
#     for ig, gpu in enumerate(Gpus):
#         memUtil = gpu.memoryUtil*100
#         load = gpu.load*100
#         mems.append(memUtil)
#         loads.append(load)
#         if gpu_verbose: print(f'gpu-{ig}:   Memory: {memUtil:.2f}%   |   load: {load:.2f}% ')
#     bestMem = np.argmin(mems)
#     bestLoad = np.argmin(loads)
#     best = bestMem if is_bestMem else bestLoad
#     if gpu_verbose: print(f'//////   Will Use GPU - {best}  //////')
#     # print(type(best))

#     return int(best)
