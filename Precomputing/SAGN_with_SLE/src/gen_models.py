from torch.autograd.grad_mode import F
from torch.nn.init import zeros_
from models import SIGN, SAGN, PlainSAGN, MLP, MultiHeadMLP, GroupMLP, NARS, SLEModel

def get_model(in_feats, label_in_feats, n_classes, stage, args, subset_list=None):
    num_hops = args.K + 1
    use_labels = args.use_labels and ((not args.inductive) or stage > 0)

    if args.model == "sagn":
        base_model = SAGN(in_feats, args.num_hidden, n_classes, num_hops,
                        args.mlp_layer, args.num_heads, 
                        weight_style=args.weight_style,
                        dropout=args.dropout, 
                        input_drop=args.input_drop, 
                        attn_drop=args.attn_drop,
                        zero_inits=args.zero_inits,
                        position_emb=args.position_emb,
                        focal=args.focal)
    
    if args.model == "mlp":
        base_model = MLP(in_feats, args.num_hidden, n_classes,
                        args.mlp_layer,  
                        args.dropout, 
                        residual=False,
                        input_drop=args.input_drop)
            
    if args.model == "plain_sagn":
        base_model = PlainSAGN(in_feats, args.num_hidden, n_classes,
                        args.mlp_layer, args.num_heads, 
                        dropout=args.dropout, 
                        input_drop=args.input_drop, 
                        attn_drop=args.attn_drop,
                        zero_inits=args.zero_inits)
    
    if args.model == "sign":
        base_model = SIGN(in_feats, args.num_hidden, n_classes, num_hops,
                        args.mlp_layer, 
                        dropout=args.dropout, 
                        input_drop=args.input_drop)
    
    if args.avoid_features:
        base_model = None
    
    if args.dataset == "ogbn-mag":
        assert base_model is not None
        base_model = NARS(in_feats, num_hops, base_model, subset_list)

    if use_labels:
        label_model = GroupMLP(label_in_feats, 
                                   args.num_hidden, 
                                   n_classes, 
                                   args.num_heads, 
                                   args.label_mlp_layer, 
                                   args.label_drop,
                                   residual=args.label_residual,)
    else:
        label_model = None

    model = SLEModel(base_model, label_model)

    return model