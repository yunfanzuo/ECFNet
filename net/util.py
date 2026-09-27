import torch

from net.baseline.DGCNN import DGCNN
from net.baseline.GCBNet_BLS import GCBNet_BLS
from net.baseline.RGNN import RGNN
from net.baseline.TSCeption import TSception
from net.emt import EmT
from net.model import ECFNet, MyModel, MyModelCogFusion
from utils.channel import get_ch_name, get_local_ch_num

def create_model(args) -> torch.nn.Module:
    dataset = args.dataset.name
    dataset_args = args.dataset
    model_args = args.model
    graph_type = model_args.resolve_graph_type(dataset_args.graph_type)
    freq_bands = dataset_args.freq_bands
    dropout = args.training.dropout
    raw_fs = {"SEED": 200, "MLED": 128}.get(dataset.upper())
    sampling_rate = dataset_args.downsample or raw_fs
    args = model_args

    if args.name == 'ECFNet':
        model = ECFNet(
            num_bands=len(freq_bands), num_classes=args.num_classes,
            area_nodes=get_local_ch_num(dataset, graph_type), K=args.K,
            attn_heads=args.attn_heads, head_dim=args.head_dim, fusion=args.fusion,
            dropout=dropout, share_encoders=args.share_encoders,
            use_descriptors=args.use_descriptors,
        )
    elif args.name == 'MyModel':
        num_bands = len(freq_bands)
        area_nodes = get_local_ch_num(dataset, graph_type)
        model = MyModel(
            num_bands=num_bands,
            num_classes=args.num_classes,
            area_nodes=area_nodes,
            attn_layers=args.attn_layers,
            attn_heads=args.attn_heads,
            conv_K=args.conv_K,
            dropout=dropout,
            share_encoders=args.share_encoders,
        )
    elif args.name == 'MyModelCogFusion':
        num_bands = len(freq_bands)
        area_nodes = get_local_ch_num(dataset, graph_type)
        model = MyModelCogFusion(
            num_bands=num_bands,
            num_classes=args.num_classes,
            area_nodes=area_nodes,
            attn_heads=args.attn_heads,
            conv_K=args.conv_K,
            cog_dim=args.cog_dim,
            fusion=args.fusion,
            dropout=dropout,
            share_encoders=args.share_encoders,
        )
    elif args.name == 'EmT':
        num_bands = len(freq_bands)
        area_nodes = get_local_ch_num(dataset, graph_type)
        model = EmT(
            layers_graph=args.layers_graph,
            layers_transformer=args.layers_transformer,
            num_chan=sum(area_nodes),
            num_feature=num_bands,
            hidden_graph=args.hidden_graph,
            K=args.cheby_K,
            num_head=args.num_head,
            dim_head=args.dim_head,
            dropout=dropout,
            num_class=args.num_classes,
            alpha=args.sta_alpha,
            graph2token=args.graph2token,
            encoder_type=args.encoder_type,
        )
    elif args.name == 'DGCNN':
        model = DGCNN(
            num_electrodes=len(get_ch_name(dataset, graph_type)),
            in_channels=len(freq_bands),
            num_classes=args.num_classes,
            dropout_rate=dropout,
        )
    elif args.name == 'GCBNet_BLS':
        model = GCBNet_BLS(
            num_electrodes=len(get_ch_name(dataset, graph_type)),
            in_channels=len(freq_bands),
            num_classes=args.num_classes,
            dropout_rate=dropout,
        )
    elif args.name == 'RGNN':
        if dataset.upper() != "SEED":
            raise ValueError("RGNN baseline is currently only adapted for the SEED dataset.")
        model = RGNN(
            num_electrodes=len(get_ch_name(dataset, graph_type)),
            in_channels=len(freq_bands),
            num_classes=args.num_classes,
        )
    elif args.name == 'TSCeption':
        model = TSception(
            num_classes=args.num_classes,
            input_size=(1, len(get_ch_name(dataset, graph_type)), int(dataset_args.segment * sampling_rate)),
            sampling_rate=sampling_rate,
            dropout_rate=dropout,
        )
    else:
        raise ValueError(f"Not supported model: {args.name}")
    
    return model
