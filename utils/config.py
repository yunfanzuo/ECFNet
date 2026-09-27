from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

DE_BASELINE_MODELS = {"DGCNN", "GCBNet_BLS", "RGNN"}
RAW_BASELINE_MODELS = {"TSCeption"}
BASELINE_MODELS = DE_BASELINE_MODELS | RAW_BASELINE_MODELS


def load_config(path: Path) -> "RunConfig":
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    
    cfg = _read_config_tree(path, set())
    model_name = cfg['model']['name']
    model_cfg_map = {
        'MyModel': MyModelConfig,
        'EmT': EmTConfig,
        'MyModelCogFusion': MyModelCogFusionConfig,
        'ECFNet': ECFNetConfig,
        'DGCNN': DGCNNConfig,
        'GCBNet_BLS': GCBNetBLSConfig,
        'RGNN': RGNNConfig,
        'TSCeption': TSCeptionConfig,
    }
    if model_name not in model_cfg_map:
        raise ValueError(f"Not supported model config: {model_name}")
    model_cfg_cls = model_cfg_map[model_name]

    args = RunConfig(
        dataset=DatasetConfig.from_args(cfg['dataset']),
        training=TrainingConfig.from_args(cfg['training']),
        model=model_cfg_cls.from_args(cfg['model']),
        reproduce=ReproduceConfig.from_args(cfg['reproduce']),
        logging=LoggingConfig.from_args(cfg['logging']),
    )
    if args.model.name == "ECFNet":
        required = args.model.required_features
        if args.dataset.features != required or args.model.features != required:
            raise ValueError(f"ECFNet features must be ordered as {required}")
        if args.dataset.name != "SEED" or args.dataset.downsample not in {None, 200}:
            raise ValueError("The paper ECFNet implementation requires SEED at 200 Hz")
        if args.dataset.graph_type == "TS" or args.model.num_classes != 3:
            raise ValueError("The paper uses all 62 electrodes and three classes")
        if list(args.dataset.freq_bands.values()) != [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]:
            raise ValueError("ECFNet requires the five paper DE bands, in order")
        if (args.dataset.segment, args.dataset.segment_step, args.dataset.sequence, args.dataset.sequence_step) != (20, 4, 2, 0.5):
            raise ValueError("ECFNet uses 20-s segments (stride 4 s), 2-s windows (stride 0.5 s)")
        if args.model.class_names != ["negative", "neutral", "positive"]:
            raise ValueError("SEED class order is negative, neutral, positive")
    return args


def _merge_config(base, update):
    import copy
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_config_tree(path, seen):
    import json
    import yaml
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Circular configuration inheritance: {path}")
    if path.suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError(f"Unsupported config file format: {path.suffix}")
    with path.open(encoding="utf-8") as f:
        cfg = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    if "extends" in cfg:
        parent = path.parent / cfg.pop("extends")
        cfg = _merge_config(_read_config_tree(parent, seen | {resolved}), cfg)
    return cfg


class BaseConfig:
    @classmethod
    def from_args(cls, args: dict):
        return cls(**args)


@dataclass
class DatasetConfig(BaseConfig):
    name: str
    raw_root: str
    save_root: str
    config_name: str
    downsample: Optional[int]
    segment: float
    segment_step: float
    sequence: float
    sequence_step: float
    graph_type: str
    features: list[str]
    freq_bands: dict[str, tuple[int, int]]
    cog_params: dict[str, dict]
    session_to_load: Optional[list[int]]
    online_transform: Optional[str]

    def __post_init__(self):
        if min(self.segment, self.segment_step, self.sequence, self.sequence_step) <= 0 or self.sequence > self.segment:
            raise ValueError("Window lengths and strides must be positive; sequence must fit the segment")
        if self.graph_type not in {'original', 'general', 'frontal', 'hemisphere', 'TS'}:
            raise ValueError(f"Unsupported graph type: {self.graph_type}")
        if self.online_transform not in {None, "sub_internal", "channel_wise", "zscore"}:
            raise ValueError(f"Unsupported online_transform type: {self.online_transform}")
        # Explicitly verify the key and value, ensure value is a tuple of two ints
        for band, (low, high) in self.freq_bands.items():
            self.freq_bands[band] = (low, high)
        if isinstance(self.session_to_load, int):
            self.session_to_load = [self.session_to_load]
    
    @property
    def save_dir(self) -> Path:
        return Path(self.save_root) / self.config_name


@dataclass
class TrainingConfig(BaseConfig):
    save_root: str
    records_root: str
    tensorboard_root: str
    exp_name: str
    monitor: str
    mode: str
    save_best_n: int
    save_last_k: int
    batch_size: int
    max_epochs: int
    device: torch.device
    patience: int
    learning_rate: float
    dropout: float
    loss_weight: Optional[list[float]]
    label_smoothing: float
    optimizer: str
    wd: float
    wd_head: float
    # Legacy source used the held-out test subject to select epochs.
    selection_protocol: str = "legacy_test"
    validation_subject_offset: int = 1

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.device)
        # Convert to float correctly when 'e' in the string
        self.learning_rate = float(self.learning_rate)
        self.dropout = float(self.dropout)
        self.label_smoothing = float(self.label_smoothing)
        self.wd = float(self.wd)
        self.wd_head = float(self.wd_head)
        if self.selection_protocol not in {"legacy_test", "validation_subject"}:
            raise ValueError("selection_protocol must be legacy_test or validation_subject")
        if self.max_epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("Epochs, batch size and patience must be positive")
        if self.save_best_n < 1:
            raise ValueError("Keep at least one best checkpoint for final evaluation")
        if not 0 <= self.dropout < 1 or not 0 <= self.label_smoothing <= 1:
            raise ValueError("Invalid dropout or label smoothing")
    
    @classmethod
    def from_args(cls, args: dict):
        return cls(**args)
    
    @property
    def save_dir(self) -> Path:
        return Path(self.save_root) / self.exp_name
    
    @property
    def records_dir(self) -> Path:
        return Path(self.records_root) / self.exp_name
    
    @property
    def tb_dir(self) -> Path:
        return Path(self.tensorboard_root) / self.exp_name


@dataclass
class ModelConfig(BaseConfig):
    name: str
    num_classes: int
    class_names: list[str]
    features: list[str]

    def __post_init__(self):
        if isinstance(self.features, str):
            self.features = [self.features]
    
    @classmethod
    def from_args(cls, args):
        model_name = args['name']
        params = args.get('params', {}).get(model_name, {})

        return cls(
            **{k: v for k, v in args.items() if k != 'params'},
            **{k: v for k, v in params.items()},
        )

    @property
    def data_mode(self) -> str:
        if self.name in DE_BASELINE_MODELS:
            return "segment_feature"
        if self.name in RAW_BASELINE_MODELS:
            return "segment_raw"
        return "segment_sequence_feature"

    @property
    def uses_sequence_slicing(self) -> bool:
        return self.data_mode == "segment_sequence_feature"

    @property
    def required_features(self) -> list[str]:
        if self.name in DE_BASELINE_MODELS:
            return ["de"]
        if self.name in RAW_BASELINE_MODELS:
            return ["raw"]
        return list(self.features)

    def resolve_graph_type(self, dataset_graph_type: str) -> str:
        if self.name in DE_BASELINE_MODELS:
            return "original"
        if self.name in RAW_BASELINE_MODELS:
            return "TS"
        return dataset_graph_type


@dataclass
class MyModelConfig(ModelConfig):
    attn_layers: int
    attn_heads: int
    conv_K: list[int]
    share_encoders: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.name != 'MyModel':
            raise ValueError(f"Model name mismatch for MyModelConfig: {self.name}")


@dataclass
class MyModelCogFusionConfig(ModelConfig):
    attn_heads: int
    conv_K: list[int]
    cog_dim: int
    fusion: str
    share_encoders: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.name != 'MyModelCogFusion':
            raise ValueError(f"Model name mismatch for MyModelCogFusionConfig: {self.name}")
        if self.fusion not in {'simple', 'gate', 'cross', 'self', 'self+cross'}:
            raise ValueError(f"Unsupported fusion type: {self.fusion}")


@dataclass
class ECFNetConfig(ModelConfig):
    K: int = 2
    attn_heads: int = 4
    head_dim: int = 16
    fusion: str = "self+cross"
    share_encoders: bool = False
    use_descriptors: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.fusion not in {"addition", "cross", "self+addition", "self+cross"}:
            raise ValueError(f"Unsupported ECFNet fusion: {self.fusion}")
        if self.K < 1 or self.attn_heads != 4 or self.head_dim != 16:
            raise ValueError("ECFNet uses positive K and 4 attention heads of width 16")

    @property
    def required_features(self):
        return ["de", "wavelet"] if self.use_descriptors else ["de"]


@dataclass
class EmTConfig(ModelConfig):
    layers_graph: list[int]
    layers_transformer: int
    hidden_graph: int
    cheby_K: int
    num_head: int
    dim_head: int
    sta_alpha: float
    graph2token: str
    encoder_type: str

    def __post_init__(self):
        super().__post_init__()
        if self.name != 'EmT':
            raise ValueError(f"Model name mismatch for EmTConfig: {self.name}")
        if self.graph2token not in {'Linear', 'AvgPool', 'MaxPool', 'Flatten'}:
            raise ValueError(f"Unsupported graph2token type: {self.graph2token}")
        if self.encoder_type not in {'Cheby', 'GCN'}:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")


@dataclass
class DGCNNConfig(ModelConfig):
    def __post_init__(self):
        super().__post_init__()
        if self.name != 'DGCNN':
            raise ValueError(f"Model name mismatch for DGCNNConfig: {self.name}")
        self.features = self.required_features


@dataclass
class GCBNetBLSConfig(ModelConfig):
    def __post_init__(self):
        super().__post_init__()
        if self.name != 'GCBNet_BLS':
            raise ValueError(f"Model name mismatch for GCBNetBLSConfig: {self.name}")
        self.features = self.required_features


@dataclass
class RGNNConfig(ModelConfig):
    def __post_init__(self):
        super().__post_init__()
        if self.name != 'RGNN':
            raise ValueError(f"Model name mismatch for RGNNConfig: {self.name}")
        self.features = self.required_features


@dataclass
class TSCeptionConfig(ModelConfig):
    def __post_init__(self):
        super().__post_init__()
        if self.name != 'TSCeption':
            raise ValueError(f"Model name mismatch for TSCeptionConfig: {self.name}")
        self.features = self.required_features


@dataclass
class ReproduceConfig(BaseConfig):
    random_seed: int
    deterministic: bool


@dataclass
class LoggingConfig(BaseConfig):
    save_root: str
    save_dir: str

    @property
    def log_dir(self) -> Path:
        return Path(self.save_root) / self.save_dir
    

@dataclass
class RunConfig:
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    reproduce: ReproduceConfig
    logging: LoggingConfig
