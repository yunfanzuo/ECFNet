import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Union

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from utils import console, logger
from utils.channel import get_ch_index, get_ch_name
from utils.cognitive import CognitiveMetrics
from utils.feature_process import bandpower_welch, diff_entropy, moving_average
from utils.signal_process import butter_bandpass
from utils.tool import TimeColumn
from utils.transform import BandDecompose, ChannelReorder, Downsampling, TrailSpliter
from utils.wavelet import WaveletDescriptors


class HDF5Writer:
    """
    HDF5 file writer for building eeg dataset

    Args:
        file_path (str): Path to the HDF5 file.
        mode (str): File mode, e.g., 'a' for append, 'w' for write.

    Notes:
        The samples dict should have the following structure:
        
        ```python
        {
            "path": str,  # HDF5 group path
            "features": Dict[str, np.ndarray],  # Features to store, keyed by feature name, values are numpy arrays
            "label": np.ndarray,  # Label to store
        }
        ```
    """

    def __init__(self, file_path: str, *, mode: str = "a"):
        self.file_path = file_path
        self.mode = mode
        self._file: Optional[h5py.File] = None

    def __enter__(self) -> "HDF5Writer":
        self._file = h5py.File(self.file_path, self.mode)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file is not None:
            self.write_attrs({
                "updated_at": datetime.now().isoformat(),
            })
            self._file.close()
            self._file = None

    def write_attrs(
        self,
        attrs: Dict[str, Any],
    ) -> None:
        self._ensure_open()
        for key, value in attrs.items():
            if key == "updated_at":
                if self._file.attrs.get("created_at") is None:
                    self._file.attrs["created_at"] = value
                self._file.attrs["updated_at"] = value
            else:
                self._file.attrs[key] = value

    def write_samples(
        self,
        samples: Dict[str, Any],
        *, overwrite: bool = False, compression_on_features: Optional[str] = "gzip"
    ) -> None:
        self._ensure_open()
        path: str = samples["path"]
        group = self._get_group(path)

        self._write_features(group, samples["features"], overwrite=overwrite, compression=compression_on_features)
        self._write_label(group, samples["label"], overwrite=overwrite)

    def _ensure_open(self) -> None:
        if self._file is None:
            raise RuntimeError("HDF5Writer is not opened (use 'with')")

    def _get_group(self, path: str) -> h5py.Group:
        if not path:
            raise ValueError("path must not be empty")
        h5_path = path.strip("/")
        return self._file.require_group(h5_path)

    def _write_features(
        self,
        group: h5py.Group, features: Dict[str, np.ndarray],
        *, overwrite: bool = False, compression: Optional[str] = "gzip", **kwargs
    ) -> None:
        if not isinstance(features, dict):
            raise TypeError("features must be a dict[str, np.ndarray]")

        feat_group = group.require_group("features")
        for name, array in features.items():
            self._write_array(feat_group, name, array, overwrite=overwrite, compression=compression, **kwargs)

    def _write_label(
        self,
        group: h5py.Group, label: np.ndarray,
        *, overwrite: bool = False, **kwargs
    ) -> None:
        self._write_array(group, "label", label, overwrite=overwrite, **kwargs)

    @staticmethod
    def _write_array(
        group: h5py.Group, name: str, array: np.ndarray,
        *, overwrite: bool, **kwargs
    ) -> None:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Feature '{name}' must be numpy array")

        if name in group:
            if overwrite:
                del group[name]
            else:
                raise RuntimeError(f"Key '{name}' already exists in {group}")

        group.create_dataset(name, data=array, **kwargs)


class BasePreprocessor(ABC):
    def __init__(self, raw_root, data_dir, pipeline):
        self.raw_root = Path(raw_root)
        self.data_dir = Path(data_dir)
        self.pipeline = pipeline

    @abstractmethod
    def index_keys(self) -> list[str]:
        """返回索引表中包含的键列表"""
        ...

    @abstractmethod
    def iter_raw_trials(self) -> Iterable:
        """遍历所有原始 trial 数据"""
        ...

    @property
    @abstractmethod
    def num_subs(self) -> int:
        """实际被试数量"""
        ...

    @property
    @abstractmethod
    def sub_ids(self) -> list[int]:
        """实际被试ID列表"""
        ...

    def run(self):
        """
        扫描所有原始数据 → 处理数据 → 写特征文件 / 索引表
        """
        # 保存的文件路径
        feature_file = self.data_dir / "data.h5"
        feature_file.parent.mkdir(parents=True, exist_ok=True)

        # 收集每个样本的索引信息
        index_records = {
            "sample_id": [],
            **{k: [] for k in self.index_keys()},
            "label": [],
        }
        num_samples = 0

        logger.info(f"Processing raw data from {self.raw_root} ...")

        # Rebuild instead of retaining stale groups from an older configuration.
        with HDF5Writer(str(feature_file), mode="w") as writer:
            # 处理每个trial
            for raw_trial in self.iter_raw_trials():
                # samples example:
                # the samples here is a dict containing a batch of multiple samples processed from one trail
                # samples = {
                #     "count": N,   # 必须有的Key, 样本数量
                #     "index": {    # 必须有的key, 用于索引
                #         "subject_id": [1, 1, 1, ..., 1],      # 每个样本的subject_id, 应全部相同
                #         "session_id": [1, 1, 1, ..., 1],      # 每个样本的session_id, 应全部相同
                #         "trial_id": [1, 1, 1, ..., 1],        # 每个样本的trial_id, 应全部相同
                #         "t_index": [0, 1, 2, ..., N-1],       # 每个样本在该trail中的时间索引
                #         "t_start": [0.0, 0.5, 1.0, ..., ],    # 每个样本的起始时间(s)
                #         "t_end": [0.5, 1.0, 1.5, ..., ]       # 每个样本的结束时间(s)
                #     },
                #     "path": "subject_01/session_01/trail_01",  # 必须有的key, 保存在h5py文件的group路径
                #     "features": {  # 必须有的key
                #         "de": "numpy array batches",  # 整个trail所有样本的features存储为一个array，加速读取速度, shape=(N, ...)
                #     },
                #     "label": "numpy array batches"  # 必须有的key, shape=(N,)
                # }

                # 处理trial数据, 获取批量样本
                samples = self.pipeline.process_trial(raw_trial)
                if samples is None:
                    # 该trial无法处理, 跳过
                    continue
                # 将样本特征和标签写入HDF5文件
                writer.write_samples(samples, overwrite=True)

                # 收集样本索引信息
                indices: dict[str, list[int]] = samples["index"]
                for key, value in indices.items():
                    index_records[key].extend(list(value))
                # 记录样本标签至索引表
                index_records["label"].extend(
                    samples["label"].tolist()
                )

                # 样本计数
                count = samples["count"]
                index_records["sample_id"].extend(
                    list(range(num_samples, num_samples + count))
                )
                num_samples += count

        logger.info(f"Processed {num_samples} samples from raw data, saved to {feature_file}")

        # 保存索引表
        df = pd.DataFrame(index_records)
        if df.empty:
            raise ValueError("No complete EEG segments were produced")
        index_path = self.data_dir / "index.parquet"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(index_path)

        logger.info(f"Saved index table to {index_path}")

        # 保存预处理Pipeline配置
        config_path = self.data_dir / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = self.pipeline.meta_attrs
        config["raw_root"] = str(self.raw_root)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved pipeline config to {config_path}")


class MyTrialPipeline:
    def __init__(self, args):
        dataset_args = args.dataset
        model_args = args.model

        self.args = dataset_args
        self.model_args = model_args
        self.downsample = dataset_args.downsample       # 下采样率
        self.segment_length = dataset_args.segment      # 分段长度 (s)
        self.segment_step = dataset_args.segment_step   # 分段步长 (s)
        self.sequence_length = dataset_args.sequence    # 子序列长度 (s)
        self.sequence_step = dataset_args.sequence_step # 子序列步长 (s)
        self.dataset = dataset_args.name                # 数据集名称
        self.requested_graph_type = dataset_args.graph_type
        self.graph_type = model_args.resolve_graph_type(dataset_args.graph_type)
        self.features = list(model_args.required_features)
        self.uses_sequence_slicing = model_args.uses_sequence_slicing
        self.ch_idx = get_ch_index(dataset=self.dataset, graph_type=self.graph_type)  # 通道重排索引
        self.ch_names = get_ch_name(dataset=self.dataset, graph_type=self.graph_type)  # 通道名列表
        self.freq_bands = list(dataset_args.freq_bands.values())    # 频段划分

        if self.downsample:
            self.down_sampling = Downsampling(self.downsample)
        else:
            self.down_sampling = lambda x: x
        self.channel_reorder = ChannelReorder(self.ch_idx)
        self.band_decompose = BandDecompose(self.freq_bands)
        self.trial_spliter = TrailSpliter(
            self.segment_length, self.segment_step,
            self.sequence_length if self.uses_sequence_slicing else None,
            self.sequence_step if self.uses_sequence_slicing else None,
        )

    @staticmethod
    def index_keys() -> list[str]:
        return ["t_index", "t_start", "t_end"]

    def process_trial(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个 trial 数据"""
        trial = self.down_sampling(trial)       # eeg: (C, T)
        trial = self.channel_reorder(trial)     # eeg: (C, T)

        try:
            # these features need band decomposed first
            need_band_decompose = any(f for f in self.features if f in ["de", "de_mov"])
            if need_band_decompose:
                decomposed = self.band_decompose(trial)     # eeg: (C, B, T)
                decomposed = self.trial_spliter(decomposed) # eeg: (N, Seq, C, B, Ts)
            else:
                decomposed = None
            trial = self.trial_spliter(trial)               # eeg: (N, Seq, C, Ts)
        except ValueError as exc:
            # 样本长度不足以分段, 该试次跳过
            if trial["eeg"].shape[-1] < int(self.segment_length * trial["fs"]):
                logger.warning("Skipping short trial {}: {}", trial.get("path"), exc)
                return None
            raise

        features = {}
        for feat in self.features:
            if feat in ("de", "de_mov"):
                features[feat] = diff_entropy(decomposed["eeg"], normal=True)  # (N, Seq, C, B)
                if feat == "de_mov":
                    features[feat] = moving_average(features[feat], k=3, axis=1)  # (N, Seq, C, B)
            elif feat == "raw":
                eeg = trial["eeg"]  # (N, C, Ts)
                if eeg.ndim == 3:
                    eeg = np.expand_dims(eeg, axis=1)  # (N, 1, C, Ts)
                features[feat] = eeg
            elif feat in ("psd", "r_psd"):
                features[feat] = bandpower_welch(   # (N, Seq, C, B)
                    trial["eeg"], trial["fs"], self.freq_bands, relative=(feat == "r_psd")
                )
            elif feat == "wavelet":
                # Preserve the original per-window 1-64 Hz filtering scope.
                eeg = butter_bandpass(trial["eeg"], 1, 64, trial["fs"], order=4)
                # Vectorize the 37 windows of each segment; bound DWT memory.
                features[feat] = np.stack([
                    WaveletDescriptors(segment, self.ch_names).compute() for segment in eeg
                ])
            elif feat.startswith("cog_"):    # Legacy aliases; use wavelet for ECFNet.
                cog_params = self.args.cog_params
                cog_names = feat.split("_")[1].split("+")  # e.g. cog_b+a+w -> ['b', 'a', 'w']
                eeg = trial["eeg"]  # (N, Seq, C, Ts)
                eeg = butter_bandpass(eeg, 1, 64, trial["fs"])
                n_seg, n_seq = eeg.shape[:2]
                cog_fea = np.zeros(shape=(n_seg, n_seq, len(cog_names)), dtype=eeg.dtype)  # (N, Seq, n_fea)
                for i in range(n_seg):
                    for j in range(n_seq):
                        seq = eeg[i, j]  # (C, Ts)
                        ci = CognitiveMetrics(seq, self.ch_names)
                        for k, cog in enumerate(cog_names):
                            if cog == 'b':
                                cog_fea[i, j, k] = ci.band_ratio(
                                    cog_params['band_ratio']['band1'],
                                    cog_params['band_ratio']['band2'],
                                    cog_params['band_ratio']['ch']
                                )
                            elif cog == 'a':
                                cog_fea[i, j, k] = ci.asymmetry(
                                    cog_params['asymmetry']['band'],
                                    cog_params['asymmetry']['l_ch'],
                                    cog_params['asymmetry']['r_ch']
                                )
                            elif cog == 'w':
                                cog_fea[i, j, k] = ci.weng(
                                    cog_params['weng']['ch']
                                )
                            else:
                                raise ValueError(f"{feat} 中不支持的认知特征类型: {cog}")
                features[feat] = cog_fea  # (N, Seq, n_fea)
            else:
                raise ValueError(f"Unsupported feature: {feat}")

            if not np.isfinite(features[feat]).all():
                raise ValueError(f"Non-finite {feat} features in {trial.get('path')}")
            if self.model_args.name == "ECFNet":
                features[feat] = features[feat].astype(np.float32)

        trial["features"] = features
        return trial

    @property
    def meta_attrs(self) -> Dict[str, Any]:
        return {
            "preprocessing_version": 2,
            "dataset": self.dataset,
            "downsample": self.downsample,
            "segment": self.segment_length,
            "segment_step": self.segment_step,
            "sequence": self.sequence_length if self.uses_sequence_slicing else None,
            "sequence_step": self.sequence_step if self.uses_sequence_slicing else None,
            "requested_graph_type": self.requested_graph_type,
            "graph_type": self.graph_type,
            "features": list(self.features),
            "freq_bands": {k: list(v) for k, v in self.args.freq_bands.items()},
            "data_mode": self.model_args.data_mode,
            "model_name": self.model_args.name,
            "cog_params": self.args.cog_params if any(f.startswith("cog_") for f in self.features) else {},
            "wavelet": {
                "name": "db4", "level": 4, "mode": "symmetric",
                "filter_hz": [1, 64], "filter_order": 4,
                "filter_scope": "window", "ratio_epsilon": 1e-10,
                "descriptor_order": ["frontal_cross_scale_ratio", "lateral_asymmetry", "frontal_detail_energy"],
            } if "wavelet" in self.features else None,
            "de_filter_order": 4,
            "de_filter_scope": "trial",
            "de_variance_ddof": 0,
            "session_to_load": self.args.session_to_load,
        }


FeaturesData = Dict[str, Union[list[np.ndarray], np.ndarray, torch.Tensor]]
ArrList = list[np.ndarray]


class MyDataPipe(ABC):
    def __init__(self, args, pipeline, preprocessor):
        self.args = args
        self.features = args.model.features
        self.data_dir = Path(args.dataset.save_dir)
        self.pipeline = pipeline
        self.preprocessor = preprocessor

    def build(self):
        logger.info("Building dataset at {} ...", self.data_dir)
        # An interrupted rebuild must not leave an old valid fingerprint.
        self.config_path.unlink(missing_ok=True)
        self.preprocessor.run()
        logger.info("Dataset built successfully.")

    @property
    def data_path(self) -> Path:
        return self.data_dir / "data.h5"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.parquet"

    @property
    def config_path(self) -> Path:
        return self.data_dir / "config.json"

    @property
    def prepared_config(self) -> Dict[str, Any]:
        return {
            **self.pipeline.meta_attrs,
            "raw_root": str(self.preprocessor.raw_root),
        }

    def is_prepared(self) -> bool:
        if not (self.data_path.exists() and self.index_path.exists() and self.config_path.exists()):
            return False
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                saved_config = json.load(f)
        except Exception:
            return False
        return saved_config == self.prepared_config

    @property
    @abstractmethod
    def required_index_keys(self) -> list[str]:
        """索引表中必须包含的键列表"""
        ...

    @property
    @abstractmethod
    def grouped_keys(self) -> list[str]:
        """用于分组加载数据的键列表"""
        ...

    @property
    def unique_key_in_group(self) -> str:
        """分组中唯一标识样本的键"""
        return "t_index"

    @abstractmethod
    def group_index_2_path(self, group_index) -> str:
        """根据分组索引返回对应的 HDF5 路径"""
        ...

    @property
    def num_subs(self) -> int:
        """实际被试数量"""
        return self.preprocessor.num_subs
    
    @property
    def sub_ids(self) -> list[int]:
        """实际被试ID列表"""
        return self.preprocessor.sub_ids

    def _load(
        self, grp: h5py.Group, indices: pd.DataFrame, *, on_load: Optional[Callable[..., Any]] = None
    ) -> tuple[FeaturesData, np.ndarray]:
        """
        根据索引记录加载对应样本的数据并返回 (data, label)
        Args:
            grp: h5py.Group, 已打开的 HDF5 文件对象
            indices: 包含若干样本行的 Pandas DataFrame, 必须包含列 ``self.required_index_keys``
            on_load: 可选的回调函数, 在加载每个trial后调用, 接收 ``tuple(self.grouped_keys)`` 参数
        """
        # 验证输入列
        required_cols = set(self.required_index_keys)
        if not required_cols.issubset(set(indices.columns)):
            raise ValueError(f"indices must contain columns: {required_cols}")

        data: FeaturesData = {f: [] for f in self.features}
        label: Union[list[np.ndarray], np.ndarray] = []

        # 按 grouped_keys 分组以减少文件随机读取
        grouped = indices.groupby(self.grouped_keys, sort=False)
        for group_index, group_df in grouped:
            path = self.group_index_2_path(group_index)
            if path not in grp:
                raise KeyError(f"Path '{path}' not found in HDF5 file")

            feat_grp = grp[path]["features"]
            indices = group_df[self.unique_key_in_group].to_numpy(dtype=int)

            for feat in self.features:
                if feat not in feat_grp:
                    raise KeyError(f"Feature '{feat}' not found under {path}/features")
                data[feat].append(feat_grp[feat][indices])

            label.append(group_df["label"].to_numpy())

            if on_load is not None:
                on_load(group_index)

        data = {
            k: (np.concatenate(v, axis=0) if len(v) > 0 else np.zeros((0,)))
            for k, v in data.items()
        }
        label = np.concatenate(label, axis=0) if len(label) > 0 else np.zeros((0,))
        return data, label

    def load(self, index_df: pd.DataFrame) -> tuple[FeaturesData, np.ndarray]:
        with h5py.File(self.data_path, "r") as f:
            data, label = self._load(f, index_df)
        return data, label

    def load_one_sub(self,
        sub_id: int,
        pre_filter: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df,
        after_load: Callable[[FeaturesData, np.ndarray], tuple[FeaturesData, np.ndarray]] = lambda data, label: (data, label),
    ) -> tuple[FeaturesData, np.ndarray]:
        """
        加载单个被试的数据
        Args:
            sub_id: 被试ID
            pre_filter: 加载前对索引表的筛选函数
            after_load: 加载后对数据的后处理函数
        """
        # 读取索引表
        index_df = pd.read_parquet(self.index_path)
        # 筛选该被试索引
        sub_indices = index_df[index_df["subject_id"] == sub_id]
        # 进一步筛选
        sub_indices = pre_filter(sub_indices)

        # 计算trials数量用于进度条
        total_trials = sub_indices.groupby(self.grouped_keys).ngroups

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.fields[path]}"),
            BarColumn(), MofNCompleteColumn(), TimeColumn(),
            console=console,
        ) as progress, h5py.File(self.data_path, "r") as f:
            task = progress.add_task("[green]Loading", total=total_trials, path="")
            load_cbk = lambda group_index: progress.update(
                task, advance=1, path=self.group_index_2_path(group_index)
            )
            # 使用_load方法加载数据
            data, label = self._load(f, sub_indices, on_load=load_cbk)

        data, label = after_load(data, label)
        return data, label

    def iter_subs(
        self,
        sub_ids: list[int] = None,
        exclude: list[int] = None,
        pre_filter: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df,
        after_load: Callable[[FeaturesData, np.ndarray], tuple[FeaturesData, np.ndarray]] = lambda data, label: (data, label),
    ) -> Iterable[tuple[int, tuple[FeaturesData, np.ndarray]]]:
        if sub_ids is None:
            sub_ids = self.sub_ids
        if exclude is not None:
            sub_ids = tuple(sid for sid in sub_ids if sid not in exclude)

        # 读取索引表并筛选
        index_df = pd.read_parquet(self.index_path)
        all_indices = index_df[index_df["subject_id"].isin(sub_ids)]

        # 进一步筛选
        all_indices = pre_filter(all_indices)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.fields[path]}"),
            BarColumn(), MofNCompleteColumn(), TimeColumn(),
            console=console,
        ) as progress, h5py.File(self.data_path, "r") as f:
            for sub_id in sub_ids:
                # 筛选当前被试的数据
                sub_indices = all_indices[all_indices["subject_id"] == sub_id]

                # 创建进度条任务
                sub_trials = sub_indices.groupby(self.grouped_keys).ngroups
                task = progress.add_task("[green]Loading", total=sub_trials, path="")
                load_cbk = lambda group_index: progress.update(
                    task, advance=1, path=self.group_index_2_path(group_index)
                )

                # 使用_load方法加载数据
                data, label = self._load(f, sub_indices, on_load=load_cbk)
                data, label = after_load(data, label)

                progress.remove_task(task)
                yield sub_id, (data, label)

    def load_subs(
        self,
        sub_ids: list[int] = None,
        exclude: list[int] = None,
        concat: bool = True,
        pre_filter: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df,
        after_load: Callable[[FeaturesData, np.ndarray], tuple[FeaturesData, np.ndarray]] = lambda data, label: (data, label),
    ) -> tuple[FeaturesData, Union[np.ndarray, ArrList]]:
        all_data: FeaturesData = {f: [] for f in self.features}
        all_label: list[np.ndarray] | np.ndarray = []

        for _, (data, label) in self.iter_subs(sub_ids, exclude, pre_filter, after_load):
            for k, v in data.items():
                all_data[k].append(v)
            all_label.append(label)

        if concat:
            all_data = {
                k: np.concatenate(v, axis=0) if len(v) > 0 else np.zeros((0,))
                for k, v in all_data.items()
            }
            all_label = np.concatenate(all_label, axis=0) if len(all_label) > 0 else np.zeros((0,))

        return all_data, all_label

    def iter_loso(
        self,
        sub_ids: list[int] = None,
        exclude: list[int] = None,
        pre_filter: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df,
        after_load: Callable[[FeaturesData, np.ndarray], tuple[FeaturesData, np.ndarray]] = lambda data, label: (data, label),
    ) -> Iterable[tuple[int, tuple[FeaturesData, np.ndarray, FeaturesData, np.ndarray]]]:
        if sub_ids is None:
            sub_ids = self.sub_ids
        if exclude is not None:
            sub_ids = tuple(sid for sid in sub_ids if sid not in exclude)

        # 读取索引表并筛选
        index_df = pd.read_parquet(self.index_path)
        all_indices = index_df[index_df["subject_id"].isin(sub_ids)]

        # 进一步筛选
        all_indices = pre_filter(all_indices)

        # 计算trials数量用于进度条  
        total_trials = all_indices.groupby(self.grouped_keys).ngroups

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.fields[path]}"),
            BarColumn(), MofNCompleteColumn(), TimeColumn(),
            console=console,
        ) as progress, h5py.File(self.data_path, "r") as f:
            for sub_id in sub_ids:
                # 创建进度条任务
                task = progress.add_task("[green]Loading", total=total_trials, path="")
                load_cbk = lambda group_index: progress.update(
                    task, advance=1, path=self.group_index_2_path(group_index)
                )

                # 使用其它被试数据作为训练集
                train_data: FeaturesData = {f: [] for f in self.features}
                train_label: list[np.ndarray] | np.ndarray = []

                for other_sub in sub_ids:
                    if other_sub == sub_id:
                        continue
                    sub_indices = all_indices[all_indices["subject_id"] == other_sub]
                    sub_data, sub_label = self._load(f, sub_indices, on_load=load_cbk)
                    sub_data, sub_label = after_load(sub_data, sub_label)
                    for k, v in sub_data.items():
                        train_data[k].append(v)
                    train_label.append(sub_label)

                # 合并训练集数据
                train_data = {
                    k: np.concatenate(v, axis=0) if len(v) > 0 else np.zeros((0,))
                    for k, v in train_data.items()
                }
                train_label = np.concatenate(train_label, axis=0) if len(train_label) > 0 else np.zeros((0,))

                # 使用当前被试数据作为测试集
                test_indices = all_indices[all_indices["subject_id"] == sub_id]
                test_data, test_label = self._load(f, test_indices, on_load=load_cbk)
                test_data, test_label = after_load(test_data, test_label)

                progress.remove_task(task)
                yield sub_id, (train_data, train_label, test_data, test_label)


class SeedPreprocessor(BasePreprocessor):
    NUM_SUBS = 15
    NUM_SESSIONS = 3
    NUM_TRAILS = 15
    RAW_FS = 200

    def __init__(self, raw_root, save_dir, pipeline):
        super().__init__(raw_root, save_dir, pipeline)
        self._trials_label = None

    @property
    def num_subs(self):
        return self.NUM_SUBS
    
    @property
    def sub_ids(self):
        return list(range(1, self.NUM_SUBS + 1))

    @property
    def num_sessions(self):
        return self.NUM_SESSIONS

    @property
    def num_trials(self):
        return self.NUM_TRAILS

    @property
    def trials_label(self):
        if self._trials_label is None:
            self._trials_label = self.load_trials_label()
        return self._trials_label

    def index_keys(self) -> list[str]:
        return ["subject_id", "session_id", "trial_id"] + self.pipeline.index_keys()

    def iter_raw_trials(self) -> Iterable:
        """
        遍历 SEED 数据集的所有原始 trial 数据
        """
        all_files = tuple(self.all_files())
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.fields[path]}"),
            BarColumn(), MofNCompleteColumn(), TimeColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing", total=len(all_files)*self.NUM_TRAILS, path="")
            load_cbk = lambda data: progress.update(task, advance=1, path=data["path"])

            for f in self.all_files():
                logger.info("Loading data file: {}", f["file_path"])
                yield from self.load_data_file(f, on_load=load_cbk)

    def load_data_file(self, file_info: dict, *, on_load: Optional[Callable[[dict], Any]] = None):
        file_path: str = file_info["file_path"]
        subject_id: int = file_info["subject_id"]
        session_id: int = file_info["session_id"]

        mat = sio.loadmat(file_path)
        # MATLAB dictionary insertion order need not be numeric trial order.
        trials_key = [k for k in mat if re.search(r"eeg\d+$", k)]
        trials_key.sort(key=lambda k: int(re.search(r"eeg(\d+)$", k).group(1)))
        trial_ids = [int(re.search(r"eeg(\d+)$", k).group(1)) for k in trials_key]
        if trial_ids != list(range(1, self.NUM_TRAILS + 1)) or len(self.trials_label) != self.NUM_TRAILS:
            raise ValueError(f"Expected EEG trials 1-{self.NUM_TRAILS} in {file_path}")

        for trial_id, key in enumerate(trials_key, 1):
            trial_data: np.ndarray = mat[key]  # shape=(C, T)
            if trial_data.ndim != 2 or trial_data.shape[0] != 62:
                raise ValueError(f"Expected 62-channel trial in {file_path}:{key}")
            label: int = self.trials_label[trial_id - 1]
            data = {
                "index": {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "trial_id": trial_id,
                },
                "path": f"subject_{subject_id:02d}/session_{session_id:02d}/trial_{trial_id:02d}",
                "eeg": trial_data,
                "label": label,
                "fs": self.RAW_FS,
            }
            if on_load is not None:
                on_load(data)
            yield data

    def load_trials_label(self) -> list[int]:
        label_path = str(self.raw_root / "label.mat")
        label = sio.loadmat(label_path)['label']
        label = np.squeeze(label) + 1  # 标签从0开始
        if label.shape != (15,) or set(label.tolist()) != {0, 1, 2}:
            raise ValueError("SEED label.mat must contain 15 labels in {-1, 0, 1}")
        return label.tolist()

    def all_files(self):
        for sub_id in self.sub_ids:
            yield from self.sub_files(sub_id)

    def sub_files(self, sub_id):
        files = self.raw_root.glob(f"{sub_id}_*.mat")
        files = sorted(files)
        if self.NUM_SESSIONS != len(files):
            raise ValueError(f"Subject {sub_id}: expected 3 .mat sessions under {self.raw_root}, found {len(files)}")
        for sess_id, f in enumerate(files, 1):
            yield {
                "file_path": str(f.absolute()),
                "subject_id": sub_id,
                "session_id": sess_id,
            }


class SeedDataPipe(MyDataPipe):
    def __init__(self, args):
        super().__init__(
            args,
            MyTrialPipeline(args),
            SeedPreprocessor(args.dataset.raw_root, args.dataset.save_dir, MyTrialPipeline(args))
        )
        self.sess_to_load = args.dataset.session_to_load
        self.num_classes = args.model.num_classes

    @property
    def required_index_keys(self) -> list[str]:
        return ["subject_id", "session_id", "trial_id", "t_index", "label"]

    @property
    def grouped_keys(self) -> list[str]:
        return ["subject_id", "session_id", "trial_id"]

    def group_index_2_path(self, group_index: tuple) -> str:
        sub_id, sess_id, trial_id = group_index
        return f"subject_{int(sub_id):02d}/session_{int(sess_id):02d}/trial_{int(trial_id):02d}"

    def _pre_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        # 筛选指定session的数据
        if self.sess_to_load is not None:
            df = df[df["session_id"].isin(self.sess_to_load)]
        # 如果是二分类，从index中筛选掉中性类样本
        if self.num_classes == 2:
            df = df[df["label"] != 1]
        return df

    def _after_load(self, data: FeaturesData, label: np.ndarray) -> tuple[FeaturesData, np.ndarray]:
        # 如果是二分类，将正类标签改为1，负类标签改为0
        if self.num_classes == 2:
            label = np.where(label == 2, 1, 0)
        return data, label

    def load_one_sub(
        self,
        sub_id: int,
        pre_filter = lambda df: df,
        after_load = lambda data, label: (data, label)
    ) -> tuple[FeaturesData, np.ndarray]:
        def _combined_filter(_df):
            df = self._pre_filter(_df)
            return pre_filter(df)
        def _combined_load(_data, _label):
            data, label = self._after_load(_data, _label)
            return after_load(data, label)

        return super().load_one_sub(
            sub_id,
            pre_filter=_combined_filter,
            after_load=_combined_load
        )

    def iter_subs(
        self,
        sub_ids = None,
        exclude = None,
        pre_filter = lambda df: df,
        after_load = lambda data, label: (data, label)
    ):
        def _combined_filter(_df):
            df = self._pre_filter(_df)
            return pre_filter(df)
        def _combined_load(_data, _label):
            data, label = self._after_load(_data, _label)
            return after_load(data, label)

        yield from super().iter_subs(
            sub_ids,
            exclude,
            pre_filter=_combined_filter,
            after_load=_combined_load,
        )

    def load_subs(
        self,
        sub_ids = None,
        exclude = None,
        concat = True,
        pre_filter = lambda df: df,
        after_load = lambda data, label: (data, label)
    ):
        def _combined_filter(_df):
            df = self._pre_filter(_df)
            return pre_filter(df)
        def _combined_load(_data, _label):
            data, label = self._after_load(_data, _label)
            return after_load(data, label)

        return super().load_subs(
            sub_ids,
            exclude,
            concat=concat,
            pre_filter=_combined_filter,
            after_load=_combined_load,
        )

    def iter_loso(
        self,
        sub_ids = None,
        exclude = None,
        pre_filter = lambda df: df,
        after_load = lambda data, label: (data, label)
    ):
        def _combined_filter(_df):
            df = self._pre_filter(_df)
            return pre_filter(df)
        def _combined_load(_data, _label):
            data, label = self._after_load(_data, _label)
            return after_load(data, label)

        yield from super().iter_loso(
            sub_ids=sub_ids,
            exclude=exclude,
            pre_filter=_combined_filter,
            after_load=_combined_load,
        )


class MLEDPreprocessor(BasePreprocessor):
    NUM_SUBS = 54
    NUM_TRIALS = 12
    RAW_FS = 128
    EXCLUDED_SUBS = [6, 7, 8]

    def __init__(self, raw_root, data_dir, pipeline):
        super().__init__(raw_root, data_dir, pipeline)

    @property
    def num_subs(self):
        return self.NUM_SUBS - len(self.EXCLUDED_SUBS)

    @property
    def sub_ids(self):
        return [sid for sid in range(1, self.NUM_SUBS + 1) if sid not in self.EXCLUDED_SUBS]

    @property
    def num_trials(self):
        return self.NUM_TRIALS

    @property
    def trials_label(self):
        # 0: 中性, 1: 困惑, 2: 专注, 3: 无聊
        return [1, 2, 3, 1, 2, 0, 1, 2, 3, 1, 2, 0]

    def index_keys(self) -> list[str]:
        return ["subject_id", "trial_id"] + self.pipeline.index_keys()

    def iter_raw_trials(self) -> Iterable:
        """
        遍历 MLED 数据集的所有原始 trial 数据
        """
        all_files = tuple(self.all_files())
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.fields[path]}"),
            BarColumn(), MofNCompleteColumn(), TimeColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing", total=len(all_files), path="")
            load_cbk = lambda data: progress.update(task, advance=1, path=data["path"])

            for f in self.all_files():
                logger.info("Loading data file: {path}", path=f["file_path"])
                yield from self.load_data_file(f, on_load=load_cbk)

    def load_data_file(self, file_info: dict, *, on_load: Optional[Callable[[dict], Any]] = None):
        file_path: str = file_info["file_path"]
        subject_id: int = file_info["subject_id"]
        trial_id: int = file_info["trial_id"]

        mat = sio.loadmat(file_path)
        trial_data: np.ndarray = mat['data'] # shape=(C, T)
        label: int = self.trials_label[trial_id - 1]
        data = {
            "index": {
                "subject_id": subject_id,
                "trial_id": trial_id,
            },
            "path": f"subject_{subject_id:02d}/trial_{trial_id:02d}",
            "eeg": trial_data,
            "label": label,
            "fs": self.RAW_FS,
        }
        if on_load is not None:
            on_load(data)
        yield data

    def all_files(self):
        for sub_id in self.sub_ids:
            yield from self.sub_files(sub_id)

    def sub_files(self, sub_id):
        files = self.raw_root.glob(f"P{sub_id:02d}_*.mat")
        files = sorted(files)
        assert self.NUM_TRIALS == len(files), "被试试次数量与文件数量不匹配"
        for f in files:
            yield {
                "file_path": str(f.absolute()),
                "subject_id": sub_id,
                "trial_id": int(f.stem.split("_")[1])
            }


class MLEDDataPipe(MyDataPipe):
    def __init__(self, args):
        super().__init__(
            args,
            MyTrialPipeline(args),
            MLEDPreprocessor(args.dataset.raw_root, args.dataset.save_dir, MyTrialPipeline(args))
        )

    @property
    def required_index_keys(self) -> list[str]:
        return ["subject_id", "trial_id", "t_index", "label"]

    @property
    def grouped_keys(self) -> list[str]:
        return ["subject_id", "trial_id"]

    def group_index_2_path(self, group_index: tuple) -> str:
        sub_id, trial_id = group_index
        return f"subject_{int(sub_id):02d}/trial_{int(trial_id):02d}"


def create_datapipe(args) -> MyDataPipe:
    dataset_name = args.dataset.name.upper()
    if dataset_name == "SEED":
        return SeedDataPipe(args)
    elif dataset_name == "MLED":
        return MLEDDataPipe(args)
    else:
        raise ValueError(f"不支持的数据集名称: {args.dataset.name}")
