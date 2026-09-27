from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import numpy as np

from utils.feature_process import bandpower_welch, diff_entropy
from utils.signal_process import butter_bandpass, downsampling, sliding_window_view

class Transform(ABC):
    """单个预处理步骤"""

    @abstractmethod
    def __call__(self, raw: Dict[str, Any], *, inplace = False, **kwargs) -> Dict[str, Any]:
        ...

class BandPassFilter(Transform):
    """带通滤波器"""

    def __init__(self, low_cut: float, high_cut: float, *, order: int = 4, use_lfilter: bool = False):
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.order = order
        self.use_lfilter = use_lfilter

    def __call__(self, raw: Dict[str, Any], *, inplace = False, axis: int = -1) -> Dict[str, Any]:
        eeg: np.ndarray = raw["eeg"]
        fs: int = raw["fs"]
        target = raw if inplace else dict(raw)

        filtered = butter_bandpass(
            eeg,
            low_cut=self.low_cut,
            high_cut=self.high_cut,
            fs=fs,
            axis=axis,
            order=self.order,
            use_lfilter=self.use_lfilter,
        )

        target["eeg"] = filtered
        return target

class Downsampling(Transform):
    """降采样"""

    def __init__(self, target_fs: int):
        self.target_fs = target_fs

    def __call__(self, raw: Dict[str, Any], *, inplace = False, axis: int = -1) -> Dict[str, Any]:
        eeg: np.ndarray = raw["eeg"]
        fs: int = raw["fs"]
        target = raw if inplace else dict(raw)

        downsampled = downsampling(
            eeg,
            original_fs=fs,
            target_fs=self.target_fs,
            axis=axis
        )

        target["eeg"] = downsampled
        target["fs"] = self.target_fs
        return target
    
class BandDecompose(Transform):
    """频段分解"""

    def __init__(self, filter_bank: list[tuple[int, int]], *, order: int = 4, use_lfilter: bool = False):
        self.bands = filter_bank
        self.order = order
        self.use_lfilter = use_lfilter

    def __call__(self, raw: Dict[str, Any], *, inplace = False, axis: int = -1) -> Dict[str, Any]:
        eeg: np.ndarray = raw["eeg"]  # (..., T, ...)
        fs: int = raw["fs"]
        target = raw if inplace else dict(raw)

        decomposed = []
        for low_cut, high_cut in self.bands:
            filtered = butter_bandpass(  # (..., T, ...)
                eeg,
                low_cut=low_cut,
                high_cut=high_cut,
                fs=fs,
                axis=axis,
                order=self.order,
                use_lfilter=self.use_lfilter,
            )
            decomposed.append(filtered)
        decomposed = np.stack(decomposed, axis=axis - 1 if axis < 0 else axis)  # (..., num_bands, T, ...)
        
        target["eeg"] = decomposed
        return target

class TrailSpliter(Transform):
    """
    试次样本分割
    
    Args:
        window(float): 试次窗口长度, 单位秒
        stride(float): 试次窗口滑动步长, 单位秒
        sub_window(float, optional): 子窗口长度, 单位秒. 默认不进行子窗口划分
        sub_stride(float, optional): 子窗口滑动步长, 单位秒. 默认不进行子窗口划分
    """

    def __init__(
        self,
        window: float, stride: float,
        sub_window: float = None, sub_stride: float = None
    ):
        self.window = window
        self.stride = stride
        self.sub_window = sub_window
        self.sub_stride = sub_stride

    def __call__(self, raw: Dict[str, Any], *, inplace = False, axis: int = -1) -> Dict[str, Any]:
        """
        分割试次样本
        Args:
            raw: Dict[str, Any]
                包含原始信号数据的字典, shape=(..., T, ...)
            inplace: bool
                是否在原字典上进行修改, 默认为 False
            axis: int
                要在哪个轴上进行试次划分, 默认为最后一个轴
        Returns:
            Dict[str, Any]
                包含分割后信号数据的字典, shape=(num_windows, ..., window_size, ...),
                或 (num_windows, num_sub_windows, ..., sub_window_size, ...) 如果进行了子窗口划分
        """
        eeg: np.ndarray = raw["eeg"]  # (..., T, ...)
        fs: int = raw["fs"]
        target = raw if inplace else dict(raw)
        axis = axis % eeg.ndim

        # (..., num_windows, window_size, ...)
        segments = sliding_window_view(eeg, self.window, self.stride, fs, axis)

        if self.sub_window is not None and self.sub_stride is not None:
            sub_axis = axis + 1
            # (..., num_windows, num_sub_windows, sub_window_size, ...)
            segments = sliding_window_view(segments, self.sub_window, self.sub_stride, fs, sub_axis)
            # (num_windows, num_sub_windows, ..., sub_window_size, ...)
            segments = np.moveaxis(segments, [axis, sub_axis], [0, 1])
        else:
            # (num_windows, ..., window_size, ...)
            segments = np.moveaxis(segments, axis, 0)
        
        batches = int(segments.shape[0])

        target["eeg"] = segments
        # 复制标签
        target["label"] = np.full(batches, raw["label"])
        # 重写索引
        target["index"] = {
            **{k_id: [v] * batches for k_id, v in raw["index"].items()}, # 扩展原索引, xxx_id
            "t_index": list(range(batches)),
            "t_start": [i * self.stride for i in range(batches)],
            "t_end": [i * self.stride + self.window for i in range(batches)]
        }
        # 添加批量数
        target["count"] = batches
        return target

class TemporalFeatureExtract(Transform):
    """时域特征提取"""

    def __init__(self, feature_names: Union[list[str], str]):
        self.feature_names = feature_names if isinstance(feature_names, list) else [feature_names]

    def __call__(self, raw: Dict[str, Any], *, inplace=False, axis: int = -1, **kwargs) -> Dict[str, Any]:
        """
        进行特征提取
        Args:
            raw: Dict[str, Any]
                包含原始信号数据的字典, 其中 "eeg" 键对应的值为 numpy.ndarray, 形状为 (..., T, ...)
            inplace: bool
                是否在原字典上进行修改, 默认为 False
            axis: int
                要在哪个轴上计算特征, 默认为最后一个轴
            kwargs: 其他可选参数
                normal: bool
                    计算微分熵时数据是否符合正态分布, 默认为 True
                filter_bank: list[tuple[int, int]]
                    计算频带功率时的频带划分, 每个元素为 (low, high)
        Notes:
            特征名为 "de" "mean" "std" "max" "min" 时, 在 axis 轴上被压缩, shape 变为 (..., ...);
            特征名为 "psd" "r_psd" 时, 在 axis 轴上被替换为频带轴, shape 变为 (..., num_bands, ...);
        """

        eeg: np.ndarray = raw["eeg"]  # (..., T, ...)
        fs: int = raw["fs"]
        target = raw if inplace else dict(raw)

        features = {}
        for key in self.feature_names:
            if key == "de":
                feat = diff_entropy(eeg, axis=axis, normal=kwargs.get("normal", True))
            elif key in ("psd", "r_psd"):
                filter_bank: list[tuple[int, int]] = kwargs["filter_bank"]
                feat = bandpower_welch(eeg, axis=axis, fs=fs, filter_bank=filter_bank, relative=(key == "r_psd"))
            elif key == "mean":
                feat = np.mean(eeg, axis=axis)
            elif key == "std":
                feat = np.std(eeg, axis=axis)
            elif key == "max":
                feat = np.max(eeg, axis=axis)
            elif key == "min":
                feat = np.min(eeg, axis=axis)
            else:
                from utils import logger
                logger.error(f"不支持的特征名称: {key}")
                continue
            features[key] = feat

        if "features" in target:
            target["features"].update(features)
        else:
            target["features"] = features
        return target

class ChannelReorder(Transform):
    """通道重排"""

    def __init__(self, order: list[int]):
        self.order = order

    def __call__(self, raw: Dict[str, Any], *, inplace=False, axis: int = 0) -> Dict[str, Any]:
        eeg: np.ndarray = raw["eeg"]  # (..., C, ...)
        target = raw if inplace else dict(raw)

        reordered = np.take(eeg, indices=self.order, axis=axis)

        target["eeg"] = reordered
        return target

class ComposeTransform(Transform):
    """多个预处理步骤组合"""

    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, raw: Dict[str, Any], transform_args: list[dict[str, Any]]) -> Dict[str, Any]:
        for transform, kargs in zip(self.transforms, transform_args):
            raw = transform(raw, **kargs)
        return raw
