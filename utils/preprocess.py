from typing import Union

import numpy as np
import torch


class ZScoreStandardizer:
    def __init__(self, axis: int | tuple[int, ...] = None, eps: float = 1e-8):
        self.axis = axis
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=self.axis, keepdims=True)
        self.std = x.std(axis=self.axis, keepdims=True)
        self.std = np.maximum(self.std, self.eps)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer has not been fitted yet.")
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x)


class ChannelWiseStandardizer:
    def __init__(self, axis: int = -2, eps: float = 1e-8):
        self.axis = axis
        self.eps = eps
        self.mean = None
        self.std = None

    @staticmethod
    def _reduce_axis(axis: int, ndim: int) -> tuple[int, ...]:
        axis = axis % ndim
        return tuple(i for i in range(ndim) if i != axis)

    def fit(self, x: np.ndarray):
        axis = self._reduce_axis(self.axis, x.ndim)

        self.mean = x.mean(axis=axis, keepdims=True)
        self.std = x.std(axis=axis, keepdims=True)
        self.std = np.maximum(self.std, self.eps)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer has not been fitted yet.")
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x)


def numpy2tensor(
    *args: Union[np.ndarray, dict[str, np.ndarray]],
    dtype: Union[torch.dtype, str] = torch.float32
):
    out = []
    for data in args:
        if isinstance(data, np.ndarray):
            out.append(torch.from_numpy(data).type(dtype))
        elif isinstance(data, dict):
            data = {k: numpy2tensor(v, dtype=dtype) for k, v in data.items()}
            out.append(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    return tuple(out) if len(out) > 1 else out[0]
