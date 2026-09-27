import numpy as np
from scipy import stats
from scipy.signal import welch
from scipy.integrate import simpson

def diff_entropy(raw, axis=-1, normal=True):
    """
    微分熵

    Parameters
    ----------
    raw: numpy.ndarray
        原始信号数据, 支持单通道或多通道
    axis: int
        要在哪个轴上求取微分熵, 默认为最后一个轴
    normal: bool
        数据是否符合正态分布

    Returns
    ----------
    de: numpy.number | numpy.ndarray
        微分熵
    """
    if normal:
        var = np.var(raw, axis=axis)
        if np.any(var <= 0) or not np.isfinite(var).all():
            raise ValueError("DE requires strictly positive finite population variance")
        de = 0.5 * np.log(2 * np.pi * np.e * var)
    else:
        de = stats.differential_entropy(raw, axis=axis)
    return de


def moving_average(raw, k=3, axis=-1):
    """
    简单移动平均过滤
    
    Parameters
    ----------
    raw: numpy.ndarray
        原始信号数据, 支持单通道或多通道
    k: int
        移动平均窗口大小
    axis: int
        要在哪个轴上进行移动平均, 默认为最后一个轴

    Returns
    ----------
    moving_avg: numpy.ndarray
        移动平均过滤后的信号

    Notes:
    -----
    对于窗口大小k, 建议使用奇数; 使用偶数时窗口会右对齐, 导致结果相对于原始信号在指定轴方向上整体向前偏移约 k//2 个采样点
    """
    if k <= 1:
        return raw
    
    raw = np.asarray(raw)
    axis = axis % raw.ndim
    
    pad_width = [(0, 0)] * raw.ndim
    pad_width[axis] = (k // 2, k - 1 - k // 2)
    padded = np.pad(raw, pad_width, mode='edge')

    # 在 axis 维度前补一个 0, 方便后续计算累积和
    pad_shape = list(padded.shape)
    pad_shape[axis] = 1
    zero = np.zeros(pad_shape, dtype=padded.dtype)
    
    padded = np.concatenate([zero, padded], axis=axis)
    cumsum = np.cumsum(padded, axis=axis)

    slices1 = [slice(None)] * raw.ndim
    slices2 = [slice(None)] * raw.ndim
    slices1[axis] = slice(k, None)
    slices2[axis] = slice(None, -k)

    moving_avg = (cumsum[tuple(slices1)] - cumsum[tuple(slices2)]) / k
    return moving_avg


def bandpower_welch(raw, fs, filter_bank, window_length=1.0, overlap=None, relative=True, axis=-1):
    """
    使用Welch方法计算频带功率

    Parameters
    ----------
    raw: numpy.ndarray
        原始信号数据, 支持单通道或多通道, 形状为 (..., T, ...)
    fs: float
        采样频率
    filter_bank: list[tuple[int, int]]
        频带划分, 每个元素为 (low, high)
    window_length: float
        窗口长度, 单位秒
    overlap: float
        窗口重叠部分长度, 单位秒
    relative: bool
        是否计算相对功率
    axis: int
        要在哪个轴上计算功率谱密度, 默认为最后一个轴

    Returns
    ----------
    band_powers: numpy.ndarray
        频带功率, 形状为 (..., num_bands, ...)
    """
    nperseg = int(window_length * fs)
    noverlap = int(overlap * fs) if overlap is not None else nperseg // 2

    freqs, psd = welch(raw, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=axis)

    dx = freqs[1] - freqs[0]
    if relative:
        total_power = simpson(psd, dx=dx, axis=axis)

    band_powers = []
    for (low, high) in filter_bank:
        mask = (freqs >= low) & (freqs <= high)
        band = np.take(psd, np.where(mask)[0], axis=axis)
        band_power = simpson(band, dx=dx, axis=axis)  # (..., T, ...) -> (..., ...)
        if relative:
            band_power /= total_power
        band_powers.append(band_power)

    band_powers = np.stack(band_powers, axis=axis)  # (..., num_bands, ...)

    return band_powers
