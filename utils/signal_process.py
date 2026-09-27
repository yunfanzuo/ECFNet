import numpy as np
from scipy import signal

def butter_bandpass(raw, low_cut, high_cut, fs, axis=-1, order=4, use_lfilter=False):
    """
    Butterworth带通滤波

    Parameters
    ----------
    raw: numpy.ndarray
        原始信号数据, 支持单通道或多通道
    low_cut: float
        低截止频率
    high_cut: float
        高截止频率
    fs: int
        采样率
    axis: int
        要在哪个轴上进行滤波, 默认为最后一个轴
    order: int
        阶数
    use_lfilter: bool
        使用因果滤波器(``lfilter``, 有相位延迟), 否则使用 ``filtfilt`` (零相位滤波)滤波器,
        对于较短的实时信号, 建议使用 ``lfilter``; 而离线数据, 建议使用 ``filtfilt``. 默认使用 ``filtfilt``

    Returns
    ----------
    filtered: numpy.ndarray
        滤波后的数据
    """
    nyq = 0.5 * fs  # Nyquist频率
    lowpass, highpass = low_cut / nyq, high_cut / nyq
    [b, a] = signal.butter(order, [lowpass, highpass], btype='bandpass')
    if use_lfilter:
        filtered = signal.lfilter(b, a, raw, axis=axis)
    else:
        filtered = signal.filtfilt(b, a, raw, axis=axis)
    return filtered

def downsampling(raw, original_fs, target_fs, axis=-1, ftype='iir', zero_phase=True):
    """
    降采样

    Parameters
    ----------
    raw: numpy.ndarray
        原始信号数据, 支持单通道或多通道
    original_fs: int
        原始采样率
    target_fs: int
        目标采样率
    axis: int
        要在哪个轴上进行降采样, 默认为最后一个轴
    ftype: str
        降采样滤波器类型, 默认为 'iir'
    zero_phase: bool
        是否使用零相位滤波, 默认为 True

    Returns
    ----------
    downsampled: numpy.ndarray
        降采样后的数据
    """
    if original_fs == target_fs:
        return raw
    assert original_fs > target_fs, "目标采样率必须低于原始采样率"

    factor = int(original_fs / target_fs)
    downsampled = signal.decimate(raw, factor, axis=axis, ftype=ftype, zero_phase=zero_phase)
    return downsampled

def sliding_window_view(raw, window, stride, fs, axis=-1):
    r"""
    滑动窗口视图

    Parameters
    ----------
    raw: numpy.ndarray
        原始信号数据, shape=(..., L, ...), 其中 L 是目标 axis 轴的数据点长度
    window: float
        窗口大小(秒)
    stride: float
        步长(秒)
    fs: int
        采样率
    axis: int
        要在哪个轴上进行滑动窗口切片, 默认为最后一个轴

    Returns
    ----------
    fragments: numpy.ndarray
        切片后的数据, 新增一个维度表示窗口数量, shape=(..., num_windows, window_size, ...)

    Notes
    ----------
    - 窗口数量 :math:`num\_windows = \left\lfloor \frac{L - window_{size}}{step_{size}} \right\rfloor + 1`
    - 需要注意, 如果末尾的数据点不足以构成一个完整的窗口, 则会被丢弃
    """
    x = np.asarray(raw)
    window_size = int(window * fs)
    step_size = int(stride * fs)
    
    if window_size <= 0:
        return x
    if step_size <= 0:
        raise ValueError("步长必须非负")

    if axis < 0:
        axis += x.ndim
    assert 0 <= axis < x.ndim, "axis out of range"

    L = x.shape[axis]
    if L < window_size:
        raise ValueError("信号长度小于窗口大小, 无法进行切片")

    num_windows = (L - window_size) // step_size + 1

    new_shape = x.shape[:axis] + (num_windows, window_size) + x.shape[axis + 1:]
    new_strides = x.strides[:axis] + (step_size * x.strides[axis], x.strides[axis]) + x.strides[axis + 1:]

    return np.lib.stride_tricks.as_strided(x, shape=new_shape, strides=new_strides)
