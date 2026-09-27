import numpy as np
import pywt

ch_emotiv = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
             'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
             'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4',
             'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2',
             'FT7', 'T7', 'TP7', 'FT8', 'T8', 'TP8']


def name2index(name, slist):
    """
    将字符串名转换为索引

    Parameters
    --------
    name: str | list[str]
        字符串名
    slist: list[str]
        字符串列表

    Returns
    --------
    idx: int | list[int] | slice
    """
    if name is None:
        idx = slice(None)
    elif isinstance(name, list):
        idx = [slist.index(n) for n in name]
        if len(idx) == 1:
            idx = idx[0]
    else:
        idx = slist.index(name)
    return idx


class CognitiveMetrics:
    """
    计算脑电信号的注意力、压力、脑力负荷指数

    Legacy compatibility interface. Its band names are positional aliases for
    a4/d4/d3/d2/d1, not physical DE frequency bands. ECFNet uses the explicit
    WaveletDescriptors interface in utils/wavelet.py instead.

    Parameters
    --------
    raw: numpy.ndarray
        原始EEG数据, 应确保其带通滤波至1~64Hz, shape=(ch, points)
    ch_names: list[str]
        通道名列表
    """

    def __init__(self, raw, ch_names=None):
        self.raw = raw
        self.ch_names = ch_emotiv if ch_names is None else ch_names
        self.band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']

        if len(self.ch_names) != self.raw.shape[0]:
            raise ValueError('Channel names length should be equal to data channel number')

        # 离散小波变换, [cA4, cD4, cD3, cD2, cD1]
        coeffs = pywt.wavedec(raw, 'db4', level=4)
        # 计算每个频带的能量
        energy = [np.sum(np.square(c), axis=-1) for c in coeffs]
        # 按频带堆叠, shape=(bands, ch)
        self._energy = np.stack(energy)
        # 计算每个子频带的相对能量, shape=(bands, ch)
        self._r_energy = self._energy / np.sum(self._energy, axis=0, keepdims=True)

    def energy(self, band=None, ch=None, relative=True, keepdims=False):
        """
        获取能量数据

        Parameters
        --------
        band: str | list[str]
            频带名, 默认所有频带
        ch: str | list[str]
            通道名, 默认所有通道
        relative: bool
            若为True, 则返回相对能量数据, 否则返回绝对能量数据
        keepdims: bool
            是否强制保留维度为(num_band, num_ch)

        Returns
        --------
        energy: numpy.ndarray
            能量数据, shape=(num_band, num_ch)
        """
        # 获取频带名对应的索引
        band_idx = name2index(band, self.band_names)
        # 获取通道名对应的索引
        ch_idx = name2index(ch, self.ch_names)

        all_energy = self._r_energy if relative else self._energy

        if isinstance(band_idx, list) and isinstance(ch_idx, list):
            # 使用np.ix_进行网格索引, 避免点对点索引
            energy = all_energy[np.ix_(band_idx, ch_idx)]
        else:
            energy = all_energy[band_idx, ch_idx]

        if keepdims:
            if energy.ndim == 0:
                # 单频带, 单通道
                energy = energy.reshape(1, 1)
            elif energy.ndim == 1:
                if isinstance(band_idx, list):
                    # 多频带, 单通道
                    energy = energy.reshape(-1, 1)
                elif isinstance(ch_idx, list):
                    # 单频带, 多通道
                    energy = energy.reshape(1, -1)

        return energy

    def asymmetry(self, band='alpha', l_ch='F7', r_ch='F8'):
        """
        计算左右脑不对称性

        Parameters
        --------
        band: str
            频带名
        l_ch: str
            左侧通道名
        r_ch: str
            右侧通道名

        Returns
        --------
        asym: float
            左右脑不对称性
        """
        l_energy = self.energy(band, l_ch)
        r_energy = self.energy(band, r_ch)
        asym = np.log(l_energy) - np.log(r_energy)
        return asym

    def band_ratio(self, band1='theta', band2='beta', ch=None):
        """
        计算两个频带的比值

        Parameters
        --------
        band1: str
            频带1
        band2: str
            频带2
        ch: str | list[str]
            通道名

        Returns
        --------
        ratio: float
            两个频带的比值
        """
        energy1 = np.sum(self.energy(band1, ch)).item()
        energy2 = np.sum(self.energy(band2, ch)).item()
        ratio = energy1 / (energy2 + 1e-10)
        return ratio

    def weng(self, ch=None):

        """
        计算Weng指数

        Parameters
        --------
        ch: str | list[str]
            通道名

        Returns
        --------
        weng: float
            Weng指数

        Notes
        --------
        Weng, Energy of the approximate coefficients, 指近似系数的能量
        """
        weng = np.sum(self.energy('theta', ch))
        return weng


if __name__ == '__main__':
    raw = np.random.randn(62, 1000)
    cm = CognitiveMetrics(raw)
    print(cm.band_ratio(ch=['AF3', 'AF4', 'F3', 'F4']))
    print(cm.asymmetry(l_ch='F7', r_ch='F8'))
    print(cm.weng(ch=['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6']))
