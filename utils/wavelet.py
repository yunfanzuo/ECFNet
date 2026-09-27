"""Fixed db4 descriptors in ECFNet Eqs. (2)-(3).

Wavelet scales are not the Butterworth DE frequency bands. At 200 Hz,
their nominal intervals are a4: 0-6.25, d4: 6.25-12.5, d3: 12.5-25,
d2: 25-50, d1: 50-100 Hz, before the 1-64 Hz input filter.
"""
import numpy as np
import pywt


class WaveletDescriptors:
    scales = ("a4", "d4", "d3", "d2", "d1")
    frontal_ratio = ("AF3", "AF4", "F3", "F4")
    frontal_energy = frontal_ratio + ("F7", "F8", "FC5", "FC6")

    def __init__(self, filtered_eeg, ch_names):
        """Accept already 1-64 Hz filtered (..., channels, samples) EEG."""
        raw = np.asarray(filtered_eeg, dtype=np.float64)
        if raw.ndim < 2 or raw.shape[-2] != len(ch_names):
            raise ValueError("EEG channel axis does not match ch_names")
        if not np.isfinite(raw).all():
            raise ValueError("Wavelet input contains non-finite values")
        if len(set(ch_names)) != len(ch_names):
            raise ValueError("Channel names must be unique")
        self.ch_names = list(ch_names)
        coeffs = pywt.wavedec(raw, "db4", mode="symmetric", level=4, axis=-1)
        self.coefficient_counts = tuple(c.shape[-1] for c in coeffs)
        self.energy = np.stack([np.square(c).sum(axis=-1) for c in coeffs], axis=-1)
        total = self.energy.sum(axis=-1, keepdims=True)
        if np.any(total <= 0) or not np.isfinite(total).all():
            raise ValueError("Wavelet relative energy requires positive finite total energy")
        self.relative_energy = self.energy / total

    def compute(self):
        r = self.relative_energy
        a = [self.ch_names.index(c) for c in self.frontal_ratio]
        w = [self.ch_names.index(c) for c in self.frontal_energy]
        left = r[..., self.ch_names.index("F7"), 2]
        right = r[..., self.ch_names.index("F8"), 2]
        if np.any(left <= 0) or np.any(right <= 0):
            raise ValueError("Lateral asymmetry requires positive d3 energies at F7 and F8")
        u1 = r[..., a, 1].sum(axis=-1) / (r[..., a, 3].sum(axis=-1) + 1e-10)
        u2 = np.log(left) - np.log(right)
        u3 = r[..., w, 1].sum(axis=-1)
        return np.stack((u1, u2, u3), axis=-1)
