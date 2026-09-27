import numpy as np
import pytest
import torch

pytest.importorskip("mne")
pytest.importorskip("mne_connectivity")

from visualize import ModelInterpreter, _loso_collect_data
from net import create_model
from test_evaluation_protocol import config_for_tmp, small_cached_dataset


def test_interpretation_loader_uses_saved_stats_and_only_test_subject(tmp_path):
    cfg = config_for_tmp(tmp_path)
    pipe = small_cached_dataset(cfg)
    records = cfg.training.records_dir
    records.mkdir(parents=True)
    mean_de, std_de = np.full((1,1,62,1), -4), np.full((1,1,62,1), 2)
    mean_u, std_u = np.full((1,1,3), -7), np.full((1,1,3), 3)
    np.savez(records / "norm.npz", de__mean=mean_de, de__std=std_de,
             wavelet__mean=mean_u, wavelet__std=std_u)
    (records / "loso_sub_1_fold.json").write_text('{"normalization_file":"norm.npz"}')
    raw, _ = pipe.load_one_sub(1)
    loader, has_descriptors = _loso_collect_data(cfg, 1, batch_size=3)
    x,u = next(iter(loader))
    assert len(loader.dataset) == 3 and has_descriptors
    np.testing.assert_allclose(x, (raw["de"] - mean_de) / std_de, rtol=1e-6)
    np.testing.assert_allclose(u, (raw["wavelet"] - mean_u) / std_u, rtol=1e-6)
    model = create_model(cfg).eval()
    interpreter = ModelInterpreter(model, band_names=list(cfg.dataset.freq_bands))
    with interpreter.capture(x,u):
        assert len(interpreter.gcn_adjs) == 5
        assert interpreter.cross_attn_weights[0].shape == (37,37)
        np.testing.assert_allclose(interpreter.cross_attn_weights[0].sum(-1),1,rtol=1e-6)
    assert not interpreter._hooks
