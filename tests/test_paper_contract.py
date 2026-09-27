from pathlib import Path

import numpy as np
import pywt
import pytest
import torch
import torch.nn.functional as F

from datapipe import MyTrialPipeline, SeedPreprocessor
from net import create_model
from net.layer import GlobalAttentionPooling, MultiheadAttention, SGConv
from net.model import GlobalGraphEmbedding, LocalGraphEmbedding
from utils.channel import get_ch_index, get_ch_name, get_local_ch_num
from utils.cognitive import CognitiveMetrics
from utils.config import load_config
from utils.feature_process import diff_entropy
from utils.signal_process import butter_bandpass
from utils.wavelet import WaveletDescriptors

ROOT = Path(__file__).resolve().parents[1]


def test_local_mapping_scalar_and_region_mean_equation4():
    torch.manual_seed(7)
    layer = LocalGraphEmbedding(1, 64, 128, [2, 3], dropout=0).double()
    x = torch.randn(2, 5, 1, dtype=torch.float64)
    assert layer.weights.shape == layer.bias.shape == (5, 1)
    transformed = F.linear(x, layer.proj1.weight, layer.proj1.bias)
    transformed = F.relu(transformed * layer.weights - layer.bias)
    transformed = F.linear(transformed, layer.proj2.weight, layer.proj2.bias)
    expected = torch.stack([transformed[:, :2].mean(1), transformed[:, 2:].mean(1)], 1)
    torch.testing.assert_close(layer(x), expected)


def test_adjacency_propagation_and_residual_equations5to8():
    torch.manual_seed(8)
    layer = GlobalGraphEmbedding(3, 128, 128, 64, [2], dropout=0).double()
    h = torch.randn(2, 3, 128, dtype=torch.float64)
    conv = layer.layers[0][0]
    a = F.relu((h @ h.transpose(-2, -1)) * (conv.attn_mask + conv.attn_mask.T)) + torch.eye(3)
    assert torch.all(a.diagonal(dim1=-2, dim2=-1) >= 1)
    torch.testing.assert_close(a, a.transpose(-2, -1))
    d = torch.diag_embed(a.sum(-1).pow(-0.5))
    s = d @ a @ d
    z = F.relu(conv.ln2(F.linear(s @ s @ conv.ln1(h), conv.conv.lin.weight, conv.conv.lin.bias)))
    query = layer.layers[0][2].query
    weights = torch.softmax((z @ query.squeeze(0).T) / np.sqrt(128), dim=1)
    pooled = (weights * z).sum(1)
    residual = F.relu(F.linear(h.flatten(1), layer.linear[0].weight, layer.linear[0].bias))
    expected = F.linear((pooled + residual) / 2, layer.proj.weight, layer.proj.bias)
    torch.testing.assert_close(layer(h), expected)


def test_attention_pooling_equation7():
    layer = GlobalAttentionPooling(4).double()
    x = torch.randn(2, 5, 4, dtype=torch.float64)
    scores = torch.einsum("bnd,d->bn", x, layer.query.flatten()) / 2
    expected = (scores.softmax(-1)[..., None] * x).sum(1)
    torch.testing.assert_close(layer(x), expected)


def test_wavelet_descriptors_equations2and3_and_legacy_agreement():
    raw = np.random.default_rng(17).normal(size=(62, 400))
    names = get_ch_name("SEED", "original")
    raw = butter_bandpass(raw, 1, 64, fs=200)
    descriptor = WaveletDescriptors(raw, names)
    coeffs = pywt.wavedec(raw, "db4", level=4, mode="symmetric")
    energy = np.array([(c * c).sum(-1) for c in coeffs])
    r = energy / energy.sum(0)
    a = [names.index(c) for c in ["AF3", "AF4", "F3", "F4"]]
    w = [names.index(c) for c in ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6"]]
    expected = [r[1, a].sum() / (r[3, a].sum() + 1e-10),
                np.log(r[2, names.index("F7")]) - np.log(r[2, names.index("F8")]), r[1, w].sum()]
    np.testing.assert_allclose(descriptor.compute(), expected, rtol=1e-12)
    assert descriptor.coefficient_counts == tuple(c.shape[-1] for c in coeffs)
    legacy = CognitiveMetrics(raw, names)
    np.testing.assert_allclose(descriptor.compute(), [legacy.band_ratio("theta", "beta", list(descriptor.frontal_ratio)),
                                                      legacy.asymmetry(), legacy.weng(list(descriptor.frontal_energy))])
    np.testing.assert_allclose(descriptor.relative_energy.sum(-1), 1)


def test_wavelet_vectorized_windows_and_channel_reorder():
    raw = np.random.default_rng(18).normal(size=(3, 62, 400))
    names = get_ch_name("SEED", "original")
    expected = np.stack([WaveletDescriptors(x, names).compute() for x in raw])
    for graph in ("general", "frontal", "hemisphere"):
        indices = get_ch_index("SEED", graph)
        actual = WaveletDescriptors(raw[:, indices], get_ch_name("SEED", graph)).compute()
        np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_de_population_variance_equation1_and_invalid_energy():
    raw = np.array([[1., 2., 6., 8.], [2., 4., 8., 9.]])
    np.testing.assert_allclose(diff_entropy(raw), 0.5 * np.log(2 * np.pi * np.e * raw.var(-1, ddof=0)))
    with pytest.raises(ValueError, match="variance"):
        diff_entropy(np.zeros((62, 400)))
    with pytest.raises(ValueError, match="energy"):
        WaveletDescriptors(np.zeros((62, 400)), get_ch_name("SEED", "original"))


@pytest.mark.parametrize("partition,regions", [("general",11),("frontal",14),("hemisphere",19),("original",62)])
def test_region_partition(partition, regions):
    names = get_ch_name("SEED", partition)
    assert len(names) == len(set(names)) == 62
    assert set(names) == set(get_ch_name("SEED", "original"))
    counts = get_local_ch_num("SEED", partition)
    assert len(counts) == regions and sum(counts) == 62
    if partition == "general":
        assert counts == [3, 2, 9, 7, 7, 7, 9, 7, 5, 3, 3]


@pytest.mark.parametrize("need_weights", [False, True])
def test_attention_eval_disables_dropout_even_with_gradients(need_weights):
    layer = MultiheadAttention(64, 4, 16, dropout=0.5).eval()
    q, kv = torch.randn(2, 7, 64), torch.randn(2, 5, 64)
    y1, _ = layer(q, kv, kv, need_weights=need_weights)
    y2, _ = layer(q, kv, kv, need_weights=need_weights)
    torch.testing.assert_close(y1, y2, rtol=0, atol=0)
    y_fast, _ = layer(q, kv, kv)
    y_weights, _ = layer(q, kv, kv, need_weights=True)
    torch.testing.assert_close(y_fast, y_weights)
    layer.train()
    y1, _ = layer(q, kv, kv, need_weights=need_weights)
    y2, _ = layer(q, kv, kv, need_weights=need_weights)
    assert not torch.equal(y1, y2)


def test_ecfnet_independent_permutation_invariance_and_gradient_flow():
    cfg = load_config(ROOT / "config/index.yaml")
    model = create_model(cfg).eval()
    torch.manual_seed(44)
    x, u = torch.randn(2, 37, 62, 5), torch.randn(2, 37, 3)
    y = model((x, u))
    shuffled = model((x[:, torch.randperm(37)], u[:, torch.randperm(37)]))
    torch.testing.assert_close(y, shuffled, rtol=1e-5, atol=2e-6)
    loss = F.cross_entropy(y, torch.tensor([0, 2]), label_smoothing=0.05)
    loss.backward()
    for encoder in model.sb_learning.graph_encoders:
        assert encoder[0].proj1.weight.grad is not None
        assert torch.isfinite(encoder[0].proj1.weight.grad).all()
    assert model.fusion_layer.proj_cog.weight.grad.abs().sum() > 0
    assert model.fusion_layer.proj_deep.weight.grad.abs().sum() > 0
    assert model.fusion_layer.attention.attention.head_dim == 16


def test_label_smoothed_loss_equation11():
    logits = torch.tensor([[1.1, -0.4, 0.7], [0.1, 0.4, -0.8]], dtype=torch.float64)
    y = torch.tensor([0, 1])
    target = F.one_hot(y, 3).double() * 0.95 + 0.05 / 3
    expected = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    torch.testing.assert_close(F.cross_entropy(logits, y, label_smoothing=0.05), expected)


def test_full_trial_pipeline_shape_window_count_and_timestamps():
    cfg = load_config(ROOT / "config/index.yaml")
    raw = np.random.default_rng(5).normal(size=(62, 4800))  # 24 seconds -> 2 segments.
    trial = {"eeg": raw, "fs":200, "label":1, "index":{"subject_id":1}, "path":"trial"}
    pipeline = MyTrialPipeline(cfg)
    result = pipeline.process_trial(trial)
    assert result["features"]["de"].shape == (2, 37, 62, 5)
    assert result["features"]["wavelet"].shape == (2, 37, 3)
    assert result["index"]["t_start"] == [0, 4]
    assert result["index"]["t_end"] == [20, 24]
    assert pipeline.meta_attrs["wavelet"]["mode"] == "symmetric"
    reordered = raw[get_ch_index("SEED", "general")]
    first_window = butter_bandpass(reordered[:, :400], 1, 64, 200)
    expected_u = WaveletDescriptors(first_window, pipeline.ch_names).compute()
    np.testing.assert_allclose(result["features"]["wavelet"][0, 0], expected_u, rtol=1e-6)
    filtered = butter_bandpass(reordered, 1, 4, 200)
    np.testing.assert_allclose(result["features"]["de"][0, 0, :, 0], diff_entropy(filtered[:, :400]), rtol=1e-6)


@pytest.mark.parametrize("file", sorted((ROOT / "config/ablations").glob("*.yaml")), ids=lambda f:f.stem)
def test_ablation_configs_forward(file):
    cfg = load_config(file)
    model = create_model(cfg)
    x, u = torch.randn(2, 37, 62, 5), torch.randn(2, 37, 3)
    y = model((x, u) if cfg.model.use_descriptors else x)
    assert y.shape == (2, 3) and torch.isfinite(y).all()
    if cfg.model.share_encoders:
        assert hasattr(model.sb_learning, "shared_encoder")
        assert not hasattr(model.sb_learning, "graph_encoders")


def test_seed_mat_numeric_trial_order(tmp_path):
    from scipy.io import savemat
    labels = np.array([-1,0,1] * 5)
    savemat(tmp_path / "label.mat", {"label":labels})
    # Intentionally lexical insertion order: 1,10,...,2,...
    data = {f"person_eeg{i}": np.full((62,4000), i, dtype=np.float32) for i in sorted(range(1,16), key=str)}
    savemat(tmp_path / "1_20131027.mat", data)
    pre = SeedPreprocessor(tmp_path, tmp_path / "processed", None)
    trials = list(pre.load_data_file({"file_path":str(tmp_path / "1_20131027.mat"),"subject_id":1,"session_id":1}))
    assert [t["eeg"][0,0] for t in trials] == list(range(1,16))
    assert [t["label"] for t in trials] == list(labels + 1)
