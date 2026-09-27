import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from cross_validation import MyCrossValidation
from datapipe import MyTrialPipeline, create_datapipe
from utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def config_for_tmp(tmp_path, protocol="validation_subject"):
    cfg = load_config(ROOT / ("config/index-strict.yaml" if protocol == "validation_subject" else "config/index.yaml"))
    cfg.dataset.save_root = str(tmp_path / "processed")
    cfg.dataset.raw_root = str(tmp_path / "raw")
    cfg.training.save_root = str(tmp_path / "checkpoints")
    cfg.training.records_root = str(tmp_path / "records")
    cfg.training.tensorboard_root = str(tmp_path / "tensorboard")
    cfg.training.device = torch.device("cpu")
    cfg.training.max_epochs = 2
    cfg.training.batch_size = 3
    return cfg


def small_cached_dataset(cfg):
    """Three subjects with distinct shifted features, all three classes."""
    pipe = create_datapipe(cfg)
    pipe.preprocessor.NUM_SUBS = 3
    pipe.data_dir.mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(104)
    with h5py.File(pipe.data_path, "w") as f:
        for sid in (1,2,3):
            group = f.create_group(f"subject_{sid:02d}/session_01/trial_01")
            group.create_dataset("label", data=[0,1,2])
            fg = group.create_group("features")
            fg.create_dataset("de", data=(rng.normal(size=(3,37,62,5)) + sid).astype(np.float32))
            fg.create_dataset("wavelet", data=(rng.normal(size=(3,37,3)) + sid).astype(np.float32))
            for t in range(3):
                rows.append({"sample_id": len(rows), "subject_id":sid, "session_id":1,
                             "trial_id":1, "t_index":t, "label":t})
    pd.DataFrame(rows).to_parquet(pipe.index_path)
    pipe.config_path.write_text(json.dumps(pipe.prepared_config), encoding="utf-8")
    return pipe


def test_strict_split_is_subject_disjoint(tmp_path):
    cfg = config_for_tmp(tmp_path)
    cv = MyCrossValidation(cfg)
    cv.datapipe = small_cached_dataset(cfg)
    splits = list(cv._iter_folds())
    assert [(sid, val_sid, train_ids) for sid, val_sid, train_ids, _, _ in splits] == [(1,2,[3]),(2,3,[1]),(3,1,[2])]
    for sid, val_sid, train_ids, data, val in splits:
        assert sid not in train_ids and val_sid not in train_ids and sid != val_sid
        assert len(data[1]) == len(data[3]) == len(val[1]) == 3


def test_normalization_fits_training_only_and_orders_features(tmp_path):
    cv = MyCrossValidation(config_for_tmp(tmp_path))
    rng = np.random.default_rng(19)
    train = {"wavelet": rng.normal(size=(3,37,3)), "de":rng.normal(size=(3,37,62,5))}
    val = {k: v + 500 for k,v in train.items()}
    test = {k: v + 1000 for k,v in train.items()}
    expected = {k: v.copy() for k,v in train.items()}
    result = cv.preprocess(train, np.array([0,1,2]), test, np.array([0,1,2]), validation=(val, np.array([0,1,2])))
    assert result[0][0].shape[-2:] == (62,5)  # DE first, irrespective of dict order.
    assert result[0][1].shape[-1] == 3
    np.testing.assert_allclose(cv.normalization_state["wavelet__mean"], expected["wavelet"].mean((0,1),keepdims=True))
    assert abs(result[0][0].mean().item()) < 1e-6
    assert result[2][1].mean() > 100 and result[4][1].mean() > 100


@pytest.mark.parametrize("protocol", ["validation_subject", "legacy_test"])
def test_train_select_restore_and_re_evaluate_same_predictions(tmp_path, protocol):
    cfg = config_for_tmp(tmp_path, protocol)
    cv = MyCrossValidation(cfg)
    cv.datapipe = small_cached_dataset(cfg)
    calls = []
    original_eval = cv.trainer.eval_one_epoch

    def spy(model, loader, criterion, *callbacks):
        # Verify the held-out test labels are never used to select strict epochs.
        calls.append(loader)
        return original_eval(model, loader, criterion, *callbacks)

    cv.trainer.eval_one_epoch = spy
    cv.leave_one_sub_out(subjects=[1])
    assert len(calls) == cfg.training.max_epochs + 1
    if protocol == "validation_subject":
        assert all(loader is not calls[-1] for loader in calls[:-1])
    records = cfg.training.records_dir
    fold = json.loads((records / "loso_sub_1_fold.json").read_text())
    assert fold["validation_subject_id"] == (2 if protocol == "validation_subject" else None)
    assert fold["training_subject_ids"] == ([3] if protocol == "validation_subject" else [2,3])
    assert fold["seed"] == cfg.reproduce.random_seed + 1
    summary = json.loads((records / "loso_summary.json").read_text())
    assert summary["complete_loso"] is False
    assert summary["selection_protocol"] == protocol
    assert (cfg.training.save_dir / fold["checkpoint"]).is_file()
    before = np.load(records / "loso_sub_1_predictions.npz")
    output = records / "evaluation"
    cv.evaluate(output_dir=output, subjects=[1])
    after = np.load(output / "loso_sub_1_predictions.npz")
    for key in ("trues","preds","probs"):
        np.testing.assert_allclose(before[key], after[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="checkpoint directory"):
        cv.evaluate(model_path=cfg.training.save_dir / fold["checkpoint"])
    with pytest.raises(FileExistsError, match="already exists"):
        cv.leave_one_sub_out(subjects=[1])
    if protocol == "validation_subject":
        cv.leave_one_sub_out(subjects=[2,3])
        complete = json.loads((records / "loso_summary.json").read_text())
        assert complete["subjects"] == [1,2,3] and complete["complete_loso"]
        folds = [json.loads((records / f"loso_sub_{sid}_fold.json").read_text()) for sid in (1,2,3)]
        np.testing.assert_allclose(complete["accuracy_mean"], np.mean([f["accuracy"] for f in folds]))


def test_cache_fingerprint_detects_descriptor_settings_and_version(tmp_path):
    cfg = config_for_tmp(tmp_path)
    pipe = small_cached_dataset(cfg)
    assert pipe.is_prepared()
    meta = json.loads(pipe.config_path.read_text())
    meta["wavelet"]["mode"] = "periodization"
    pipe.config_path.write_text(json.dumps(meta))
    assert not pipe.is_prepared()


def test_strict_rejects_test_subject_normalization(tmp_path):
    cfg = config_for_tmp(tmp_path)
    cfg.dataset.online_transform = "sub_internal"
    cv = MyCrossValidation(cfg)
    with pytest.raises(ValueError, match="train-fitted"):
        next(cv._iter_folds([1]))


def test_config_inheritance_cycle(tmp_path):
    a,b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("extends: b.yaml\n")
    b.write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(a)


def test_raw_trial_to_feature_cache_and_subject_loader(tmp_path):
    cfg = config_for_tmp(tmp_path)
    pipe = create_datapipe(cfg)
    rng = np.random.default_rng(20)
    trial = {"eeg":rng.normal(size=(62,4800)), "fs":200, "label":2,
             "index":{"subject_id":1,"session_id":1,"trial_id":1},
             "path":"subject_01/session_01/trial_01"}
    # Exercise the real writer, index and metadata path without licensed recordings.
    pipe.preprocessor.iter_raw_trials = lambda: iter([trial])
    pipe.build()
    assert pipe.is_prepared()
    features, labels = pipe.load_one_sub(1)
    assert features["de"].shape == (2,37,62,5)
    assert features["wavelet"].shape == (2,37,3)
    np.testing.assert_array_equal(labels, [2,2])
    index = pd.read_parquet(pipe.index_path)
    np.testing.assert_array_equal(index["t_start"], [0,4])
    # A failing rebuild invalidates the old success marker.
    def fail():
        raise ValueError("synthetic preprocessing failure")
    pipe.preprocessor.run = fail
    with pytest.raises(ValueError, match="failure"):
        pipe.build()
    assert not pipe.is_prepared()
