import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datapipe import create_datapipe
from dataset import MyEEGDataset
from net import create_model
from train import ModelTrainer
from utils import console, logger
from utils.preprocess import ChannelWiseStandardizer, ZScoreStandardizer, numpy2tensor
from utils.tool import EvaluateMeter, MofNCurrentColumn, build_adamw_param_groups, plot_confusion_matrix, plot_roc_curve, seed_everything


def sub_internal_standardize(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """对每个被试的数据进行标准化"""
    standardized_data = {}
    for feat, values in data.items():
        keep_dims = 1 if feat == "wavelet" or feat.startswith("cog_") else 2
        reduce_dims = max(values.ndim - keep_dims, 1)
        reduce_axis = tuple(range(reduce_dims))
        mean = values.mean(axis=reduce_axis, keepdims=True)
        std = values.std(axis=reduce_axis, keepdims=True) + 1e-8
        standardized_data[feat] = (values - mean) / std
    return standardized_data


class MyCrossValidation:
    def __init__(self, args):
        self.args = args
        
        self.datapipe = create_datapipe(args)
        self.online_transform = args.dataset.online_transform

        self.save_dir = Path(args.training.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir = Path(args.training.records_dir)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir = Path(args.training.tb_dir)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        self.trainer = ModelTrainer(args)
        self.monitor = args.training.monitor
        self.criterion = self.create_criterion()
        self.optimizer_type = args.training.optimizer

        self.lr = args.training.learning_rate
        self.batch_size = args.training.batch_size

    def ensure_data_prepared(self):
        if not self.datapipe.is_prepared():
            logger.warning("Data cache missing or config changed, rebuilding data at {} ...", self.datapipe.data_dir)
            self.datapipe.build()

    def _iter_folds(self, subjects=None):
        all_subjects = sorted(self.datapipe.sub_ids)
        subjects = all_subjects if subjects is None else subjects
        if len(set(subjects)) != len(subjects) or not set(subjects) <= set(all_subjects):
            raise ValueError("Requested folds must be unique valid subject IDs")
        after_load = lambda f, y: (sub_internal_standardize(f) if self.online_transform == "sub_internal" else f, y)
        if self.args.training.selection_protocol == "legacy_test":
            for sid, data in self.datapipe.iter_loso(after_load=after_load):
                if sid in subjects:
                    yield sid, None, [s for s in all_subjects if s != sid], data, None
            return
        if self.online_transform == "sub_internal":
            raise ValueError("Strict evaluation requires train-fitted normalization or none; use index-strict.yaml")
        offset = self.args.training.validation_subject_offset % len(all_subjects)
        if offset == 0 or len(all_subjects) < 3:
            raise ValueError("Validation offset must select a different subject, with at least 3 subjects")
        for sid in subjects:
            val_sid = all_subjects[(all_subjects.index(sid) + offset) % len(all_subjects)]
            train_ids = [s for s in all_subjects if s not in {sid, val_sid}]
            train, train_y = self.datapipe.load_subs(sub_ids=train_ids)
            val = self.datapipe.load_one_sub(val_sid)
            test, test_y = self.datapipe.load_one_sub(sid)
            yield sid, val_sid, train_ids, (train, train_y, test, test_y), val
    
    def leave_one_sub_out(self, subjects=None):
        """被试留一交叉验证"""
        subjects = sorted(self.datapipe.sub_ids) if subjects is None else subjects
        for sid in subjects:
            if (self.records_dir / f"loso_sub_{sid}_fold.json").exists():
                raise FileExistsError(f"Fold {sid} already exists; use a new training.exp_name to run it again")
        resolved = json.loads(json.dumps(asdict(self.args), default=str))
        config_path = self.records_dir / "resolved_config.json"
        if config_path.exists():
            with config_path.open(encoding="utf-8") as f:
                if json.load(f) != resolved:
                    raise ValueError("This exp_name already has a different configuration; choose a new exp_name")
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            MofNCurrentColumn(), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            console=console,
        )
        progress.start()
        num_folds = len(subjects) if subjects is not None else self.datapipe.num_subs
        task = progress.add_task(f"LOSO Fold", total=num_folds)

        # 记录每个被试最佳监控值对应的评估指标
        sub_metrics: list[tuple[int, EvaluateMeter]] = []

        # Save the resolved config, not just an inherited YAML fragment.
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(resolved, f, indent=2)
        if self.args.training.selection_protocol == "legacy_test":
            logger.warning("Legacy selection: held-out test accuracy chooses the epoch; this is not an independent test estimate.")

        for fold_id, (sub_id, val_sid, train_ids, data, val) in enumerate(self._iter_folds(subjects), start=1):
            logger.info('-' * 20 + f" Leave Subject {sub_id} Out " + '-' * 20)
            fold_seed = self.args.reproduce.random_seed + sub_id
            seed_everything(fold_seed, self.args.reproduce.deterministic)

            processed = self.preprocess(*data, validation=val)
            train_data, train_label, test_data, test_label = processed[:4]
            if val is None:
                val_data, val_label = test_data, test_label
            else:
                val_data, val_label = processed[4:]
            norm_path = self.records_dir / f"loso_sub_{sub_id}_normalization.npz"
            np.savez(norm_path, **self.normalization_state)

            self.show_shape(train_data, train_label, test_data, test_label)

            train_loader = self.get_dataloader(train_data, train_label)
            val_loader = self.get_dataloader(val_data, val_label, shuffle=False)
            test_loader = self.get_dataloader(test_data, test_label, shuffle=False)

            model = create_model(self.args)
            optimizer = self.create_optimizer(model)

            tb_writer = SummaryWriter(log_dir=str(self.tb_dir / f"loso_sub_{sub_id}"))
            save_file = str(self.save_dir / f"loso_sub_{sub_id}_epoch_{{epoch}}_{self.monitor}_{{{self.monitor}:.5f}}.pth")

            records = self.trainer.train(
                model, train_loader, val_loader, self.criterion, optimizer,
                save_filepath=save_file, tb_writer=tb_writer,
            )
            # Restore exactly the model selected on the designated selection set.
            checkpoint = torch.load(records.best_model_file, map_location=self.args.training.device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            # Strict protocol touches test metrics only here, after model selection.
            _, test_metrics = self.trainer.eval_one_epoch(model, test_loader, self.criterion)
            sub_metrics.append((sub_id, test_metrics))
            np.savez_compressed(
                self.records_dir / f"loso_sub_{sub_id}_predictions.npz",
                trues=np.asarray(test_metrics.trues), preds=np.asarray(test_metrics.preds),
                probs=test_metrics.probs,
            )
            fold_metadata = {
                "subject_id": sub_id, "validation_subject_id": val_sid,
                "training_subject_ids": train_ids, "seed": fold_seed,
                "selection_protocol": self.args.training.selection_protocol,
                "normalization": self.online_transform,
                "normalization_file": norm_path.name,
                "train_samples": len(train_label), "validation_samples": len(val_label),
                "test_samples": len(test_label),
                "checkpoint": str(Path(records.best_model_file).resolve().relative_to(self.save_dir.resolve())),
                "accuracy": test_metrics.accuracy,
                "macro_f1": test_metrics.f1_score(), "best_epoch": records.best_epoch,
            }
            with (self.records_dir / f"loso_sub_{sub_id}_fold.json").open("w", encoding="utf-8") as f:
                json.dump(fold_metadata, f, indent=2)

            logger.info(
                "\U0001F449 LOSO Fold {}/{}, Sub {}: Best {}: {:.5f} at epoch {}",
                fold_id, num_folds, sub_id, self.monitor, records.best_value, records.best_epoch
            )

            tb_writer.close()
            self.save_records(records, f"loso_sub_{sub_id}_records.json")
            del model, optimizer, train_loader, val_loader, test_loader
            progress.advance(task)
        
        progress.stop()
        # A run may add previously uncompleted folds under the same config.
        # Aggregate all completed subjects, not just this invocation's subset.
        completed_ids = {sid for sid, _ in sub_metrics}
        for sid in sorted(self.datapipe.sub_ids):
            if sid in completed_ids or not (self.records_dir / f"loso_sub_{sid}_fold.json").exists():
                continue
            with np.load(self.records_dir / f"loso_sub_{sid}_predictions.npz", allow_pickle=False) as preds:
                sub_metrics.append((sid, EvaluateMeter.from_predictions(preds["trues"], preds["preds"], preds["probs"])))
        sub_metrics.sort(key=lambda item: item[0])
        self.handle_sub_metrics(sub_metrics)

    def save_records(self, records, filename):
        """保存训练记录到文件"""
        filepath = self.records_dir / filename
        records_dict = records.as_dict()
        with open(filepath, 'w', encoding="utf-8") as f:
            json.dump(records_dict, f, indent=2)
        logger.info("Training records saved to {}", filepath)
    
    def create_criterion(self):
        if self.args.training.loss_weight is not None:
            weight_tensor = torch.tensor(self.args.training.loss_weight, dtype=torch.float, device=self.args.training.device)
            criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=self.args.training.label_smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.args.training.label_smoothing)
        return criterion
    
    def create_optimizer(self, model):
        if self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        elif self.optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(
                build_adamw_param_groups(model, self.args.training.wd, self.args.training.wd_head),
                lr=self.lr
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        return optimizer

    def get_dataloader(self, data, label, shuffle=True, **kwargs):
        dataset = MyEEGDataset(data, label)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=self.args.training.device.type == "cuda",
            **kwargs,
        )
        return dataloader

    def preprocess(
        self, train_data: dict[str, np.ndarray], train_label: np.ndarray, test_data: dict[str, np.ndarray], test_label: np.ndarray,
        validation=None,
    ):
        """数据预处理"""
        self.normalization_state = {}
        val_data, val_label = validation if validation is not None else (None, None)
        for feat in self.args.model.features:
            if feat not in ("de", "de_mov", "psd", "r_psd", "raw", "wavelet") and not feat.startswith("cog_"):
                continue
            if self.online_transform == "channel_wise":
                standardizer = ChannelWiseStandardizer(
                    # 认知特征, 对最后一位不同子特征分别标准化
                    # 其余特征, 对倒数第二位通道分别标准化
                    axis=-1 if feat == "wavelet" or feat.startswith("cog_") else -2
                )
            elif self.online_transform == "zscore":
                standardizer = ZScoreStandardizer()
            else:
                continue
            train_data[feat] = standardizer.fit_transform(train_data[feat])
            test_data[feat] = standardizer.transform(test_data[feat])
            if val_data is not None:
                val_data[feat] = standardizer.transform(val_data[feat])
            self.normalization_state[f"{feat}__mean"] = standardizer.mean
            self.normalization_state[f"{feat}__std"] = np.maximum(standardizer.std, standardizer.eps)

        train_data, test_data = numpy2tensor(train_data, test_data, dtype=torch.float32)
        train_label, test_label = numpy2tensor(train_label, test_label, dtype=torch.long)

        train_data = tuple(train_data[f] for f in self.args.model.features)
        test_data = tuple(test_data[f] for f in self.args.model.features)

        if len(train_data) == 1:
            train_data = train_data[0]
            test_data = test_data[0]

        result = (train_data, train_label, test_data, test_label)
        if val_data is not None:
            val_data = tuple(numpy2tensor(val_data[f], dtype=torch.float32) for f in self.args.model.features)
            if len(val_data) == 1:
                val_data = val_data[0]
            result += (val_data, numpy2tensor(val_label, dtype=torch.long))
        return result
    
    def show_shape(self, train_data, train_label, test_data, test_label):
        """显示数据形状"""
        if isinstance(train_data, tuple):
            train_shape = ', '.join(str(tuple(d.shape)) for d in train_data)
        else:
            train_shape = str(tuple(train_data.shape))
        logger.debug("Train shape, data: {}, label: {}", train_shape, tuple(train_label.shape))

        if isinstance(test_data, tuple):
            test_shape = ', '.join(str(tuple(d.shape)) for d in test_data)
        else:
            test_shape = str(tuple(test_data.shape))
        logger.debug("Test shape, data: {}, label: {}", test_shape, tuple(test_label.shape))
    
    def evaluate(self, model_path=None, output_dir=None, subjects=None):
        """Re-evaluate each subject using its own fold checkpoint and normalization.

        model_path, when supplied, is a checkpoint directory. A single model
        cannot represent LOSO, since it may have trained on other test subjects.
        """
        checkpoint_dir = Path(model_path) if model_path is not None else self.save_dir
        if not checkpoint_dir.is_dir():
            raise ValueError("LOSO evaluation requires a checkpoint directory, not one shared model")
        output_path = Path(output_dir) if output_dir is not None else self.records_dir / "evaluation"
        output_path.mkdir(parents=True, exist_ok=True)
        with (self.records_dir / "resolved_config.json").open(encoding="utf-8") as f:
            saved_config = json.load(f)
        current = asdict(self.args)
        if saved_config["model"] != current["model"]:
            raise ValueError("Model configuration differs from the saved run")
        for key in ("selection_protocol", "dropout"):
            if saved_config["training"][key] != current["training"][key]:
                raise ValueError(f"Training setting {key} differs from the saved run")
        for key in ("name", "graph_type", "downsample", "segment", "segment_step", "sequence",
                    "sequence_step", "freq_bands", "features", "online_transform", "session_to_load", "cog_params"):
            # JSON converts tuples to lists.
            if saved_config["dataset"][key] != json.loads(json.dumps(current["dataset"][key])):
                raise ValueError(f"Dataset setting {key} differs from the saved run")
        subjects = sorted(self.datapipe.sub_ids) if subjects is None else subjects
        if len(set(subjects)) != len(subjects) or not set(subjects) <= set(self.datapipe.sub_ids):
            raise ValueError("Requested folds must be unique valid subject IDs")
        sub_metrics = []
        for sid in subjects:
            with (self.records_dir / f"loso_sub_{sid}_fold.json").open(encoding="utf-8") as f:
                fold = json.load(f)
            if fold["subject_id"] != sid:
                raise ValueError("Fold manifest subject mismatch")
            model = create_model(self.args).to(self.args.training.device)
            checkpoint = torch.load(checkpoint_dir / fold["checkpoint"],
                                    map_location=self.args.training.device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            data, label = self.datapipe.load_one_sub(sid)
            if self.online_transform == "sub_internal":
                data = sub_internal_standardize(data)
            else:
                with np.load(self.records_dir / fold["normalization_file"], allow_pickle=False) as norm:
                    for feat in self.args.model.features:
                        mean_key, std_key = f"{feat}__mean", f"{feat}__std"
                        if self.online_transform in {"channel_wise", "zscore"} and mean_key not in norm:
                            raise ValueError(f"Missing training normalization for {feat}")
                        if mean_key in norm:
                            data[feat] = (data[feat] - norm[mean_key]) / norm[std_key]
            tensors = tuple(numpy2tensor(data[f], dtype=torch.float32) for f in self.args.model.features)
            tensors = tensors[0] if len(tensors) == 1 else tensors
            loader = self.get_dataloader(tensors, numpy2tensor(label, dtype=torch.long), shuffle=False)
            _, metrics = self.trainer.eval_one_epoch(model, loader, self.criterion)
            sub_metrics.append((sid, metrics))
            np.savez_compressed(output_path / f"loso_sub_{sid}_predictions.npz",
                                trues=np.asarray(metrics.trues), preds=np.asarray(metrics.preds), probs=metrics.probs)
        self.handle_sub_metrics(sub_metrics, output_dir=output_path, tag="Evaluation")
        return sub_metrics

    def handle_sub_metrics(
        self,
        sub_metrics: list[tuple[int, EvaluateMeter]],
        output_dir: Path | str | None = None,
        tag: str = "LOSO",
    ):
        """处理每个被试的评估指标并输出可视化图表

        Args:
            sub_metrics: 每个被试的 (sub_id, EvaluateMeter) 列表
            output_dir: 图表输出目录，默认使用 self.records_dir
            tag: 标识字符串，用于图表标题和文件名前缀如("LOSO"、"Evaluation")
        """
        output_path = Path(output_dir) if output_dir is not None else self.records_dir
        file_prefix = tag.lower()

        sub_id, acc, f1 = [], [], []
        all_trues, all_preds = [], []
        all_probs = []

        for sid, metrics in sub_metrics:
            sub_id.append(sid)
            acc.append(metrics.accuracy)
            f1.append(metrics.f1_score())
            all_trues.extend(metrics.trues)
            all_preds.extend(metrics.preds)
            if metrics.probs is not None:
                all_probs.append(metrics.probs)

        # 合并所有被试的概率输出
        all_probs = np.concatenate(all_probs, axis=0) if all_probs else None

        # 输出每个被试的评估结果
        logger.info("Sub ID\t{}", '\t'.join(str(sid) for sid in sub_id))
        logger.info("Acc   \t{}", '\t'.join(f"{a:.5f}" for a in acc))
        logger.info("F1    \t{}", '\t'.join(f"{f:.5f}" for f in f1))
        logger.info("mAcc {:.5f}, std {:.5f}", np.mean(acc), np.std(acc))
        logger.info("mF1  {:.5f}, std {:.5f}", np.mean(f1), np.std(f1))
        summary = {
            "selection_protocol": self.args.training.selection_protocol,
            "subjects": sub_id, "accuracy": acc, "macro_f1": f1,
            "accuracy_mean": float(np.mean(acc)), "accuracy_std": float(np.std(acc, ddof=0)),
            "macro_f1_mean": float(np.mean(f1)), "macro_f1_std": float(np.std(f1, ddof=0)),
            "std_ddof": 0, "units": "fraction",
            "complete_loso": len(sub_id) == self.datapipe.num_subs,
        }
        with (output_path / f"{file_prefix}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 绘制所有被试的混淆矩阵
        if all_trues and all_preds:
            cm = confusion_matrix(all_trues, all_preds, labels=list(range(self.args.model.num_classes)), normalize='true')
            np.savez_compressed(output_path / f"{file_prefix}_pooled_predictions.npz",
                                trues=all_trues, preds=all_preds, probs=all_probs,
                                confusion_matrix=cm)
            fig = plot_confusion_matrix(
                cm,
                class_names=self.args.model.class_names,
                title=f"Confusion Matrix (All Subjects, {tag})",
            )
            fig_path = output_path / f"{file_prefix}_all_sub_confusion_matrix.svg"
            fig.savefig(str(fig_path), format="svg", bbox_inches="tight")
            plt.close(fig)
            logger.info("Confusion matrix saved to {}", fig_path)

        # 计算并绘制ROC曲线
        if all_probs is not None:
            aucs = {}
            for k, name in enumerate(self.args.model.class_names):
                binary = np.asarray(all_trues) == k
                aucs[name] = float(roc_auc_score(binary, all_probs[:, k])) if np.unique(binary).size == 2 else None
            with (output_path / f"{file_prefix}_class_auc.json").open("w", encoding="utf-8") as f:
                json.dump(aucs, f, indent=2)
            fig = plot_roc_curve(
                all_trues,
                all_probs,
                class_names=self.args.model.class_names,
                title=f"ROC Curve (All Subjects, {tag})",
            )
            fig_path = output_path / f"{file_prefix}_all_sub_roc_curve.svg"
            fig.savefig(str(fig_path), format="svg", bbox_inches="tight")
            plt.close(fig)
            logger.info("ROC curve saved to {}", fig_path)
