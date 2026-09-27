import argparse
import ast
import logging
import os
from datetime import datetime
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Callable, Iterable, Optional, Type, Union

import numpy as np
import scipy
import seaborn as sns
import torch
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import ProgressColumn, Task
from rich.style import Style
from rich.table import Column
from rich.text import Text
from rich.theme import Theme
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

import sys
plt.rcParams['font.family'] = ['DejaVu Sans']

# shared console instance
console = Console(
    theme=Theme({
        "logging.level.debug": "white",
        "logging.level.success": "green"
    })
)

def _log_exc_hook(typ: Type[BaseException], val: BaseException, tb: Optional[TracebackType]):
    logger.opt(exception=(typ, val, tb)).error("Exception eccored:")


def install_logger(
    level: Union[int, str] = logging.NOTSET,
    time_format: Union[str, Callable[[datetime], str]] = "[%x %X]",
    hook_exceptions: bool = True,
    rich_tracebacks: bool = True,
    tracebacks_suppress: Iterable[Union[str, ModuleType]] = (),
    tracebacks_show_locals: bool = False,
):
    # remove default logger
    logger.remove()
    logger.add(
        RichHandler(
            console=console,
            log_time_format=time_format,
            rich_tracebacks=rich_tracebacks,
            tracebacks_suppress=tracebacks_suppress,
            tracebacks_show_locals=tracebacks_show_locals
        ),
        format=(lambda _ : "{message}") if rich_tracebacks else "{message}",
        backtrace=not rich_tracebacks,
        level=level
    )

    if hook_exceptions:
        import sys
        sys.excepthook = _log_exc_hook


def move2deivce(
    *args: Union[torch.Tensor, dict[str, torch.Tensor]],
    device: torch.device
):
    out = []
    for data in args:
        if isinstance(data, torch.Tensor):
            out.append(data.to(device))
        elif isinstance(data, (list, tuple)):
            out.append(type(data)(move2deivce(item, device=device) for item in data))
        elif isinstance(data, dict):
            out.append({k: move2deivce(v, device=device) for k, v in data.items()})
        else:
            out.append(data)
    return tuple(out) if len(out) > 1 else out[0]


def seed_everything(seed: int, deterministic: bool = False):
    """设置所有随机种子以确保实验可复现"""
    import os
    import random
    import numpy as np

    # 固定Python哈希函数的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 固定Python内置random模块的随机种子
    random.seed(seed)
    # 固定NumPy的随机种子
    np.random.seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # 固定PyTorch的随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        # current gpu
        torch.cuda.manual_seed_all(seed)    # all gpus
    # 使用确定性算法, 可能会降低性能
    if deterministic:
        # 强制cuDNN使用确定性算法
        torch.backends.cudnn.deterministic = True
        # 禁用cuDNN自动调优
        torch.backends.cudnn.benchmark = False
        # 强制确定性算子, 当使用不支持确定性的算子时抛出异常
        torch.use_deterministic_algorithms(True)


def get_device(args):
    if args.device is not None:
        return torch.device(args.device)
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_adamw_param_groups(model: torch.nn.Module, wd: float = 1e-3, wd_head: float = 3e-3):
    """为AdamW优化器构建参数组, 对分类头使用不同的权重衰减"""
    decay, head_deacy, no_decay = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_bias = name.endswith(".bias")
        is_norm = any(k in name.lower() for k in ["norm", "bn", "ln", "gn"])
        is_head = any(k in name.lower() for k in ["classifier", "fc", "output", "out"])
        if is_bias or is_norm:
            no_decay.append(param)
        elif is_head:
            head_deacy.append(param)
        else:
            decay.append(param)
    
    param_groups = []
    if decay:
        param_groups.append({"params": decay, "weight_decay": wd})
    if head_deacy:
        param_groups.append({"params": head_deacy, "weight_decay": wd_head})
    if no_decay:
        param_groups.append({"params": no_decay, "weight_decay": 0.0})
    
    return param_groups

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    figsize: tuple[int, int] = (8, 6),
    class_names: Optional[list[str]] = None,
    callback: Optional[Callable[[Figure], Any]] = None,
    *,
    xlabel: str = 'Predicted Label',
    ylabel: str = 'True Label',
    title: str = 'Confusion Matrix',
    annot=True,
    cmap='Blues',
    fmt='.2g',
):
    """
    绘制混淆矩阵热力图
    
    Args:
        confusion_matrix (np.ndarray): 混淆矩阵数据, shape=(N_classes, N_classes)
        figsize (tuple[int, int]): 图像大小
        callback (Callable[[Figure], Any], optional): 图像创建后的回调函数, 如果提供则调用该函数并传入图像对象, 否则返回图像对象
    Returns:
        Optional[Figure]: 如果未提供 callback, 则返回 Figure 对象, 请注意使用后调用 plt.close(fig) 以避免内存泄漏
    """
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    hm = sns.heatmap(
        confusion_matrix,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        cbar=True,
        ax=ax,
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
    )

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # 坐标轴刻度线加边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    # 颜色条加边框
    cbar = hm.collections[0].colorbar
    cbar.outline.set_linewidth(1)
    cbar.outline.set_edgecolor('black')

    if callback is not None:
        callback(fig)
        plt.close(fig)
    else:
        # note: you need to call plt.close(fig) after using the returned fig to avoid memory leak
        return fig

def plot_roc_curve(
    y_trues: list | np.ndarray,
    y_probs: np.ndarray,
    class_names: Optional[list[str]] = None,
    figsize: tuple[int, int] = (8, 6),
    callback: Optional[Callable[[Figure], Any]] = None,
    *,
    xlabel: str = 'False Positive Rate',
    ylabel: str = 'True Positive Rate',
    title: str = 'ROC Curve',
) -> Optional[Figure]:
    """
    绘制多分类 ROC 曲线

    Args:
        y_trues (list | np.ndarray): 真实标签(非 one-hot), shape=(N,)
        y_probs (np.ndarray): 各类别概率输出, shape=(N, n_classes)
        class_names (list[str], optional): 类别名称列表，长度须等于 n_classes
        figsize (tuple[int, int]): 图像大小
        callback (Callable[[Figure], Any], optional): 图像创建后的回调函数，
            若提供则调用后关闭图像；否则返回 Figure 对象
        xlabel (str): x 轴标签
        ylabel (str): y 轴标签
        title (str): 图像标题

    Returns:
        Optional[Figure]: 未提供 callback 时返回 Figure 对象，
            请注意使用后调用 plt.close(fig) 以避免内存泄漏
    """
    y_trues = np.asarray(y_trues)
    n_classes = y_probs.shape[1]
    y_trues_bin = label_binarize(y_trues, classes=range(n_classes))

    # sklearn 在二分类时返回 (N, 1)，需补全为 (N, 2)
    if n_classes == 2 and y_trues_bin.shape[1] == 1:
        y_trues_bin = np.hstack([1 - y_trues_bin, y_trues_bin])

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_trues_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")

    if callback is not None:
        callback(fig)
        plt.close(fig)
    else:
        return fig


class AverageMeter:
    def __init__(self):
        self.count = 0
        self.sum = 0.0

    def update(self, value: float | torch.Tensor, n: int = 1):
        """
        更新统计量

        Args:
            value (float | torch.Tensor): 要累加的值
            n (int): 值的样本数量
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.sum += value * n
        self.count += n
    
    @property
    def avg(self) -> float:
        """获取当前均值"""
        if self.count == 0:
            return 0.0
        return self.sum / self.count


class EvaluateMeter:
    def __init__(self, save_logits: bool = False):
        self.save_logits = save_logits
        self.num_classes = None
        self._cached_probs = None
        
        self.count = 0
        self.correct = 0

        self._logits: list[np.ndarray] = [] if save_logits else None # 每个批次未归一化的概率输出
        self.preds: list[int] = []  # 每个样本的预测标签
        self.trues: list[int] = []  # 每个样本的真实标签
    
    def update(self, logits: torch.Tensor, trues: torch.Tensor):
        """
        更新分类指标

        Args:
            logits (torch.Tensor): 模型输出的logits, shape=(Batch_sizes, N_classes)
            trues (torch.Tensor): 真实标签, shape=(Batch_sizes,)
        """
        preds = torch.argmax(logits, dim=1)
        self.num_classes = logits.shape[1]
        self.count += trues.size(0)
        self.correct += (preds == trues).sum().item()

        if self.save_logits:
            self._logits.append(logits.detach().cpu().numpy())
        self.preds.extend(preds.detach().cpu().tolist())
        self.trues.extend(trues.detach().cpu().tolist())

    @property
    def logits(self) -> np.ndarray:
        """未归一化的logits输出"""
        return np.concatenate(self._logits, axis=0) if self.save_logits else None
     
    @property
    def probs(self) -> np.ndarray:
        """归一化后的类别概率"""
        if self._cached_probs is not None:
            return self._cached_probs
        return scipy.special.softmax(self.logits, axis=1) if self.save_logits else None

    @classmethod
    def from_predictions(cls, trues, preds, probs):
        """Reconstruct a completed fold without evaluating a model again."""
        meter = cls()
        meter.trues = np.asarray(trues).tolist()
        meter.preds = np.asarray(preds).tolist()
        meter._cached_probs = np.asarray(probs)
        if len(meter.trues) != len(meter.preds) or meter._cached_probs.shape[0] != len(meter.trues):
            raise ValueError("Saved prediction lengths do not match")
        meter.num_classes = meter._cached_probs.shape[1]
        meter.count = len(meter.trues)
        meter.correct = int(np.sum(np.asarray(trues) == np.asarray(preds)))
        return meter
    
    @property
    def accuracy(self) -> float:
        """准确率"""
        if self.count == 0:
            return 0.0
        return self.correct / self.count
    
    def precision(self, average = 'macro', **kwargs):
        """精确率"""
        return precision_score(self.trues, self.preds, average=average, **kwargs)
    
    def recall(self, average = 'macro', **kwargs):
        """召回率"""
        return recall_score(self.trues, self.preds, average=average, **kwargs)
    
    def f1_score(self, average = 'macro', **kwargs):
        """F1分数"""
        kwargs.setdefault("zero_division", 0)
        if self.num_classes is not None:
            kwargs.setdefault("labels", list(range(self.num_classes)))
        return f1_score(self.trues, self.preds, average=average, **kwargs)
    
    def confusion_matrix(self, normalize: Optional[str] = 'true', **kwargs):
        """混淆矩阵"""
        if self.num_classes is not None:
            kwargs.setdefault("labels", list(range(self.num_classes)))
        return confusion_matrix(self.trues, self.preds, normalize=normalize, **kwargs)
    
    def classification_report(self, target_names: Optional[list[str]] = None, **kwargs):
        """分类报告"""
        return classification_report(self.trues, self.preds, target_names=target_names, **kwargs)


class ParseFreqBands(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        bands = {}

        for item in values:
            if "=" not in item:
                raise argparse.ArgumentError(
                    self, f"Invalid format '{item}', expected band=(low, high)"
                )

            key, val = item.split("=", 1)

            try:
                # 安全解析 "(l, h)" → tuple
                freq = ast.literal_eval(val)
            except Exception:
                raise argparse.ArgumentError(
                    self, f"Invalid tuple format: {val}"
                )

            if (
                not isinstance(freq, tuple)
                or len(freq) != 2
                or not all(isinstance(x, (int, float)) for x in freq)
            ):
                raise argparse.ArgumentError(
                    self, f"Frequency band must be a tuple of two numbers: {val}"
                )

            bands[key] = tuple(freq)

        setattr(namespace, self.dest, bands)


class TimeColumn(ProgressColumn):
    """显示已用时间和总时间的进度列"""

    def __init__(
        self,
        compact: bool = True,
        sep: str = "<",
        table_column: Optional[Column] = None
    ):
        self.compact = compact
        self.sep = sep
        super().__init__(table_column=table_column)

    def _render_time(self, time_seconds: float, style: Union[str, Style] = "") -> Text:
        if time_seconds is None:
            return Text("--:--" if self.compact else "--:--:--", style=style)
        
        minutes, seconds = divmod(int(time_seconds), 60)
        hours, minutes = divmod(minutes, 60)

        if self.compact and not hours:
            formatted = f"{minutes:02d}:{seconds:02d}"
        else:
            formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        return Text(formatted, style=style)

    def render(self, task: Task) -> Text:
        if task.total is None:
            return Text("")
        
        elpased_time = task.finished_time if task.finished else task.elapsed
        remaining_time = task.time_remaining

        elapsed_text = self._render_time(elpased_time, style="progress.elapsed")
        remaining_text = self._render_time(remaining_time, style="progress.remaining")

        return Text.assemble(elapsed_text, self.sep, remaining_text)
    

class ProgressSpeedColumn(ProgressColumn):
    """显示处理速度的进度列"""
    def __init__(
        self,
        unit: str = "it",
        table_column: Optional[Column] = None
    ):
        self.unit = unit
        super().__init__(table_column=table_column)

    def render(self, task: Task) -> Text:
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text(f"?{self.unit}/s", style="progress.data.speed")
        if 0 < speed < 1.0:
            formatted = f"{1/speed:.2f}s/{self.unit}"
        else:
            formatted = f"{speed:.2f}{self.unit}/s"
        return Text(formatted, style="progress.data.speed")


class MofNCurrentColumn(ProgressColumn):
    """显示当前进度为 M of N 的进度列"""
    def __init__(
        self,
        sep: str = "/",
        table_column: Optional[Column] = None
    ):
        self.sep = sep
        super().__init__(table_column=table_column)

    def render(self, task: Task) -> Text:
        current = int(task.completed) + (1 if not task.finished else 0)
        total = int(task.total) if task.total is not None else "?"
        total_width = len(str(total))
        return Text(
            f"{current:{total_width}d}{self.sep}{total}",
            style="progress.download"
        )


class ModelCheckpoint:
    """
    模型检查点管理器

    Args:
        filepath (str): 模型保存路径, 可包含格式化字段如 '{epoch:02d}-{val_accuracy:.2f}'
        monitor (str): 监控的指标名称, 如 'val_accuracy'
        mode (str): 'min', 'max' 或 'auto', 指定监控指标的优化方向
        save_best_n (int): 保存最佳模型的数量
        save_last_k (int): 保存最后 k 个 epoch 的模型数量
        save_weights_only (bool): 是否只保存模型权重而不包含优化器等其他信息
    
    Examples:
        >>> checkpoint = ModelCheckpoint(
        ...     filepath='checkpoints/model-{epoch:02d}-{val_accuracy:.3f}.pth',
        ...     monitor='val_accuracy',
        ...     mode='max',
        ...     save_best_n=3,
        ...     save_last_k=2
        ... )
        >>> # 在训练循环中使用
        >>> checkpoint.step(model, epoch=10, val_accuracy=0.95, optimizer=optimizer)
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = "val_accuracy",
        mode: str = "auto",
        save_best_n: int = 1,
        save_last_k: int = 1,
        save_weights_only: bool = False,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_n = save_best_n
        self.save_last_k = save_last_k
        self.save_weights_only = save_weights_only

        # 创建保存目录
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if mode not in {"min", "max", "auto"}:
            raise ValueError(f"Invalid mode: {mode}. Expected one of 'min', 'max', 'auto'.")
        if mode == "auto":
            self.mode = "min" if "loss" in monitor else "max"

        if self.mode == "min":
            self.monitor_op = lambda a, b: a < b
            self.best_values = [float("inf")] * save_best_n
        elif self.mode == "max":
            self.monitor_op = lambda a, b: a > b
            self.best_values = [float("-inf")] * save_best_n

        # 用于跟踪已保存的文件
        self.best_model_paths: list[tuple[float, str]] = []  # [(metric_value, filepath), ...]
        self.last_model_paths: list[str] = []  # [filepath, ...]
    
    def step(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **metrics
    ) -> Optional[str]:
        """
        检查是否需要保存模型，并根据配置保存检查点

        Args:
            model (torch.nn.Module): 要保存的模型
            optimizer (torch.optim.Optimizer, optional): 优化器, 如果 save_weights_only=False 则一起保存
            **metrics: 当前的指标值, 必须包含 monitor 指定的指标, 可用于文件名格式化

        Returns:
            Optional[str]: 如果保存了模型，返回保存的路径；否则返回 None
        
        Examples:
            >>> checkpoint.step(model, epoch=1, val_accuracy=0.85, val_loss=0.3)
        """
        if self.monitor not in metrics:
            logger.error(
                "Monitior metric {} not found in metrics: {}",
                self.monitor, list(metrics.keys())
            )
            return None

        current_value = metrics[self.monitor]

        saved_path = self._save_checkpoint(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            current_value=current_value,
            save_to_best=self._should_save_best(current_value),
            save_to_last=self.save_last_k > 0
        )

        return saved_path
    
    def _should_save_best(self, current_value: float) -> bool:
        """判断当前值是否应该保存到最佳模型"""
        save_best_count = len(self.best_model_paths)
        if save_best_count < self.save_best_n:
            return True
        elif save_best_count > 0:        
            worst_best_value = self.best_model_paths[-1][0]
            return self.monitor_op(current_value, worst_best_value)
        else:
            return False
    
    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        metrics: dict,
        current_value: float,
        save_to_best: bool,
        save_to_last: bool
    ) -> Optional[str]:
        if not save_to_best and not save_to_last:
            return None

        """保存检查点文件"""
        # 格式化文件路径
        try:
            filepath = self.filepath.format(**metrics)
        except KeyError as e:
            logger.error(
                "Failed to format filepath '{}': {}. Available metrics: {}",
                self.filepath, e, list(metrics.keys())
            )
            # 回退到不格式化的默认路径
            filepath = Path(self.filepath).parent / "checkpoint.pth"
        
        # 确保唯一性: 如果文件已存在, 添加时间戳
        if os.path.exists(filepath):
            p = Path(filepath)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(p.parent / f"{p.stem}_{timestamp}{p.suffix}")
        
        # 准备保存内容
        if self.save_weights_only:
            checkpoint = model.state_dict()
        else:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
            }
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # 保存文件
        torch.save(checkpoint, filepath)
        logger.info("Saved checkpoint to {}", filepath)
        
        # 更新跟踪列表
        if save_to_best:
            self._update_best_models(current_value, filepath)
        if save_to_last:
            self._update_last_models(filepath)
        
        return filepath
    
    def _update_best_models(self, current_value: float, filepath: str):
        """更新最佳模型列表"""
        # 添加新模型
        self.best_model_paths.append((current_value, filepath))
        
        # 按照监控指标排序（最好的在前）
        self.best_model_paths.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        
        # 删除超出限制的模型
        while len(self.best_model_paths) > self.save_best_n:
            _, old_path = self.best_model_paths.pop()
            # 确保不删除最近k个模型
            if os.path.exists(old_path) and old_path not in self.last_model_paths:
                try:
                    os.remove(old_path)
                    logger.debug("Removed old checkpoint: {}", old_path)
                except Exception as e:
                    logger.warning("Failed to remove {}: {}", old_path, e)
    
    def _update_last_models(self, filepath: str):
        """更新最近k个模型列表"""        
        self.last_model_paths.append(filepath)
        
        # 删除超出限制的模型
        while len(self.last_model_paths) > self.save_last_k:
            old_path = self.last_model_paths.pop(0)
            # 确保不删除best模型
            if os.path.exists(old_path) and old_path not in [p for _, p in self.best_model_paths]:
                try:
                    os.remove(old_path)
                    logger.debug("Removed old checkpoint: {}", old_path)
                except Exception as e:
                    logger.warning("Failed to remove {}: {}", old_path, e)
    
    def get_best_model_path(self, rank: int = 0) -> Optional[str]:
        """
        获取第 rank 好的模型路径

        Args:
            rank (int): 排名, 0 表示最好的模型, 1 表示第二好, 以此类推

        Returns:
            Optional[str]: 模型路径，如果不存在则返回 None
        """
        if 0 <= rank < len(self.best_model_paths):
            return self.best_model_paths[rank][1]
        return None
    
    def load_checkpoint(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> dict:
        """
        加载检查点

        Args:
            filepath (str): 检查点文件路径
            model (torch.nn.Module): 要加载权重的模型
            optimizer (torch.optim.Optimizer, optional): 要加载状态的优化器
            device (torch.device, optional): 加载到的设备

        Returns:
            dict: 检查点中的 metrics 信息
        
        Examples:
            >>> checkpoint = ModelCheckpoint('checkpoints/model.pth')
            >>> metrics = checkpoint.load_checkpoint('checkpoints/model.pth', model, optimizer)
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            metrics = checkpoint.get('metrics', {})
        else:
            # 仅包含权重的检查点
            model.load_state_dict(checkpoint)
            metrics = {}
        
        logger.info("Loaded checkpoint from {}", filepath)
        return metrics
