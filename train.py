from typing import Any, Callable, Optional, Union

import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端以避免显示图形窗口
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from utils import console, logger
from utils.tool import AverageMeter, EvaluateMeter, ModelCheckpoint, move2deivce, ProgressSpeedColumn, TimeColumn, plot_confusion_matrix


class TrainingRecords:
    def __init__(self, monitor: str = "val_acc", mode: str = "auto"):
        self.train_losses = []      # 每个 epoch 的训练损失
        self.train_accuracies = []  # 每个 epoch 的训练准确率
        self.train_f1_scores = []   # 每个 epoch 的训练 F1 分数
        self.val_losses = []        # 每个 epoch 的验证损失
        self.val_accuracies = []    # 每个 epoch 的验证准确率
        self.val_f1_scores = []     # 每个 epoch 的验证 F1 分数
        self.monitor = monitor      # 监控指标名称
        self.best_value = 0.0       # 最佳监控指标值
        self.best_epoch = -1        # 最佳监控指标对应的 epoch
        self.related_metrics = None # 最佳监控指标对应的评估指标
        self.best_model_file = ""   # 最佳监控指标对应的模型文件路径
        self.done_epochs = 0        # 已完成的 epoch 数
        self.es_counter = 0         # 早停计数器

        if monitor not in {"val_acc", "val_loss", "val_f1"}:
            raise ValueError(f"Unknown monitor {monitor} for TrainingRecords")
        if mode not in {"min", "max", "auto"}:
            raise ValueError(f"Unknown mode {mode} for TrainingRecords")
        if mode == "auto":
            self.mode = "min" if "loss" in monitor else "max"
        else:
            self.mode = mode
        
        if self.mode == "min":
            self.compare_op = lambda a, b: a < b
            self.best_value = float('inf')
        elif self.mode == "max":
            self.compare_op = lambda a, b: a > b
            self.best_value = float('-inf')

    def update_train(self, loss: float, metrics: EvaluateMeter):
        self.train_losses.append(loss)
        self.train_accuracies.append(metrics.accuracy)
        self.train_f1_scores.append(metrics.f1_score())

    def update_val(self, loss: float, metrics: EvaluateMeter):
        val_acc, f1_score = metrics.accuracy, metrics.f1_score()
        
        self.val_losses.append(loss)
        self.val_accuracies.append(val_acc)
        self.val_f1_scores.append(f1_score)

        # 更新最佳模型
        if self.monitor == "val_acc":
            val_metric = val_acc
        elif self.monitor == "val_loss":
            val_metric = loss
        else:
            val_metric = f1_score
        
        self.done_epochs += 1

        if self.compare_op(val_metric, self.best_value):
            self.best_value = val_metric
            self.best_epoch = self.done_epochs
            self.related_metrics = metrics
            self.es_counter = 0
        else:
            self.es_counter += 1
    
    def update(self, stage: str, loss: float, metrics: EvaluateMeter):
        if stage.lower() == "train":
            self.update_train(loss, metrics)
        elif stage.lower() == "val" or stage.lower() == "validation":
            self.update_val(loss, metrics)
        else:
            logger.warning("Unknown stage {} for TrainingRecords update", stage)

    def as_dict(self):
        return {
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "train_f1_scores": self.train_f1_scores,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "val_f1_scores": self.val_f1_scores,
            "monitor": self.monitor,
            "mode": self.mode,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "best_model_file": self.best_model_file,
            "done_epochs": self.done_epochs,
            "es_counter": self.es_counter,
        }


class ModelTrainer:
    def __init__(self, args):
        train_args = args.training
        self.args = args
        self.device = train_args.device
        self.batch_size = train_args.batch_size
        self.max_epochs = train_args.max_epochs
        self.patience = train_args.patience
        self.save_best_n = train_args.save_best_n
        self.save_last_k = train_args.save_last_k
        self.class_names = args.model.class_names

        # TensorBoard writer
        self.tb_writer: Optional[SummaryWriter] = None
        self.train_step = 0
        self.val_step = 0

    def _init_tb(self, tb_writer: SummaryWriter = None):
        self.tb_writer = tb_writer
        self.train_step = 0
        self.val_step = 0

    def _update_tb(
        self,
        stage: str,
        advance: int = 0,
        epoch: int = None,
        **data: Union[int, float, matplotlib.figure.Figure, tuple[str, Any]]
    ):
        """更新 TensorBoard 记录
        Args:
            stage(str): 阶段名称，如 "Train" 或 "Validation"
            advance(int): 前进的步数
            epoch(int, optional): 当前 epoch, 使用epoch时, 将忽略advance, 按epoch更新步数
            data: 其他要记录的数据, 键为标签, 值为数据或数据元组, 如果为元组, 则第一个元素为数据类型("scalar"或"figure"), 第二个元素为数据本身
        """
        if self.tb_writer is None:
            return
        advance = advance if advance > 0 else 0

        stage = stage.title()
        if stage == "Train":
            self.train_step += advance
        else:
            self.val_step += advance

        if epoch is not None:
            global_step = epoch
        else:
            global_step = self.train_step if stage == "Train" else self.val_step

        for key, value in data.items():
            tag = '/'.join([key.title(), stage])

            if isinstance(value, tuple) and len(value) == 2:
                data_type, data_value = value
            else:
                data_type, data_value = None, value
                if isinstance(value, (int, float)):
                    data_type = "scalar"
                elif isinstance(value, matplotlib.figure.Figure):
                    data_type = "figure"
            
            if data_type == "scalar":
                self.tb_writer.add_scalar(tag, data_value, global_step)
            elif data_type == "figure":
                self.tb_writer.add_figure(tag, data_value, global_step)
            else:
                logger.warning("Unsupported data type {} for TensorBoard logging", type(value))

    def _progress(self, show_epoch: bool = True):
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeColumn(),
            ProgressSpeedColumn(unit="batch"),
            TextColumn("acc={task.fields[acc]:.5f}, loss={task.fields[loss]:.5f}"),
        ]
        if show_epoch:
            columns.insert(2, TextColumn("Epoch {task.fields[epoch]}/{task.fields[epochs]}"))
        return Progress(
            *columns,
            console=console,
        )

    @staticmethod
    def _extract_logits(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            return outputs[0]
        raise TypeError(f"Unsupported model output type: {type(outputs)}")

    def train_one_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        *meter_updated_cbks: Callable[[float, EvaluateMeter], Any]
    ):
        """训练一个批次
        Args:
            model(nn.Module): 待训练模型
            dataloader(DataLoader): 数据加载器
            criterion(nn.Module): 损失函数
            optimizer(torch.optim.Optimizer): 优化器
            meter_updated_cbks(tuple[Callable[[float, EvaluateMeter], Any], ...]): 每次指标器更新后的回调函数组
        """
        # 设定模型为训练模式
        model.train()
        # 损失指标
        running_loss = AverageMeter()
        # 评估指标
        metrics = EvaluateMeter()
        
        for inputs, labels in dataloader:
            inputs, labels = move2deivce(inputs, labels, device=self.device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = self._extract_logits(model(inputs))
            # 计算批次损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            with torch.no_grad():
                # 更新损失
                running_loss.update(loss.item(), n=labels.size(0))
                # 更新评估指标
                metrics.update(outputs, labels)
            
            for cbk in meter_updated_cbks:
                if cbk is not None:
                    cbk(running_loss.avg, metrics)

        return running_loss.avg, metrics

    def eval_one_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        *meter_updated_cbks: Callable[[float, EvaluateMeter], Any]
    ):
        """评估一个批次
        Args:
            model(nn.Module): 待评估模型
            dataloader(DataLoader): 数据加载器
            criterion(nn.Module): 损失函数
            meter_updated_cbks(tuple[Callable[[float, EvaluateMeter], Any], ...]): 每次指标器更新后的回调函数组
        """
        # 设定模型为评估模式
        model.eval()
        # 损失
        running_loss = AverageMeter()
        # 评估指标
        metrics = EvaluateMeter(save_logits=True)

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = move2deivce(inputs, labels, device=self.device)

                # 前向传播
                outputs = self._extract_logits(model(inputs))
                # 计算批次损失
                loss = criterion(outputs, labels)

                # 更新损失
                running_loss.update(loss.item(), n=labels.size(0))
                # 更新评估指标
                metrics.update(outputs, labels)

                for cbk in meter_updated_cbks:
                    if cbk is not None:
                        cbk(running_loss.avg, metrics)

        return running_loss.avg, metrics
    
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        save_filepath: str = "model_epoch_{epoch}_valacc_{val_acc:.4f}.pth",
        tb_writer: SummaryWriter = None,
    ):
        model.to(self.device)

        # 训练过程记录
        records = TrainingRecords(self.args.training.monitor, self.args.training.mode)
        # checkpoint 管理
        checkpoint = ModelCheckpoint(save_filepath, self.args.training.monitor, self.args.training.mode, self.save_best_n, self.save_last_k)
        # 初始化 TensorBoard 记录器
        self._init_tb(tb_writer)

        epochs = self.max_epochs
        with self._progress() as progress:
            train_task = progress.add_task(f"Train", start=False, total=len(train_loader), epoch=0, epochs=epochs, acc=0, loss=0)
            val_task = progress.add_task(f"Valid", start=False, total=len(val_loader), epoch=0, epochs=epochs, acc=0, loss=0)
            
            train_cbk = (
                lambda loss, metrics: progress.update(train_task, loss=loss, acc=metrics.accuracy, advance=1),
                lambda loss, metrics: self._update_tb("Train", advance=1, loss=loss, accuracy=metrics.accuracy)
            )
            val_cbk = (
                lambda loss, metrics: progress.update(val_task, loss=loss, acc=metrics.accuracy, advance=1),
                lambda loss, metrics: self._update_tb("Validation", advance=1, loss=loss, accuracy=metrics.accuracy)
            )

            for epoch in range(1, epochs+1):
                progress.reset(train_task, epoch=epoch, epochs=epochs, acc=0, loss=0)
                train_loss, train_metrics = self.train_one_epoch(model, train_loader, criterion, optimizer, *train_cbk)

                progress.reset(val_task, epoch=epoch, epochs=epochs, acc=0, loss=0)
                val_loss, val_metrics = self.eval_one_epoch(model, val_loader, criterion, *val_cbk)

                train_acc, train_f1 = train_metrics.accuracy, train_metrics.f1_score()
                val_acc, val_f1 = val_metrics.accuracy, val_metrics.f1_score()
                
                logger.info(
                    "Epoch {}/{} - Train loss: {:.5f}, acc: {:.5f}, f1: {:.5f} - Val loss: {:.5f}, acc: {:.5f}, f1: {:.5f}",
                    epoch, epochs, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1,
                )

                records.update("train", train_loss, train_metrics)
                records.update("validation", val_loss, val_metrics)

                self._update_tb("Train", epoch=epoch, f1_score=train_f1)
                self._update_tb("Validation", epoch=epoch, f1_score=val_f1)
                if self.tb_writer is not None:
                    fig = plot_confusion_matrix(
                        val_metrics.confusion_matrix(),
                        class_names=self.class_names,
                        title=f"Confusion Matrix - Validation Epoch {epoch}"
                    )
                    self._update_tb("Validation", epoch=epoch, confusion_matrix=fig)
                    plt.close(fig)

                checkpoint.step(model, optimizer, epoch=epoch, val_acc=val_acc, val_loss=val_loss, val_f1=val_f1)
                records.best_model_file = checkpoint.get_best_model_path() or ""

                if records.es_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        return records

    def test(self, model: nn.Module, test_loader: DataLoader, criterion: nn.Module):
        model.to(self.device)
        
        with self._progress(show_epoch=False) as progress:
            task = progress.add_task("Testing", total=len(test_loader), acc=0.0, loss=0.0)
            cbk = lambda loss, metrics: progress.update(task, loss=loss, acc=metrics.accuracy, advance=1)
            
            _, test_metrics = self.eval_one_epoch(model, test_loader, criterion, cbk)
        
        return test_metrics
