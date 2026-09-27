"""
EEG 模型可视化分析模块
============================
提供对 MyModel / MyModelCogFusion 的深度可解释性分析工具，包括：

1. 脑区连接热图   —— 来自 GlobalGraphConv 的动态邻接矩阵
2. 脑区节点重要性  —— 来自 GlobalAttentionPooling 的频带注意力权重映射回电极
3. 时序自注意力   —— 来自 SelfAttentionBlock 的注意力权重矩阵
4. 跨模态注意力   —— 来自 CrossAttentionBlock (EEG Query × Cognitive Key) 的关联矩阵
5. 跨模态时间强度 —— 对跨模态注意力按行/列求平均后的时间关注强度曲线
6. 显著性图       —— 输出关于输入的梯度绝对值 |∂output/∂input|
7. 积分梯度       —— 沿基线到输入路径积分的梯度归因 (Integrated Gradients)

使用方式:
    >>> interpreter = ModelInterpreter(model, dataset='SEED', graph_type='general')
    >>> with interpreter.capture(eeg_tensor, cog_tensor):  # cog_tensor can be None
    ...     pass
    >>> interpreter.plot_brain_heatmap()
    >>> interpreter.plot_temporal_attention()
    >>> interpreter.plot_cross_attention(band_names, time_labels, cog_names)
    >>> interpreter.plot_cross_attention_strength(mode='col')

    # 梯度归因
    >>> saliency = interpreter.compute_saliency(eeg_tensor, cog_tensor, target_class=1)
    >>> interpreter.plot_saliency_topo(saliency)
    >>> ig = interpreter.compute_integrated_gradients(eeg_tensor, cog_tensor, steps=50)
    >>> interpreter.plot_ig_topo(ig)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, List, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mne
from mne_connectivity.viz import plot_connectivity_circle

from utils.channel import get_ch_name, get_local_ch_num, channel as channel_info


def _find_modules(model: nn.Module, target_cls):
    """从模型中查找所有匹配目标类的子模块列表"""
    results = []
    for module in model.modules():
        if isinstance(module, target_cls):
            results.append(module)
    return results


def _plot_eeg_topomap(data: np.ndarray, channel_names: Sequence[str], title: str = None, ax: plt.Axes = None, cmap: str = None) -> plt.Figure:
    info = mne.create_info(ch_names=list(channel_names), sfreq=128.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1005", head_size=0.09)
    info.set_montage(montage, match_alias={'CB1': 'POO7', 'CB2': 'POO8'})
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    mne.viz.plot_topomap(data, info, ch_type="eeg", names=channel_names, axes=ax, show=False, cmap=cmap)
    if title:
        ax.set_title(title)
    return ax.figure


class ModelInterpreter:
    """
    EEG 模型可解释性分析器

    通过 PyTorch forward hook(无侵入式)捕获模型推断过程中的中间激活值：
    - GlobalGraphConv  → 脑区间邻接矩阵 (动态图连接强度)
    - GlobalAttentionPooling → 频带融合注意力权重
    - MultiheadAttention (Self)  → 时序自注意力矩阵
    - MultiheadAttention (Cross) → 认知-EEG 跨模态注意力矩阵

    Args:
        model (nn.Module): 已加载权重的 MyModel 或 MyModelCogFusion 实例
        dataset (str):     数据集名称, 用于查询电极位置 ('SEED' 或 'MLED')
        graph_type (str):  图划分类型 ('general', 'frontal', 'hemisphere' 等)
    """

    def __init__(self, model: nn.Module, dataset: str = 'SEED', graph_type: str = 'general', band_names: List[str] = None):
        self.model = model
        self.dataset = dataset
        self.graph_type = graph_type
        self.band_names = band_names
        if band_names is None:
            self.band_names = ['delta', 'theta', 'Alpha', 'Beta', 'Gamma']

        # 解析电极信息
        self.ch_names: List[str] = get_ch_name(dataset, graph_type)
        self.area_nodes: List[int] = get_local_ch_num(dataset, graph_type)
        self.num_areas = len(self.area_nodes)

        # 生成脑区标签 (取每个区域中间电极名作为区域名)
        self.area_names: List[str] = self._build_area_names()

        # 捕获的中间值(调用 capture() 后填充)
        self.gcn_adjs: Dict[str, list[np.ndarray]] = {} # {band_name: n * (A, A)}
        self.band_attn_weights: List[np.ndarray] = []  # n * (num_bands,)
        self.deep_attn_weights: List[np.ndarray] = []  # n * (T, T)，来自 MyModel.self_attn / CogAttentionFusion.attn_deep
        self.cog_attn_weights: List[np.ndarray] = []   # n * (T, T)，来自 CogAttentionFusion.attn_cog
        self.cross_attn_weights: List[np.ndarray] = [] # n * (T_q, T_k)

        self._hooks = []
        self._adj_call_count = 0  # 用于追踪 hook 调用次数，以确定当前频带

    def _build_area_names(self) -> List[str]:
        """生成每个脑区的标签，用每区第一个电极名表示"""
        if self.graph_type == 'original':
            return self.ch_names  # original 模式下区域=电极
        ch_areas = channel_info[self.dataset][self.graph_type]
        return [area[len(area) // 2] for area in ch_areas]

    @contextmanager
    def capture(self, eeg_x: torch.Tensor, cog_x: Optional[torch.Tensor] = None, accumulate: bool = False):
        """
        捕获一次前向传播的所有中间激活结果。

        用法:
            >>> with interpreter.capture(eeg_x, cog_x):
            ...     pass  # 推断在 with 块内自动完成

        Args:
            eeg_x:  EEG 输入, shape=(B, S, Ch, Bands)
            cog_x:  认知特征输入, shape=(B, S, CogDim), MyModel 时传 None
            accumulate:  是否累积捕获结果。若为 False，会先重置缓存；若为 True，追加到现有缓存中
        """
        if not accumulate:
            self._reset_cache()
        self._register_hooks()
        try:
            self.model.eval()
            with torch.no_grad():
                if cog_x is not None:
                    self.model((eeg_x, cog_x))
                else:
                    self.model(eeg_x)
            yield self
        finally:
            self._remove_hooks()

    def _reset_cache(self):
        self.gcn_adjs.clear()
        self.band_attn_weights.clear()
        self.deep_attn_weights.clear()
        self.cog_attn_weights.clear()
        self.cross_attn_weights.clear()
        self._adj_call_count = 0

    def _register_hooks(self):
        """注册所有需要的 forward hooks"""
        from net.model import GlobalGraphConv, SelfAttentionBlock, CrossAttentionBlock
        from net.layer import GlobalAttentionPooling, MultiheadAttention

        # ① 捕获 GlobalGraphConv 的邻接矩阵
        for m in _find_modules(self.model, GlobalGraphConv):
            h = m.register_forward_hook(self._hook_adjacency)
            self._hooks.append(h)

        # ② 捕获 GlobalAttentionPooling (仅 band_fusion 层的)
        # band_fusion 是 SpatialBandLearning 的最后一个子模块
        if hasattr(self.model, 'sb_learning') and hasattr(self.model.sb_learning, 'band_fusion'):
            h = self.model.sb_learning.band_fusion.register_forward_hook(self._hook_band_attn)
            self._hooks.append(h)

        # ③ 捕获 deep self-attention(MyModel.self_attn 各层)
        if hasattr(self.model, 'self_attn'):
            for m in self.model.self_attn:
                if isinstance(m, SelfAttentionBlock):
                    h = m.attention.register_forward_hook(self._hook_deep_attn)
                    self._hooks.append(h)

        # ③b 捕获 CogAttentionFusion 内的 deep / cog self-attention
        if hasattr(self.model, 'fusion_layer'):
            fl = self.model.fusion_layer
            if hasattr(fl, 'attn_deep') and isinstance(fl.attn_deep, SelfAttentionBlock):
                h = fl.attn_deep.attention.register_forward_hook(self._hook_deep_attn)
                self._hooks.append(h)
            if hasattr(fl, 'attn_cog') and isinstance(fl.attn_cog, SelfAttentionBlock):
                h = fl.attn_cog.attention.register_forward_hook(self._hook_cog_attn)
                self._hooks.append(h)

        # ④ 捕获 CrossAttentionBlock 中的注意力权重
        for m in _find_modules(self.model, CrossAttentionBlock):
            h = m.attention.register_forward_hook(self._hook_cross_attn)
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _hook_adjacency(self, module, inputs, output):
        """捕获 GlobalGraphConv 的原始邻接矩阵(未归一化，用于可视化连接强度)
        
        注意：SGConv 内部会对此矩阵做 D^{-1/2}·A·D^{-1/2} 归一化用于特征传播，
        但可视化时使用原始值更能直观反映模型学到的脑区连接强弱。
        """
        # inputs[0] shape: (B*S, num_areas, in_features) — GlobalGraphConv 的原始输入
        x = inputs[0]
        with torch.no_grad():
            adj = module.get_adjacency(x)  # (B*S, A, A)，与 forward 中完全一致
        # 对 B*S 维度做平均，得到代表性邻接矩阵
        adj_mean = adj.mean(dim=0).cpu().numpy()  # (A, A)
        
        # 根据调用次数确定当前频带(假设对每个频带依次调用 GlobalGraphConv)
        band_idx = self._adj_call_count % len(self.band_names)
        band_name = self.band_names[band_idx]
        
        if band_name not in self.gcn_adjs:
            self.gcn_adjs[band_name] = []
        self.gcn_adjs[band_name].append(adj_mean)
        
        self._adj_call_count += 1

    def _hook_band_attn(self, module, inputs, output):
        """捕获 GlobalAttentionPooling 的频带注意力权重"""
        x = inputs[0]  # (B*S, num_bands, out_features)
        with torch.no_grad():
            scores = module.query @ x.transpose(-1, -2)  # (B*S, 1, num_bands)
            scores = scores * module.scale
            attn = torch.softmax(scores, dim=-1).squeeze(1)  # (B*S, num_bands)
        # 对序列维度求均值
        attn_mean = attn.mean(dim=0).cpu().numpy()  # (num_bands,)
        self.band_attn_weights.append(attn_mean)

    def _hook_deep_attn(self, module, inputs, output):
        """捕获 deep EEG 自注意力矩阵 (MyModel.self_attn / CogAttentionFusion.attn_deep)"""
        query = inputs[0]
        _, attn_weights = module.forward(query, query, query, need_weights=True)
        if attn_weights is not None:
            # attn_weights: (B, H, L, L)
            attn_mean = attn_weights.mean(dim=1).mean(dim=0).cpu().numpy()  # (L, L)
            self.deep_attn_weights.append(attn_mean)

    def _hook_cog_attn(self, module, inputs, output):
        """捕获 cognitive 自注意力矩阵 (CogAttentionFusion.attn_cog)"""
        query = inputs[0]
        _, attn_weights = module.forward(query, query, query, need_weights=True)
        if attn_weights is not None:
            # attn_weights: (B, H, L, L)
            attn_mean = attn_weights.mean(dim=1).mean(dim=0).cpu().numpy()  # (L, L)
            self.cog_attn_weights.append(attn_mean)

    def _hook_cross_attn(self, module, inputs, output):
        """捕获 CrossAttentionBlock.attention 的跨模态注意力矩阵"""
        query, key, value = inputs[0], inputs[1], inputs[2]
        # 直接调用 forward 方法，绕过 __call__ 的 hook 分发，避免无限递归
        _, attn_weights = module.forward(query, key, value, need_weights=True)
        if attn_weights is not None:
            # attn_weights: (B, H, L_q, L_k)
            attn_mean = attn_weights.mean(dim=1).mean(dim=0).cpu().numpy()
            self.cross_attn_weights.append(attn_mean)


    def plot_area_connectivity(
        self,
        band_idx: Optional[str|int] = None,
        n_cols: int = 3,
        cmap: str = None,
        top_k: int = 15,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """绘制脑区连接性
        
        Args:
            band_idx: 绘制哪个频带, 整数表示绘制某个具体频带, None表示对所有频带取均值, 'all'表示绘制各频带, 'all+'表示绘制均值+各频带
            n_cols:   绘制多个图时每行显示的子图数量
            cmap:     颜色映射
            top_k:    仅显示连接强度最高的 k 条边]
            title:    图标题
            save_path: 保存路径
        """
        assert len(self.gcn_adjs) > 0, "请先调用 capture() 进行推断"

        band_names = list(self.gcn_adjs.keys())
        _cmap = cmap or 'OrRd'

        def _avg_adj(bname: str) -> np.ndarray:
            """对某频带内所有 GCN 层的邻接矩阵取均值 → (A, A)"""
            return np.mean(self.gcn_adjs[bname], axis=0)

        def _draw(adj: np.ndarray, ax: plt.Axes, band_label: str = None):
            """在指定 polar ax 上绘制连接圆图"""
            node_importance = adj.sum(axis=1)
            node_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-8)
            adj_norm = adj / (adj.max() + 1e-8)
            plot_connectivity_circle(
                adj_norm,
                self.area_names,
                n_lines=top_k,
                facecolor='white',
                textcolor='black',
                node_colors=plt.get_cmap('Blues')(node_importance),
                colormap=_cmap,
                linewidth=2,
                node_linewidth=1,
                ax=ax,
                show=False,
            )
            if band_label:
                ax.set_title(band_label, pad=12)

        if isinstance(band_idx, int):
            # ── 情形 1: 单个频带 ──────────────────────────────────
            bname = band_names[band_idx]
            adj = _avg_adj(bname)
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='polar')
            _draw(adj, ax)
            if title:
                fig.suptitle(title, fontsize=13)

        elif band_idx is None:
            # ── 情形 2: 所有频带邻接矩阵整体均值 ─────────────────
            adj = np.mean([_avg_adj(b) for b in band_names], axis=0)
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='polar')
            _draw(adj, ax)
            if title:
                fig.suptitle(title, fontsize=13)

        elif band_idx == 'all':
            # ── 情形 3: 每个频带各一个子图 ────────────────────────
            n_bands = len(band_names)
            n_rows = (n_bands + n_cols - 1) // n_cols
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * 5, n_rows * 5),
                subplot_kw={'projection': 'polar'},
                squeeze=False,
            )
            axes_flat = axes.flatten()
            for i, bname in enumerate(band_names):
                _draw(_avg_adj(bname), axes_flat[i], band_label=bname)
            for j in range(n_bands, len(axes_flat)):
                axes_flat[j].set_visible(False)
            if title:
                fig.suptitle(title, fontsize=14)
            fig.tight_layout()

        elif band_idx == 'all+':
            # ── 情形 4: 均值子图 + 每个频带各一个子图 ─────────────
            n_bands = len(band_names)
            n_plots = 1 + n_bands
            n_rows = (n_plots + n_cols - 1) // n_cols
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * 5, n_rows * 5),
                subplot_kw={'projection': 'polar'},
                squeeze=False,
            )
            axes_flat = axes.flatten()
            avg_adj = np.mean([_avg_adj(b) for b in band_names], axis=0)
            _draw(avg_adj, axes_flat[0], band_label='All Bands (Mean)')
            for i, bname in enumerate(band_names):
                _draw(_avg_adj(bname), axes_flat[i + 1], band_label=bname)
            for j in range(n_plots, len(axes_flat)):
                axes_flat[j].set_visible(False)
            if title:
                fig.suptitle(title, fontsize=14)
            fig.tight_layout()

        else:
            raise ValueError(f"band_idx 应为 int、None、'all' 或 'all+'，当前值: {band_idx!r}")

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_gcn_adjacency(
        self,
        band_idx: Optional[str|int] = None,
        n_cols: int = 3,
        cmap: str = 'hot_r',
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制 GCN 归一化邻接矩阵热图，行列均为脑区，颜色表示连接强度。

        Args:
            band_idx:  绘制哪个频带, 整数表示绘制某个具体频带, None 表示对所有频带取均值,
                       'all' 表示每个频带各一个子图, 'all+' 表示均值子图 + 各频带子图
            n_cols:    绘制多个图时每行显示的子图数量
            cmap:      颜色映射
            title:     图标题
            save_path: 保存路径
        """
        assert len(self.gcn_adjs) > 0, "请先调用 capture() 进行推断"

        band_names = list(self.gcn_adjs.keys())
        _cmap = cmap or 'hot_r'
        names = self.area_names
        A = len(names)
        fig_w = max(6, A * 0.55)
        fig_h = max(5, A * 0.50)

        def _avg_adj_norm(bname: str) -> np.ndarray:
            """对某频带内所有 GCN 层邻接矩阵取均值后做归一化 → (A, A)"""
            adj = np.mean(self.gcn_adjs[bname], axis=0)
            adj = adj + np.eye(adj.shape[0])
            deg = adj.sum(axis=1, keepdims=True)
            d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
            adj_norm = d_inv_sqrt * adj * d_inv_sqrt.T
            return adj_norm

        def _draw(adj: np.ndarray, ax: plt.Axes, band_label: str = None):
            im = ax.imshow(adj, cmap=_cmap, aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(range(A))
            ax.set_yticks(range(A))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel('目标脑区', fontsize=10)
            ax.set_ylabel('源脑区', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='归一化连接强度')
            if band_label:
                ax.set_title(band_label, fontsize=11)

        if isinstance(band_idx, int):
            # ── 情形 1: 单个频带 ──────────────────────────────────
            bname = band_names[band_idx]
            adj = _avg_adj_norm(bname)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            _draw(adj, ax)
            ax.set_title(title or f'归一化邻接矩阵 ({bname})', fontsize=13)

        elif band_idx is None:
            # ── 情形 2: 所有频带邻接矩阵整体均值 ─────────────────
            adj = np.mean([_avg_adj_norm(b) for b in band_names], axis=0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            _draw(adj, ax)
            ax.set_title(title or '归一化邻接矩阵(所有频带均值)', fontsize=13)

        elif band_idx == 'all':
            # ── 情形 3: 每个频带各一个子图 ────────────────────────
            n_bands = len(band_names)
            n_rows = (n_bands + n_cols - 1) // n_cols
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * fig_w, n_rows * fig_h),
                squeeze=False,
            )
            axes_flat = axes.flatten()
            for i, bname in enumerate(band_names):
                _draw(_avg_adj_norm(bname), axes_flat[i], band_label=bname)
            for j in range(n_bands, len(axes_flat)):
                axes_flat[j].set_visible(False)
            if title:
                fig.suptitle(title, fontsize=14)
            fig.tight_layout()

        elif band_idx == 'all+':
            # ── 情形 4: 均值子图 + 每个频带各一个子图 ─────────────
            n_bands = len(band_names)
            n_plots = 1 + n_bands
            n_rows = (n_plots + n_cols - 1) // n_cols
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * fig_w, n_rows * fig_h),
                squeeze=False,
            )
            axes_flat = axes.flatten()
            avg_adj = np.mean([_avg_adj_norm(b) for b in band_names], axis=0)
            _draw(avg_adj, axes_flat[0], band_label='All Bands (Mean)')
            for i, bname in enumerate(band_names):
                _draw(_avg_adj_norm(bname), axes_flat[i + 1], band_label=bname)
            for j in range(n_plots, len(axes_flat)):
                axes_flat[j].set_visible(False)
            if title:
                fig.suptitle(title, fontsize=14)
            fig.tight_layout()

        else:
            raise ValueError(f"band_idx 应为 int、None、'all' 或 'all+'，当前值: {band_idx!r}")

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_band_importance(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制频带融合注意力权重(GlobalAttentionPooling 输出)，展示每个频带对最终预测的贡献度。

        Args:
            title:       图标题
            save_path:   保存路径
        """
        assert len(self.band_attn_weights) > 0, "请先调用 capture() 进行推断"
        weights = self.band_attn_weights[0]  # (num_bands,)

        fig, ax = plt.subplots(figsize=(max(5, len(weights) * 1.0), 4))
        colors = plt.get_cmap('viridis')(np.linspace(0.2, 0.85, len(weights)))
        bars = ax.bar(self.band_names, weights, color=colors, edgecolor='black', linewidth=0.8)

        for bar, w in zip(bars, weights):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{w:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('注意力权重', fontsize=11)
        ax.set_title(title or '频带融合注意力权重', fontsize=13)
        ax.set_ylim(0, weights.max() * 1.2)
        ax.spines[['top', 'right']].set_visible(False)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_temporal_attention(
        self,
        layer_index: int = -1,
        attn_type: str = 'deep',
        cmap: str = 'Blues',
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制 Transformer 时序自注意力权重矩阵(行=Query时间步, 列=Key时间步)。
        颜色越深表示对应两个时间窗口之间的依赖越强。

        Args:
            layer_index: 选取第几层 Transformer, -1 表示最后一层
            attn_type:   'deep' 表示 EEG deep 自注意力, 'cog' 表示 cognitive 自注意力
            cmap:        颜色映射
            title:       图标题
            save_path:   保存路径
        """
        if attn_type == 'cog':
            assert len(self.cog_attn_weights) > 0, "请先调用 capture() 进行推断，且模型需包含 CogAttentionFusion.attn_cog"
            attn = self.cog_attn_weights[layer_index]
        else:
            assert len(self.deep_attn_weights) > 0, "请先调用 capture() 进行推断"
            attn = self.deep_attn_weights[layer_index]  # (T, T)

        T = attn.shape[0]
        time_labels = [f't{i+1}' for i in range(T)]

        fig, ax = plt.subplots(figsize=(max(6, T * 0.45), max(5, T * 0.4)))
        im = ax.imshow(attn, cmap=cmap, aspect='auto', vmin=0)

        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(time_labels, fontsize=8)
        ax.set_xlabel('Key 时间步', fontsize=11)
        ax.set_ylabel('Query 时间步', fontsize=11)
        ax.set_title(title or f'时序自注意力权重(第 {layer_index} 层)', fontsize=13)

        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.01, label='注意力权重')
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_cross_attention(
        self,
        cmap: str = 'YlOrRd',
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制 EEG-认知跨模态注意力矩阵(EEG Query 在不同时间步对认知特征的关注程度)。
        行 = EEG 时间步(Query), 列 = 认知特征时间步(Key)

        Args:
            time_labels: EEG 时间步标签
            cog_labels:  认知特征时间步或名称标签
            cmap:        颜色映射
            title:       图标题
            save_path:   保存路径
        """
        assert len(self.cross_attn_weights) > 0, "请先调用 capture(cog_x=...) 并使用 MyModelCogFusion"
        attn = self.cross_attn_weights[0]  # (T_q, T_k)

        T_q, T_k = attn.shape
        query_labels = [f't{i+1}' for i in range(T_q)]
        key_labels = [f't{i+1}' for i in range(T_k)]

        fig, ax = plt.subplots(figsize=(max(5, T_k * 0.6), max(5, T_q * 0.45)))
        im = ax.imshow(attn, cmap=cmap, aspect='auto', vmin=0)

        ax.set_xticks(range(T_k))
        ax.set_yticks(range(T_q))
        ax.set_xticklabels(key_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(query_labels, fontsize=9)
        ax.set_xlabel('认知特征时间步(Key)', fontsize=11)
        ax.set_ylabel('时频图特征时间步(Query)', fontsize=11)
        ax.set_title(title or 'EEG-认知跨模态注意力权重', fontsize=13)

        # 在格子内写数值
        # if T_q * T_k <= 200:
        #     for i in range(T_q):
        #         for j in range(T_k):
        #             ax.text(j, i, f'{attn[i, j]:.2f}', ha='center', va='center',
        #                     fontsize=7, color='black' if attn[i, j] < 0.5 else 'white')

        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_cross_attention_strength(
        self,
        mode: str = 'col',
        color: str = '#D94801',
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        将跨模态注意力矩阵沿某一维求均值，绘制时间维度上的关注强度曲线。

        Args:
            mode:      平均方向。'col' 表示按列统计(对 Query 维求均值，默认)，
                       'row' 表示按行统计(对 Key 维求均值)
            color:     曲线颜色
            title:     图标题
            save_path: 保存路径
        """
        assert len(self.cross_attn_weights) > 0, "请先调用 capture(cog_x=...) 并使用 MyModelCogFusion"
        attn = self.cross_attn_weights[0]  # (T_q, T_k)

        mode_key = mode.lower()
        if mode_key in ('col', 'column', 'key', 'k'):
            strength = attn.mean(axis=0)  # (T_k,)
            labels = [f't{i+1}' for i in range(attn.shape[1])]
            xlabel = '认知特征时间步(Key)'
            default_title = '跨模态注意力时间强度(按列平均)'
        elif mode_key in ('row', 'query', 'q'):
            strength = attn.mean(axis=1)  # (T_q,)
            labels = [f't{i+1}' for i in range(attn.shape[0])]
            xlabel = '时频图特征时间步(Query)'
            default_title = '跨模态注意力时间强度(按行平均)'
        else:
            raise ValueError(f"mode 应为 'col' 或 'row'，当前值: {mode!r}")

        x = np.arange(len(strength))
        fig, ax = plt.subplots(figsize=(max(6, len(strength) * 0.5), 4))
        ax.plot(x, strength, color=color, marker='o', linewidth=2)
        ax.fill_between(x, 0, strength, color=color, alpha=0.18)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('平均关注强度', fontsize=11)
        ax.set_title(title or default_title, fontsize=13)
        # ax.set_ylim(bottom=0)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def compute_saliency(
        self,
        eeg_x: torch.Tensor,
        cog_x: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        abs: bool = False,
    ) -> np.ndarray:
        """
        计算输入 EEG 特征的显著性图(Saliency Map)。

        显著性 = |∂output_class / ∂eeg_input|，反映每个输入维度对模型预测结果的局部敏感程度。

        Args:
            eeg_x:        EEG 输入张量, shape=(B, S, Ch, Bands)
            cog_x:        认知特征输入(可选), shape=(B, S, CogDim)
            target_class: 目标类别索引。若为 None, 使用各样本的预测类别
            abs:          是否返回梯度绝对值。若 False, 返回带符号的梯度(正值表示正向贡献，负值表示负向贡献)

        Returns:
            np.ndarray: 梯度绝对值, shape=(B, S, Ch, Bands)
        """
        self.model.eval()
        x = eeg_x.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            if cog_x is not None:
                logits = self.model((x, cog_x))
            else:
                logits = self.model(x)

            if target_class is None:
                # 使用各样本的预测类别
                pred = logits.argmax(dim=-1)  # (B,)
                score = logits.gather(1, pred.unsqueeze(1)).sum()
            else:
                score = logits[:, target_class].sum()

            score.backward()

        x_gard = x.grad.abs() if abs else x.grad

        saliency = x_gard.detach().cpu().numpy()  # (B, S, Ch, Bands)
        return saliency

    def compute_integrated_gradients(
        self,
        eeg_x: torch.Tensor,
        cog_x: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
    ) -> np.ndarray:
        """
        计算积分梯度(Integrated Gradients)归因。

        IG(x) = (x - x') x ∫₀¹ ∂F(x' + α(x-x')) / ∂x dα

        沿基线 x'(默认全零)到输入 x 的直线路径对梯度进行梯形积分，
        满足 *完整性公理*：所有电极归因之和 ≈ F(x) - F(x')。

        Args:
            eeg_x:        EEG 输入张量, shape=(B, S, Ch, Bands)
            cog_x:        认知特征输入(可选), shape=(B, S, CogDim)，在路径上保持固定
            target_class: 目标类别索引。若为 None, 使用基于原始输入的预测类别
            baseline:     基线输入, shape 与 eeg_x 相同。默认全零张量
            steps:        积分步数(建议 50~300), 步数越多近似越精确

        Returns:
            np.ndarray: 积分梯度归因值(带符号): shape=(B, S, Ch, Bands)
        """
        self.model.eval()

        if baseline is None:
            baseline = torch.zeros_like(eeg_x)

        # 确定目标类别(基于原始输入的预测)
        use_per_sample = (target_class is None)
        if use_per_sample:
            with torch.no_grad():
                if cog_x is not None:
                    logits = self.model((eeg_x, cog_x))
                else:
                    logits = self.model(eeg_x)
            target_idx = logits.argmax(dim=-1)  # (B,)

        # 生成 steps+1 个插值点 α ∈ [0, 1]
        alphas = torch.linspace(0, 1, steps + 1, device=eeg_x.device)
        grads_list: List[torch.Tensor] = []

        for alpha in alphas:
            x_interp = (baseline + alpha * (eeg_x - baseline)).clone().detach().requires_grad_(True)

            with torch.enable_grad():
                if cog_x is not None:
                    logits = self.model((x_interp, cog_x))
                else:
                    logits = self.model(x_interp)

                if use_per_sample:
                    score = logits.gather(1, target_idx.unsqueeze(1)).sum()
                else:
                    score = logits[:, target_class].sum()

                score.backward()

            grads_list.append(x_interp.grad.detach().clone())

        # 梯形积分: (g[0]/2 + g[1] + ... + g[N-1] + g[N]/2) / N
        grads_tensor = torch.stack(grads_list, dim=0)  # (steps+1, B, S, Ch, Bands)
        avg_grads = (grads_tensor[0] + grads_tensor[-1]) / 2.0 + grads_tensor[1:-1].sum(dim=0)
        avg_grads = avg_grads / steps

        ig = (eeg_x - baseline).detach() * avg_grads  # (B, S, Ch, Bands)
        return ig.cpu().numpy()
    
    def plot_attribution_topo(
        self,
        attribution: np.ndarray,
        band_idx: Optional[int|str] = None,
        n_cols: int = 3,
        cmap: str = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        将 Sliency Map / Integrated Gradients 的结果绘制为头皮地形图。

        Args:
            attribution: 结果矩阵, shape=(B, S, Ch, Bands)
            band_idx:    绘制哪个频带, 整数表示绘制某个具体频带, None表示对所有频带取均值, 'all'表示绘制各频带, 'all+'表示绘制均值+各频带
            n_cols:      绘制多个图时每行显示的子图数量
            cmap:        颜色映射
            title:       图标题
            save_path:   保存路径
        """
        # 先沿 B 和 S 维度求均值，得到 (Bands, Ch)
        band_data = attribution.mean(axis=(0, 1)).transpose(1, 0)

        if isinstance(band_idx, int):
            # ── 情形 1: 单个频带 ──────────────────────────────────
            data = band_data[band_idx]
            band_label = self.band_names[band_idx] if band_idx < len(self.band_names) else f'Band {band_idx}'
            fig = _plot_eeg_topomap(data, self.ch_names, title=title, cmap=cmap)

        elif band_idx is None:
            # ── 情形 2: 所有频带均值 ──────────────────────────────
            data = band_data.mean(axis=0)
            fig = _plot_eeg_topomap(data, self.ch_names, title=title, cmap=cmap)

        elif band_idx == 'all':
            # ── 情形 3: 每个频带各一个子图 ────────────────────────
            n_bands = band_data.shape[0]
            n_rows = (n_bands + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * 4, n_rows * 4),
                                     squeeze=False)
            axes_flat = axes.flatten()
            for i in range(n_bands):
                band_label = self.band_names[i] if i < len(self.band_names) else f'Band {i}'
                _plot_eeg_topomap(band_data[i], self.ch_names, title=band_label, ax=axes_flat[i], cmap=cmap)
            for j in range(n_bands, len(axes_flat)):
                axes_flat[j].set_visible(False)
            if title:
                fig.suptitle(title, fontsize=14)
            fig.tight_layout()

        elif band_idx == 'all+':
            # ── 情形 4: 均值子图 + 每个频带各一个子图 ─────────────
            n_bands = band_data.shape[0]
            n_plots = 1 + n_bands
            n_rows = (n_plots + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * 4, n_rows * 4),
                                     squeeze=False)
            axes_flat = axes.flatten()
            # 第一个子图：均值
            _plot_eeg_topomap(band_data.mean(axis=0), self.ch_names, title='All Bands (Mean)', ax=axes_flat[0], cmap=cmap)
            # 后续子图：各频带
            for i in range(n_bands):
                band_label = self.band_names[i] if i < len(self.band_names) else f'Band {i}'
                _plot_eeg_topomap(band_data[i], self.ch_names, title=band_label, ax=axes_flat[i + 1], cmap=cmap)
            for j in range(n_plots, len(axes_flat)):
                axes_flat[j].set_visible(False)
            if title:
                fig.suptitle(title, fontsize=14)
            fig.tight_layout()

        else:
            raise ValueError(f"band_idx 应为 int、None、'all' 或 'all+'，当前值: {band_idx!r}")

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


def analyze_model(
    model: nn.Module,
    dataloader,
    has_cog: bool = False,
    dataset: str = 'SEED',
    graph_type: str = 'general',
    band_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    device: torch.device = None,
) -> ModelInterpreter:
    """
    一站式分析函数：创建 ModelInterpreter, 执行推断, 并绘制所有可视化图。

    输出图列表：
        - saliency_topo.svg          Saliency Map 头皮地形图(各频带 + 均值)
        - area_connectivity.svg      脑区连接圆图(均值 + 各频带)
        - gcn_adjacency.svg          脑区归一化邻接矩阵热图(均值 + 各频带)
        - band_importance.svg        频带重要性条形图
        - deep_attn.svg              EEG deep 自注意力权重热图
        - cog_attn.svg               Cognitive 自注意力权重热图
        - cross_attn.svg             EEG-认知跨模态注意力权重热图
        - cross_attn_strength_col.svg EEG-认知跨模态时间关注强度曲线图(按列平均)

    Args:
        model:       训练好的 MyModel 或 MyModelCogFusion 实例
        dataloader:  torch.utils.data.DataLoader 实例
        has_cog:     是否包含认知特征
        dataset:     数据集名称 ('SEED' 或 'MLED')
        graph_type:  图类型 ('general', 'frontal', 'hemisphere' 等)
        band_names:  频带名称列表，默认为 ['Delta','Theta','Alpha','Beta','Gamma']
        save_dir:    若非 None, 将所有图保存至该目录
        device:      计算设备

    Returns:
        ModelInterpreter 对象(后续可继续调用其方法)
    """
    import os
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def _sp(name):
        return os.path.join(save_dir, name) if save_dir else None

    if band_names is None:
        band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    if device is None:
        device = next(model.parameters()).device

    interpreter = ModelInterpreter(model, dataset=dataset, graph_type=graph_type, band_names=band_names)

    # 使用 DataLoader 批次捕获中间激活值和计算显著图
    print("\n批次捕获中间激活值和计算显著图...")
    saliency_list = []
    
    for batch_idx, batch in enumerate(dataloader):
        if has_cog:
            eeg_x, cog_x = batch
            eeg_x = eeg_x.to(device)
            cog_x = cog_x.to(device)
        else:
            eeg_x = batch[0].to(device)
            cog_x = None
        
        # 累积捕获中间激活值(第一个批次重置缓存，后续批次累积)
        with interpreter.capture(eeg_x, cog_x, accumulate=(batch_idx > 0)):
            pass
        
        # 计算当前批次的显著图
        saliency = interpreter.compute_saliency(eeg_x, cog_x, abs=False)
        saliency_list.append(saliency)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  已处理 {batch_idx + 1}/{len(dataloader)} 批次")
    
    # 合并所有批次的显著图
    saliency_all = np.concatenate(saliency_list, axis=0)
    print(f"显著图合并完成，shape: {saliency_all.shape}")
    
    # ① Saliency Map 头皮地形图(均值 + 各频带)
    interpreter.plot_attribution_topo(
        saliency_all, band_idx='all+',
        title='Saliency Map',
        save_path=_sp('saliency_topo.svg'),
    )

    # ② 脑区连接圆图(均值 + 各频带)
    if interpreter.gcn_adjs:
        interpreter.plot_area_connectivity(
            band_idx='all+',
            title='脑区连接图',
            save_path=_sp('area_connectivity.svg'),
        )

    # ③ 脑区归一化邻接矩阵热图(均值 + 各频带)
    if interpreter.gcn_adjs:
        interpreter.plot_gcn_adjacency(
            band_idx='all+',
            title='脑区邻接矩阵',
            save_path=_sp('gcn_adjacency.svg'),
        )

    # ④ 频带重要性条形图
    if interpreter.band_attn_weights:
        interpreter.plot_band_importance(
            title='频带重要性',
            save_path=_sp('band_importance.svg'),
        )

    # ⑤ EEG deep 自注意力权重热图
    if interpreter.deep_attn_weights:
        interpreter.plot_temporal_attention(
            attn_type='deep',
            title='时频图自注意力权重',
            save_path=_sp('deep_attn.svg'),
        )

    # ⑥ Cognitive 自注意力权重热图
    if interpreter.cog_attn_weights:
        interpreter.plot_temporal_attention(
            attn_type='cog',
            title='Cognitive 自注意力权重',
            save_path=_sp('cog_attn.svg'),
        )

    # ⑦ 跨模态注意力权重热图
    if interpreter.cross_attn_weights:
        interpreter.plot_cross_attention(
            title='DE-认知跨模态注意力权重',
            save_path=_sp('cross_attn.svg'),
        )
        interpreter.plot_cross_attention_strength(
            mode='col',
            title='DE-认知跨模态时间关注强度',
            save_path=_sp('cross_attn_strength_col.svg'),
        )

    return interpreter


def _loso_collect_data(args, test_sub_id, batch_size: int = 64):
    """Load only the held-out subject, applying the saved fold normalization."""
    import json
    from pathlib import Path
    from datapipe import create_datapipe
    from cross_validation import sub_internal_standardize
    from utils.preprocess import numpy2tensor
    from torch.utils.data import TensorDataset, DataLoader

    datapipe = create_datapipe(args)
    if not datapipe.is_prepared():
        raise ValueError("Prepare the dataset with main.py before interpretation")
    records_dir = Path(args.training.records_dir)
    with (records_dir / f"loso_sub_{test_sub_id}_fold.json").open(encoding="utf-8") as f:
        fold = json.load(f)
    data, _ = datapipe.load_one_sub(test_sub_id)
    if args.dataset.online_transform == "sub_internal":
        data = sub_internal_standardize(data)
    else:
        with np.load(records_dir / fold["normalization_file"], allow_pickle=False) as norm:
            for feat in args.model.features:
                key = f"{feat}__mean"
                if key in norm:
                    data[feat] = (data[feat] - norm[key]) / norm[f"{feat}__std"]
                elif args.dataset.online_transform in {"channel_wise", "zscore"}:
                    raise ValueError(f"Missing saved normalization for {feat}")
    tensors = tuple(numpy2tensor(data[f], dtype=torch.float32) for f in args.model.features)
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), len(tensors) == 2


def main(config_path: str, ckpt_path: str, test_sub_id: int, save_dir: Optional[str] = None):
    """
    一键分析脚本入口：加载配置、收集所有被试数据、加载模型权重并输出全套可视化图。

    Args:
        config_path: yaml配置文件路径
        ckpt_path: 模型权重文件路径(.pth)，在 ``if __name__ == '__main__'`` 块中指定。
        save_dir:  图表输出目录，在 ``if __name__ == '__main__'`` 块中指定，
                   默认为项目根目录下的 ``test_viz_out``。
    """
    import sys
    import os
    import platform
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils import load_config
    from utils.tool import seed_everything
    from net.util import create_model

    # ── 1. 加载配置 ──────────────────────────────────────────────────────────
    args = load_config(Path(config_path))
    seed_everything(args.reproduce.random_seed, args.reproduce.deterministic)

    # ── 2. 字体配置 ──────────────────────────────────────────────────────────
    chinese_font = "Songti SC" if platform.system() == "Darwin" else "SimSun"
    plt.rcParams["font.family"] = ["DejaVu Sans", chinese_font]
    matplotlib.rcParams["axes.unicode_minus"] = False

    DEVICE = torch.device(args.training.device)
    DATASET = args.dataset.name
    GRAPH_TYPE = args.dataset.graph_type
    BAND_NAMES = list(args.dataset.freq_bands.keys())

    print(f"Device   : {DEVICE}")
    print(f"Dataset  : {DATASET}  |  Graph: {GRAPH_TYPE}")
    print(f"Bands    : {BAND_NAMES}")
    print(f"Checkpoint: {ckpt_path}")

    # ── 3. 加载留出被试及已保存的归一化参数 ───────────────────────────────────
    print("\n加载留出被试及已保存的归一化参数...")
    batch_size = args.training.batch_size  # 从配置读取 batch_size
    dataloader, has_cog = _loso_collect_data(args, test_sub_id=test_sub_id, batch_size=batch_size)

    print(f"DataLoader 创建完成，批次大小: {batch_size}")
    print(f"总批次数: {len(dataloader)}")
    print(f"是否包含认知特征: {has_cog}")

    # ── 4. 加载模型权重 ───────────────────────────────────────────────────────
    model = create_model(args)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt, strict=True)
    model = model.to(DEVICE).eval()
    print(f"\n成功加载模型: {ckpt_path}")

    # ── 5. 一键出图 ───────────────────────────────────────────────────────────
    if save_dir is None:
        save_dir = "./viz_out"
    print(f"\n开始生成可视化图表，保存至: {save_dir}")

    analyze_model(
        model=model,
        dataloader=dataloader,
        has_cog=has_cog,
        dataset=DATASET,
        graph_type=GRAPH_TYPE,
        band_names=BAND_NAMES,
        save_dir=save_dir,
        device=DEVICE,
    )

    print("\n✓ 所有图表生成完毕。")


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from utils import load_config

    parser = argparse.ArgumentParser(description="Interpret a trained ECFNet held-out fold")
    parser.add_argument("--config", default="config/index-strict.yaml")
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--output", default="viz_out")
    cli = parser.parse_args()
    args = load_config(Path(cli.config))
    with (args.training.records_dir / f"loso_sub_{cli.subject}_fold.json").open(encoding="utf-8") as f:
        fold = json.load(f)
    main(cli.config, str(args.training.save_dir / fold["checkpoint"]), cli.subject, cli.output)
