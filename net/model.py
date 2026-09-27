import torch
import torch.nn as nn
import torch.nn.functional as F

from net.layer import SGConv, GlobalAttentionPooling, MultiheadAttention, PositionWiseFFN


class LocalGraphAggregate(nn.Module):
    """局部图特征聚合

    Args:
        area_nodes (list[int]): 每个局部区域的节点数
    """

    def __init__(self, area_nodes):
        super().__init__()

        self.area_nodes = area_nodes
        self.num_ares = len(area_nodes)
        self.num_nodes = sum(area_nodes)

    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入特征, shape=(batch_size, num_nodes, in_features)
        Returns:
            torch.Tensor: 聚合后的特征, shape=(batch_size, num_areas, in_features)
        """
        assert x.size(1) == self.num_nodes, f"输入节点数 {x.size(1)} 与定义的节点数 {self.num_nodes} 不匹配"

        areas = torch.split(x, self.area_nodes, dim=1)
        out = [area.mean(dim=1) for area in areas]
        return torch.stack(out, dim=1)

    def __repr__(self):
        return f"LocalGraphAggregate(area_nodes={self.area_nodes})"


class LocalGraphEmbedding(nn.Module):
    """局部图嵌入

    Args:
        in_features (int): 输入特征维度
        hidden_features (int): 隐藏层特征维度
        out_features (int): 输出特征维度
        area_nodes (list[int]): 每个局部区域的节点数
        dropout (float): Dropout 概率
    """

    def __init__(self, in_features, hidden_features, out_features, area_nodes, dropout=0.0):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.area_nodes = area_nodes
        self.num_ares = len(area_nodes)
        self.num_nodes = sum(area_nodes)

        self.proj1 = nn.Linear(in_features, hidden_features)
        # Eq. (4): one scalar per electrode, broadcast over P1's 64 outputs.
        self.weights = nn.Parameter(torch.FloatTensor(self.num_nodes, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.FloatTensor(self.num_nodes, 1), requires_grad=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj2 = nn.Linear(hidden_features, out_features)
        self.aggregate = LocalGraphAggregate(area_nodes) if self.num_ares != self.num_nodes else nn.Identity()

        nn.init.xavier_normal_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """前向传播
        Args:
            x (torch.Tensor): 输入特征, shape=(batch_size, num_nodes, in_features)
        Returns:
            torch.Tensor: 输出特征, shape=(batch_size, num_areas, out_features)
        """
        out = self.proj1(x)  # (b, nodes, hidden_features)
        out = self.relu(torch.mul(out, self.weights) - self.bias)
        out = self.drop(out)
        out = self.proj2(out)  # (b, nodes, out_features)
        out = self.aggregate(out)  # (b, areas, out_features)
        return out


class GlobalGraphConv(nn.Module):
    """全局图卷积
    Args:
        num_areas (int): 脑区数量
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        K (int): 图卷积传播次数
    """

    def __init__(self, num_areas, in_features, out_features, K):
        super().__init__()
        self.num_areas = num_areas
        self.in_features = in_features
        self.out_features = out_features
        self.K = K

        # Eq. (5): the learned symmetric modulation of sample similarity.
        self.attn_mask = nn.Parameter(torch.FloatTensor(num_areas, num_areas), requires_grad=True)
        nn.init.xavier_normal_(self.attn_mask)

        self.ln1 = nn.LayerNorm(in_features)
        self.conv = SGConv(in_features, out_features, K, add_self_loops=True)
        self.ln2 = nn.LayerNorm(out_features)

    def forward(self, x):
        """前向传播
        Args:
            x (torch.Tensor): 输入特征, shape=(batch_size, num_areas, in_features)
        Returns:
            torch.Tensor: 输出特征, shape=(batch_size, num_areas, out_features)
        """
        adj = self.get_adjacency(x)
        x = self.ln1(x)
        return self.ln2(self.conv(x, adj))

    def get_adjacency(self, x):
        """计算邻接矩阵
        Args:
            x (torch.Tensor): 输入特征, shape=(batch_size, num_areas, in_features)
        Returns:
            torch.Tensor: 邻接矩阵, shape=(batch_size, num_areas, num_areas)
        """
        # 计算自相似度矩阵
        adj = torch.bmm(x, x.permute(0, 2, 1))  # (b, n, n)
        # Eq. (5) uses M + M^T, without division by two.
        weights = self.attn_mask + self.attn_mask.T
        # 计算最终的邻接矩阵
        return F.relu(adj * weights)


class GlobalGraphEmbedding(nn.Module):
    """全局图嵌入

    Args:
        num_areas (int): 脑区数量
        in_features (int): 输入特征维度
        hidden_features (int): 隐藏层特征维度
        out_features (int): 输出特征维度
        conv_K (list[int]): 不同尺度的图卷积传播次数列表
        dropout (float): Dropout 概率
    """

    def __init__(self, num_areas, in_features, hidden_features, out_features, conv_K, dropout=0.0):
        super().__init__()

        self.num_areas = num_areas
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.conv_K = conv_K
        self.dropout = dropout
        self.num_layers = len(conv_K)

        # 直接映射
        self.linear = nn.Sequential(
            nn.Linear(num_areas * in_features, hidden_features),
            nn.ReLU()
        )
        # 多尺度图卷积
        self.layers = nn.ModuleList([
            nn.Sequential(
                GlobalGraphConv(num_areas, in_features, hidden_features, K),
                nn.ReLU(),
                GlobalAttentionPooling(hidden_features)
            ) for K in conv_K
        ])
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        """前向传播
        Args:
            x (torch.Tensor): 输入特征, shape=(batch_size, num_areas, in_features)
        Returns:
            torch.Tensor: 输出特征, shape=(batch_size, out_features)
        """
        # 残差分支, 直接线性映射
        x_ = x.reshape(x.size(0), -1)  # (b, num_areas * in_features)
        out = [self.linear(x_)]  # (b, hidden_features)

        # 多尺度图卷积分支
        for layer in self.layers:
            out.append(layer(x))  # (b, hidden_features)

        # 堆叠所有分支输出
        out = torch.stack(out, dim=1)  # (b, num_layers + 1, hidden_features)
        out = out.mean(dim=1)  # (b, hidden_features)

        out = self.drop(out)
        return self.proj(out)  # (b, out_features)


class SelfAttentionBlock(nn.Module):
    """自注意力块

    Args:
        dim (int): 输入特征维度
        num_heads (int): 注意力头数
        head_dim (int): 每个注意力头的维度
        ffn_dim (int): 前馈网络隐藏层维度
        dropout (float): Dropout 概率
    """

    def __init__(self, dim, num_heads, head_dim, ffn_dim, dropout=0.0):
        super().__init__()
        self.attention = MultiheadAttention(dim, num_heads, head_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = PositionWiseFFN(dim, ffn_dim, dim, dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """前向传播
        Args:
            x (torch.Tensor): 输入特征, shape=(batch_size, seqs, dim)
        Returns:
            torch.Tensor: 输出特征, shape=(batch_size, seqs, dim)
        """
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.norm1(x + attn_out)
        ffn_out = self.ffn(attn_out)
        out = self.norm2(attn_out + ffn_out)
        return out


class CrossAttentionBlock(nn.Module):
    """交叉注意力块

    Args:
        dim (int): QKV的特征维度
        num_heads (int): 注意力头数
        head_dim (int): 每个注意力头的维度
        ffn_dim (int): 前馈网络隐藏层维度
        dropout (float): Dropout 概率
    """

    def __init__(self, dim, num_heads, head_dim, ffn_dim, dropout=0.0):
        super().__init__()
        self.attention = MultiheadAttention(dim, num_heads, head_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = PositionWiseFFN(dim, ffn_dim, dim, dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x_q, x_kv):
        """前向传播
        Args:
            x_q (torch.Tensor): 查询特征, shape=(batch_size, seqs, dim)
            x_kv (torch.Tensor): 键值特征, shape=(batch_size, seqs, dim_kv)
        Returns:
            torch.Tensor: 输出特征, shape=(batch_size, seqs, dim)
        """
        attn_out, _ = self.attention(x_q, x_kv, x_kv)
        attn_out = self.norm1(x_q + attn_out)
        ffn_out = self.ffn(attn_out)
        out = self.norm2(attn_out + ffn_out)
        return out


class CogGateFusion(nn.Module):
    """认知和EEG深度学习特征门控融合"""

    def __init__(self, deep_dim, cog_dim, hidden_dim, out_dim):
        super().__init__()
        self.deep_dim = deep_dim
        self.cog_dim = cog_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.norm_deep = nn.LayerNorm(deep_dim)
        self.norm_cog = nn.LayerNorm(cog_dim)
        
        self.proj_cog = nn.Sequential(
            nn.Linear(cog_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate_layer = nn.Sequential(
            nn.Linear(deep_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.output_layer = nn.Linear(deep_dim, out_dim)

    def forward(self, x_deep, x_cog):
        """前向传播
        Args:
            x_deep (torch.Tensor): 深度学习特征, shape=(batch_size, seq_dim, deep_dim)
            x_cog (torch.Tensor): 认知特征, shape=(batch_size, seq_dim, cog_dim)
        Returns:
            torch.Tensor: 融合后的特征, shape=(batch_size, seq_dim, out_dim)
        """
        x_deep = self.norm_deep(x_deep)
        x_cog = self.norm_cog(x_cog)

        cog_feat = self.proj_cog(x_cog)  # (b, seq, hidden_dim)

        combined = torch.cat([x_deep, cog_feat], dim=-1)  # (b, seq, deep_dim + hidden_dim)
        gate = self.gate_layer(combined)  # (b, seq, hidden_dim)

        fused = gate * x_deep + (1 - gate) * cog_feat  # (b, seq, deep_dim)
        return self.output_layer(fused)  # (b, seq, out_dim)


class CogAttentionFusion(nn.Module):
    """认知和EEG深度学习特征注意力融合"""

    def __init__(self, deep_dim, cog_dim, attn_heads, head_dim, out_dim, dropout=0.0, use_self_attn=False, use_cross_attn=True):
        super().__init__()
        self.deep_dim = deep_dim
        self.cog_dim = cog_dim
        self.attn_heads = attn_heads
        self.head_dim = head_dim
        self.out_dim = out_dim
        self.use_self_attn = use_self_attn
        self.use_cross_attn = use_cross_attn
        self.proj_deep = nn.Linear(deep_dim, out_dim)
        self.proj_cog = nn.Linear(cog_dim, out_dim)

        if use_self_attn:
            self.attn_deep = SelfAttentionBlock(out_dim, attn_heads, head_dim, out_dim * 2, dropout)
            self.attn_cog = SelfAttentionBlock(out_dim, attn_heads, head_dim, out_dim * 2, dropout)
        else:
            self.attn_deep = nn.Identity()
            self.attn_cog = nn.Identity()

        if use_cross_attn:
            self.attention = CrossAttentionBlock(out_dim, attn_heads, head_dim, out_dim * 2, dropout)
        else:
            self.attention = None

    def forward(self, x_deep, x_cog):
        """前向传播
        Args:
            x_deep (torch.Tensor): 深度学习特征, shape=(batch_size, seq_dim, deep_dim)
            x_cog (torch.Tensor): 认知特征, shape=(batch_size, seq_dim, cog_dim)
        Returns:
            torch.Tensor: 融合后的特征, shape=(batch_size, seq_dim, out_dim)
        """
        x_deep = self.proj_deep(x_deep) # (b, seq, out_dim)
        x_cog = self.proj_cog(x_cog)    # (b, seq, out_dim)

        x_deep = self.attn_deep(x_deep)
        x_cog = self.attn_cog(x_cog)

        if self.attention is not None:
            # 交叉注意力融合
            attn_out = self.attention(x_deep, x_cog)  # (b, seq, out_dim)
        else:
            # 简单相加融合
            attn_out = x_deep + x_cog # (b, seq, out_dim)
        return attn_out  # (b, seq, out_dim)


class CogSimpleFusion(nn.Module):
    """认知和EEG深度学习特征简单融合"""

    def __init__(self, deep_dim, cog_dim, out_dim):
        super().__init__()
        self.deep_dim = deep_dim
        self.cog_dim = cog_dim
        self.out_dim = out_dim

        self.proj_deep = nn.Linear(deep_dim, out_dim)
        self.proj_cog = nn.Linear(cog_dim, out_dim)

    def forward(self, x_deep, x_cog):
        """前向传播
        Args:
            x_deep (torch.Tensor): 深度学习特征, shape=(batch_size, seq_dim, deep_dim)
            x_cog (torch.Tensor): 认知特征, shape=(batch_size, seq_dim, cog_dim)
        Returns:
            torch.Tensor: 融合后的特征, shape=(batch_size, seq_dim, out_dim)
        """
        x_deep = self.proj_deep(x_deep) # (b, seq, out_dim)
        x_cog = self.proj_cog(x_cog)    # (b, seq, out_dim)

        return x_deep + x_cog # (b, seq, out_dim)


class SpatialBandLearning(nn.Module):
    """空间-通道学习"""

    def __init__(self, num_bands, area_nodes, hidden_features, out_features, conv_K, dropout=0.0, share_encoders=False):
        super().__init__()

        self.num_bands = num_bands
        self.area_nodes = area_nodes
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.conv_K = conv_K
        self.dropout = dropout
        self.num_areas = len(area_nodes)
        self.share_encoders = share_encoders

        def _make_encoder():
            return nn.Sequential(
                LocalGraphEmbedding(1, hidden_features // 2, hidden_features, area_nodes, dropout),
                GlobalGraphEmbedding(self.num_areas, hidden_features, hidden_features, out_features, conv_K, dropout)
            )

        if share_encoders:
            self.shared_encoder = _make_encoder()
        else:
            self.graph_encoders = nn.ModuleList([_make_encoder() for _ in range(num_bands)])

        self.band_fusion = GlobalAttentionPooling(out_features)
    
    def forward(self, x):
        """前向传播

        Args:
            x (torch.Tensor): 输入特征, shape=(batches, seqs, channels, bands)
        """
        x = x.unsqueeze(-1)
        b, s, ch, band, _ = x.shape

        # (b, s, ch, band, 1) -> (b*s, ch, band, 1) -> (band, b*s, ch, 1)
        x = x.flatten(0, 1).permute(2, 0, 1, 3)

        if self.share_encoders:
            # 将 band 维并入 batch 维，一次前向完成所有频带
            x_merged = x.flatten(0, 1) # (band, b*s, ch, 1) -> (band*b*s, ch, 1)
            x_merged = self.shared_encoder(x_merged)  # (band*b*s, out_features)
            bands = x_merged.unflatten(0, [band, b * s]).permute(1, 0, 2)  # (b*s, band, out_features)
        else:
            # 逐频带前向
            bands = [encoder(xi) for xi, encoder in zip(x, self.graph_encoders)] # n_bands * (b*s, out_features)
            bands = torch.stack(bands, dim=1)  # (b*s, band, out_features)

        out = self.band_fusion(bands)  # (b*s, out_features)
        out = out.unflatten(0, [b, s])  # (b, s, out_features)
        return out


class MyModel(nn.Module):
    def __init__(self, num_bands, num_classes, area_nodes, attn_layers, attn_heads, conv_K, dropout=0.0, share_encoders=False):
        super().__init__()

        self.num_bands = num_bands
        self.num_classes = num_classes
        self.area_nodes = area_nodes
        self.conv_K = conv_K
        self.dropout = dropout
        self.num_areas = len(area_nodes)

        self.sb_learning = SpatialBandLearning(num_bands, area_nodes, 128, 64, conv_K, dropout, share_encoders)

        self.self_attn = nn.Sequential(*(
            SelfAttentionBlock(64, attn_heads, 64, 128, dropout)
            for _ in range(attn_layers)
        ))

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """前向传播

        Args:
            x (torch.Tensor): 输入特征, shape=(batches, seqs, channels, bands)
        """
        x = self.sb_learning(x)  # (b, s, 64)
        x = self.self_attn(x)  # (b, s, 64)
        x = x.mean(dim=1)  # (b, 64)
        out = self.classifier(x)  # (b, num_classes)
        return out


class MyModelCogFusion(nn.Module):
    """EEG原始特征+认知特征融合模型"""

    def __init__(self, num_bands, num_classes, area_nodes, attn_heads, conv_K, cog_dim=3, fusion='self+cross', dropout=0.0, share_encoders=False):
        super().__init__()

        self.num_bands = num_bands
        self.num_classes = num_classes
        self.area_nodes = area_nodes
        self.conv_K = conv_K
        self.cog_dim = cog_dim
        self.dropout = dropout
        self.num_areas = len(area_nodes)

        self.sb_learning = SpatialBandLearning(num_bands, area_nodes, 128, 64, conv_K, dropout, share_encoders)

        if fusion == 'gate':
            self.fusion_layer = CogGateFusion(deep_dim=64, cog_dim=cog_dim, hidden_dim=32, out_dim=64)
        elif fusion == 'simple':
            self.fusion_layer = CogSimpleFusion(deep_dim=64, cog_dim=cog_dim, out_dim=64)
        elif fusion in ['self+cross', 'self', 'cross']:
            self.fusion_layer = CogAttentionFusion(
                deep_dim=64,
                cog_dim=cog_dim,
                attn_heads=attn_heads,
                head_dim=16,
                out_dim=64,
                dropout=dropout,
                use_self_attn=False if fusion == 'cross' else True,
                use_cross_attn=False if fusion == 'self' else True
            )
        else:
            raise ValueError(f"未知的融合方式: {fusion}")

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, inputs):
        """前向传播

        Args:
            inputs (tuple[torch.Tensor, torch.Tensor]):
                - eeg_x (torch.Tensor): 低阶EEG特征, shape=(batches, seqs, channels, bands)
                - cog_x (torch.Tensor): 高阶认知特征, shape=(batches, seqs, cog_dim)
        """
        eeg_x, cog_x = inputs
        eeg_x = self.sb_learning(eeg_x)  # (b, s, 64)
        x = self.fusion_layer(eeg_x, cog_x)  # (b, s, 64)
        x = x.mean(dim=1)  # (b, 64)
        out = self.classifier(x)  # (b, num_classes)
        return out


class ECFNet(nn.Module):
    """ECFNet, Eqs. (4)-(10) of the accompanying paper.

    Input layout is (batch, windows, electrodes, DE bands), plus
    (batch, windows, 3) fixed wavelet descriptors. Returns class logits.
    There are no positional embeddings. Each band owns an encoder unless
    the explicitly labeled shared-encoder ablation is enabled.
    """

    def __init__(self, num_bands, num_classes, area_nodes, K=2,
                 attn_heads=4, head_dim=16, fusion="self+cross",
                 dropout=0.5, share_encoders=False, use_descriptors=True):
        super().__init__()
        if K < 1 or attn_heads < 1 or head_dim < 1:
            raise ValueError("K and attention dimensions must be positive")
        if fusion not in {"addition", "cross", "self+addition", "self+cross"}:
            raise ValueError(f"Unsupported ECFNet fusion: {fusion}")
        if not area_nodes or any(n < 1 for n in area_nodes):
            raise ValueError("area_nodes must contain positive region sizes")
        self.num_bands = num_bands
        self.num_nodes = sum(area_nodes)
        self.use_descriptors = use_descriptors
        self.sb_learning = SpatialBandLearning(
            num_bands, area_nodes, 128, 64, [K], dropout, share_encoders
        )
        if use_descriptors:
            self.fusion_layer = CogAttentionFusion(
                64, 3, attn_heads, head_dim, 64, dropout,
                use_self_attn=fusion in {"self+addition", "self+cross"},
                use_cross_attn=fusion in {"cross", "self+cross"},
            )
        else:
            self.graph_projection = nn.Linear(64, 64)
            self.graph_attention = SelfAttentionBlock(64, attn_heads, head_dim, 128, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, num_classes),
        )

    def forward(self, inputs):
        if self.use_descriptors:
            if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
                raise ValueError("ECFNet expects (DE, wavelet descriptors)")
            eeg, descriptors = inputs
            if descriptors.ndim != 3 or descriptors.shape[-1] != 3:
                raise ValueError("Descriptor layout must be (batch, windows, 3)")
            if eeg.shape[0] != descriptors.shape[0]:
                raise ValueError("Both streams must have the same batch size")
        else:
            eeg = inputs
        if eeg.ndim != 4 or tuple(eeg.shape[-2:]) != (self.num_nodes, self.num_bands):
            raise ValueError("DE layout must be (batch, windows, electrodes, bands)")
        graph = self.sb_learning(eeg)
        if self.use_descriptors:
            fused = self.fusion_layer(graph, descriptors)
        else:
            fused = self.graph_attention(self.graph_projection(graph))
        return self.classifier(fused.mean(dim=1))


if __name__ == "__main__":
    from net.tool import count_parameters

    eeg = torch.randn(32, 10, 24, 5)
    model = MyModel(num_bands=5, num_classes=3, area_nodes=[6, 6, 6, 6], attn_layers=2, attn_heads=4, conv_K=[1, 2], dropout=0.2)
    print(model)
    print("模型参数量: {:.2f}M".format(count_parameters(model) / 1e6))
    out = model(eeg)
    print(out.shape)  # (32, 10, 256)
