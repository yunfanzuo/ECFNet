import torch
import torch.nn as nn
import torch.nn.functional as F


class SGConv(nn.Module):
    """简化图卷积层

    Args:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        K (int): 传播次数
        add_self_loops (bool): 是否添加自环
        bias (bool): 是否使用偏置
    """

    def __init__(self, in_features, out_features, K=1, add_self_loops=True, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.add_self_loops = add_self_loops

        # 线性变换层
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        """前向传播
        Args:
            x (torch.Tensor): 输入特征, shape=(batches, nodes, in_features)
            adj (torch.Tensor): 邻接矩阵, shape=(batches, nodes, nodes)
        Returns:
            torch.Tensor: 输出特征, shape=(batches, nodes, out_features)
        Notes:
            需要确保邻接矩阵 ``adj`` 为对称矩阵(无向图)
        """
        # 检查设备一致性
        assert x.device == adj.device, "x and adj must be on the same device"
        # 检查节点数一致性
        assert x.size(1) == adj.size(1) == adj.size(2), "x and adj must have the same number of nodes"

        # 获取归一化邻接矩阵, shape=(batches, nodes, nodes)
        adj = SGConv.norm_adj(adj, add_self_loops=self.add_self_loops)

        # 进行K次传播
        out = x
        for _ in range(self.K):
            out = torch.bmm(adj, out)

        # 线性变换
        out = self.lin(out)
        return out

    @staticmethod
    def norm_adj(adj, add_self_loops=True, eps=1e-12):
        if add_self_loops:
            num_nodes = adj.size(-1)
            identity = torch.eye(num_nodes, dtype=adj.dtype, device=adj.device)
            adj = adj + identity

        # 度向量, shape=(batches, nodes)
        deg = adj.sum(dim=-1)
        if not add_self_loops:  # 如果没有添加自环, 确保度不为0
            deg = deg.clamp_min(eps)

        # 度的逆平方根向量, shape=(batches, nodes)
        d_inv_sqrt = deg.pow(-0.5)

        # 广播计算归一化邻接矩阵, 等价D^{-1/2}·A·D^{-1/2}, shape=(batches, nodes, nodes)
        return adj * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_features={self.in_features}, '
                f'out_features={self.out_features}, K={self.K}, '
                f'add_self_loops={self.add_self_loops})')


class GlobalAttentionPooling(nn.Module):
    """全局注意力池化

    Args:
        in_features (int): 输入特征维度
    """

    def __init__(self, in_features):
        super().__init__()

        self.in_features = in_features

        # 全局查询向量
        self.query = nn.Parameter(torch.FloatTensor(1, 1, in_features))
        nn.init.xavier_normal_(self.query)
        # 缩放因子
        self.scale = in_features ** -0.5

    def forward(self, x):
        """前向传播
        Args:
            x (torch.Tensor): 输入特征, shape=(B, N, D)
        Returns:
            torch.Tensor: 池化后特征, shape=(B, D)
        Notes:
            - ``B``: batches, 批量大小
            - ``N``: context nums, 上下文(节点/频带/时间步等)数量
            - ``D``: feature dim, 特征维度, 需与``in_features``一致
        """
        # 计算注意力分数, Q·X^T / sqrt(d)
        scores = self.query @ x.transpose(-1, -2)  # (B, 1, N), 1 for score of each context
        scores = scores * self.scale

        # 归一化注意力权重
        attn = F.softmax(scores, dim=-1)

        # 加权池化
        pooled = attn @ x  # (B, 1, D)

        return pooled.squeeze(1)  # (B, D)

    def __repr__(self):
        return f'{self.__class__.__name__}(in_features={self.in_features})'


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, scale=None):
    r"""缩放点积注意力
    Args:
        query (torch.Tensor): 查询张量, shape=(N, ..., H, L, E)
        key (torch.Tensor): 键张量, shape=(N, ..., H, S, E)
        value (torch.Tensor): 值张量, shape=(N, ..., H, S, E)
        attn_mask (torch.Tensor | None): 注意力掩码, shape=(N, ..., L, S)
        dropout_p (float): 注意力权重的dropout概率
        scale (float | None): 缩放因子, 默认为 None, 则使用 :math:`\frac{1}{\sqrt{Ek}}`
    Returns:
        tuple[torch.Tensor, torch.Tensor]: 输出张量和注意力权重
            - ``out``: 输出张量, shape=(N, ..., H, L, E)
            - ``attn_weight``: 注意力权重, shape=(N, ..., H, L, S)
    Notes:
        - ``N``: batches, 批量大小
        - ``H``: heads, 注意力头数
        - ``L``: target len, 目标序列长度
        - ``S``: source len, 源序列长度
        - ``E``: embed dim, 嵌入维度
    """
    E, L, S = query.size(-1), query.size(-2), key.size(-2)
    scale_factor = E ** -0.5 if scale is None else scale
    bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            bias = attn_mask + bias

    score = query @ key.transpose(-2, -1) * scale_factor + bias
    attn_weight = F.softmax(score, dim=-1)
    if dropout_p > 0:
        out = F.dropout(attn_weight, p=dropout_p, training=True)
    else:
        out = attn_weight
    out = out @ value
    return out, attn_weight


class MultiheadAttention(nn.Module):
    """简化的多头注意力模块

    Args:
        embed_dim (int): 嵌入维度
        num_heads (int): 注意力头数
        head_dim (int): 每个注意力头的嵌入维度
        bias (bool): 是否使用偏置
        dropout (float): 注意力权重的dropout概率

    Notes:
        - Q, K, V 拥有相同的嵌入维度
        - 不考虑注意力掩码
    """

    def __init__(self, embed_dim, num_heads, head_dim, bias=True, dropout=0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bias = bias
        self.dropout = dropout

        # 组合所有注意力头的总嵌入维度
        hidden_dim = num_heads * head_dim

        # QKV输入投影
        self.qkv_proj = nn.Linear(embed_dim, 3 * hidden_dim, bias=bias)
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, need_weights=False):
        """前向传播
        Args:
            query (torch.Tensor): 查询张量, shape=(B, L, D)
            key (torch.Tensor): 键张量, shape=(B, S, D)
            value (torch.Tensor): 值张量, shape=(B, S, D)
            need_weights (bool): 是否返回注意力权重
        Returns:
            tuple[torch.Tensor, torch.Tensor|None]: 输出张量和注意力权重
                - ``attn_output``: 输出张量, shape=(B, L, D)
                - ``attn_weights``: 注意力权重, shape=(B, H, L, S), 仅当need_weights为True时返回, 否则为None
        Notes:
            - ``B``: batches, 批量大小
            - ``L``: target len, 目标序列长度
            - ``S``: source len, 源序列长度
            - ``D``: embed dim, 嵌入维度
        """
        H, Hd = self.num_heads, self.head_dim

        # Step 1. QKV 投影
        if query is key and key is value:
            # 自注意力
            qkv = self.qkv_proj(query)  # (B, L, 3*H*Hd)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # 非自注意力
            q_weight, k_weight, v_weight = self.qkv_proj.weight.chunk(3, dim=0)
            q_bias, k_bias, v_bias = self.qkv_proj.bias.chunk(3, dim=0) if self.bias else (None, None, None)
            q, k, v = (
                F.linear(query, q_weight, q_bias),
                F.linear(key, k_weight, k_bias),
                F.linear(value, v_weight, v_bias)
            )

        # Step 2. 重塑为多头形状
        # (B, L, H*Hd) -> (B, L, H, Hd) -> (B, H, L, Hd)
        q = q.unflatten(-1, [H, Hd]).transpose(1, 2)
        # (B, S, H*Hd) -> (B, S, H, Hd) -> (B, H, S, Hd)
        v = v.unflatten(-1, [H, Hd]).transpose(1, 2)
        k = k.unflatten(-1, [H, Hd]).transpose(1, 2)

        # Step 3. 计算缩放点积注意力
        # outputs: (B, H, L, Hd), weights: (B, H, L, S)
        dropout_p = self.dropout if self.training else 0.0
        if need_weights:
            attn_output, attn_weights = scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            # use PyTorch built-in function for efficiency
            attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            attn_weights = None
        # (B, H, L, Hd) -> (B, L, H, Hd) -> (B, L, H*Hd)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. 应用输出投影
        # (B, L, H*Hd) -> (B, L, D)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class PreNorm(nn.Module):
    """前归一化包装器

    Args:
        dim (int): 归一化维度
        module (nn.Module): 被包装的模块
    """

    def __init__(self, dim, module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.module = module

    def forward(self, x, **kwargs):
        """前向传播
        Args:
            x (torch.Tensor): 输入张量
            **kwargs: 传递给被包装模块的其他参数
        Returns:
            torch.Tensor: 输出张量
        """
        return self.module(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    """后归一化包装器

    Args:
        dim (int): 归一化维度
        module (nn.Module): 被包装的模块
    """

    def __init__(self, dim, module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.module = module

    def forward(self, x, **kwargs):
        """前向传播
        Args:
            x (torch.Tensor): 输入张量
            **kwargs: 传递给被包装模块的其他参数
        Returns:
            torch.Tensor: 输出张量
        """
        return self.norm(self.module(x, **kwargs))


class PositionWiseFFN(nn.Module):
    """逐位前馈网络

    Args:
        in_features (int): 输入特征维度
        hidden_features (int): 隐藏层特征维度
        dropout (float): Dropout 概率
    """

    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        """前向传播
        Args:
            x (torch.Tensor): 输入特征，shape=(..., in_features)
        Returns:
            torch.Tensor: 输出特征，shape=(..., out_features)
        """
        return self.net(x)
