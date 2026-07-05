import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max


# 辅助函数：L2 Norm (带 epsilon 保护)
def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


# 辅助函数：Tuple拼接
def tuple_cat(*args, dim=-1):
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


# 辅助函数：稀疏 Softmax
def sparse_softmax(src, index, num_nodes=None):
    """
    针对稀疏图的 Softmax 操作
    src: [E, Heads] 注意力分数
    index: [E] 目标节点索引
    """
    # 1. 数值稳定性: 减去最大值
    max_val = scatter_max(src, index, dim=0, dim_size=num_nodes)[0]  # [N, Heads]
    centered_src = src - max_val[index]

    # 2. Exp
    src_exp = torch.exp(centered_src)

    # 3. Sum
    src_sum = scatter_add(src_exp, index, dim=0, dim_size=num_nodes)  # [N, Heads]

    # 4. Div (加 eps 防止除零)
    return src_exp / (src_sum[index] + 1e-12)


class GVP(nn.Module):
    """
    几何向量感知机 (Geometric Vector Perceptron)
    忠实复现原始论文逻辑，包含 Vector-to-Scalar 通信。
    """

    def __init__(
            self,
            in_dims,
            out_dims,
            h_dim=None,
            activations=(F.relu, torch.sigmoid),
            vector_gate=True,
    ):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        self.scalar_act, self.vector_act = activations

        # 1. 向量通道处理
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            # Wh: V_in -> V_hidden
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            # Wv: V_hidden -> V_out
            self.wv = nn.Linear(self.h_dim, self.vo, bias=False) if self.vo else None

            # 向量门控权重
            if self.vector_gate and self.vo:
                self.wsv = nn.Linear(self.so, self.vo)

        # 2. 标量通道处理
        # 关键点: 输入维度是 si + h_dim (Vector-to-Scalar 通信)
        # 如果有向量输入，我们会把向量模长拼接到标量里
        scalar_in_dim = self.si + (self.h_dim if self.vi else 0)
        self.ws = nn.Linear(scalar_in_dim, self.so)

    def forward(self, x):
        # x 是 tuple (s, V) 或 单个 s
        if self.vi:
            s, v = x
            # v: [..., vi, 3] -> transpose -> [..., 3, vi]
            v = torch.transpose(v, -1, -2)

            # V_hidden: [..., 3, h_dim]
            vh = self.wh(v)

            # 【核心创新点】Vector-to-Scalar 通信
            # 计算向量模长: [..., h_dim]
            vn = _norm_no_nan(vh, axis=-2)
            # 拼接到标量: [..., si + h_dim]
            s_in = torch.cat([s, vn], -1)

            # 标量输出
            s_out = self.ws(s_in)

            # 向量输出
            if self.vo:
                # V_out: [..., 3, vo] -> transpose -> [..., vo, 3]
                v_out = self.wv(vh)
                v_out = torch.transpose(v_out, -1, -2)

                # 【核心创新点】Vector Gating
                if self.vector_gate:
                    # 用标量输出来控制向量缩放
                    gate = self.wsv(self.scalar_act(s_out)) if self.scalar_act else self.wsv(s_out)
                    v_out = v_out * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v_out = v_out * self.vector_act(_norm_no_nan(v_out, axis=-1, keepdims=True))
            else:
                v_out = torch.zeros(s_out.shape[0], 0, 3, device=s_out.device)
        else:
            # 只有标量输入
            s = x
            s_out = self.ws(s)
            if self.vo:
                v_out = torch.zeros(s_out.shape[0], self.vo, 3, device=s_out.device)
            else:
                v_out = None

        if self.scalar_act:
            s_out = self.scalar_act(s_out)

        # 数值稳定性保护
        s_out = torch.nan_to_num(s_out, nan=0.0, posinf=1e5, neginf=-1e5)
        if v_out is not None:
            v_out = torch.nan_to_num(v_out, nan=0.0, posinf=1e5, neginf=-1e5)

        return (s_out, v_out) if self.vo else s_out


class LayerNorm(nn.Module):
    """
    支持 Tuple (s, V) 的层归一化
    """

    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        # 向量归一化: 除以平均模长
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        # 加上 eps 防止除零
        vn = vn + 1e-8
        return self.scalar_norm(s), v / vn


class Dropout(nn.Module):
    """
    支持 Tuple (s, V) 的 Dropout
    向量通道是整体 Dropout (整个向量要么丢弃要么保留)
    """

    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.sdropout(x)
        s, v = x
        s = self.sdropout(s)

        # 向量 Dropout
        if self.training and self.drop_rate > 0:
            # 生成 mask: [N, v_dim, 1]
            mask_shape = v.shape[:-1] + (1,)
            mask = torch.bernoulli((1 - self.drop_rate) * torch.ones(mask_shape, device=v.device))
            v = mask * v / (1 - self.drop_rate)

        return s, v


class GVPConvLayer(nn.Module):
    """
    带注意力的 GVP 图卷积层 (Attention-Enhanced GVP Graph Convolution)
    [已修复 FeedForward 维度传递 Bug]
    """

    def __init__(
            self,
            node_dims,
            edge_dims,
            n_message=3,
            n_feedforward=2,
            drop_rate=0.1,
            activations=(F.relu, torch.sigmoid),
            vector_gate=True,
            residual=True,
            attention=True
    ):
        super(GVPConvLayer, self).__init__()
        self.si, self.vi = node_dims
        self.se, self.ve = edge_dims
        self.n_message = n_message
        self.n_feedforward = n_feedforward
        self.attention = attention

        # 1. 消息传递函数 (Message Function)
        msg_func_layers = []
        # 追踪当前维度
        cur_dims = (2 * self.si + self.se, 2 * self.vi + self.ve)

        for i in range(n_message):
            gout = node_dims
            is_last = (i == n_message - 1)
            acts = (None, None) if is_last else activations

            msg_func_layers.append(GVP(cur_dims, gout, activations=acts, vector_gate=vector_gate))
            cur_dims = gout  # 更新下一层的输入维度

        self.message_func = nn.Sequential(*msg_func_layers)

        # 2. 注意力机制
        if self.attention:
            att_in_dim = self.si + 1
            self.att_score = nn.Sequential(
                nn.Linear(att_in_dim, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 1)
            )

        # 3. 归一化与 Dropout
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        # 4. 前馈网络 (FeedForward) - 【关键修正区域】
        ff_func_layers = []
        cur_dims = node_dims  # 初始输入是 node_dims

        for i in range(n_feedforward):
            # 中间层维度放大 (4x Scalar, 2x Vector)
            if i < n_feedforward - 1:
                gout = (4 * self.si, 2 * self.vi)
                acts = activations
            else:
                gout = node_dims
                acts = (None, None)

            ff_func_layers.append(GVP(cur_dims, gout, activations=acts, vector_gate=vector_gate))
            cur_dims = gout  # 【修正】：正确更新下一层的输入维度

        self.ff_func = nn.Sequential(*ff_func_layers)
        self.residual = residual

    def forward(self, x, edge_index, edge_attr):
        # ... (forward 函数代码保持不变，无需修改) ...
        # 为方便起见，这里完整列出 forward 以免您还需要去拼凑
        src, dst = edge_index
        s, v = x
        s_e, v_e = edge_attr

        # 1. Message Construction
        s_j, v_j = s[src], v[src]
        s_i, v_i = s[dst], v[dst]
        s_msg_in = torch.cat([s_j, s_e, s_i], dim=-1)
        v_msg_in = torch.cat([v_j, v_e, v_i], dim=-2)
        s_msg, v_msg = self.message_func((s_msg_in, v_msg_in))

        # 2. Attention
        if self.attention:
            v_mag = _norm_no_nan(v_msg, axis=-2, keepdims=False)
            v_mag_mean = v_mag.mean(dim=-1, keepdim=True)
            att_in = torch.cat([s_msg, v_mag_mean], dim=-1)
            scores = self.att_score(att_in)
            alpha = sparse_softmax(scores, dst, num_nodes=s.size(0))
            s_msg = s_msg * alpha
            v_msg = v_msg * alpha.unsqueeze(-1)

        # 3. Aggregation
        aggr_op = 'add' if self.attention else 'mean'
        if aggr_op == 'add':
            s_aggr = scatter_add(s_msg, dst, dim=0, dim_size=s.size(0))
            v_aggr = scatter_add(v_msg, dst, dim=0, dim_size=v.size(0))
        else:
            s_aggr = scatter_mean(s_msg, dst, dim=0, dim_size=s.size(0))
            v_aggr = scatter_mean(v_msg, dst, dim=0, dim_size=v.size(0))

        # 4. Update & Residual
        dh = (s_aggr, v_aggr)
        if self.residual:
            dh_drop = self.dropout[0](dh)
            x_res = tuple_sum(x, dh_drop)
            x = self.norm[0](x_res)
        else:
            x = self.norm[0](dh)

        # 5. FeedForward
        dh = self.ff_func(x)
        if self.residual:
            dh_drop = self.dropout[1](dh)
            x_res = tuple_sum(x, dh_drop)
            x = self.norm[1](x_res)
        else:
            x = self.norm[1](dh)

        return x


# 辅助函数：Tuple求和
def tuple_sum(*args):
    return tuple(map(sum, zip(*args)))
