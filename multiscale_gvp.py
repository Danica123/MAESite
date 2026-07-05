import torch
import torch.nn as nn
import torch.nn.functional as F
from gvp import GVPConvLayer, GVP  # 引用之前完善的 gvp.py


# 辅助函数：标准化 edge_index 格式
def normalize_edge_index(edge_index):
    if isinstance(edge_index, (list, tuple)):
        return normalize_edge_index(edge_index[0])
    return edge_index.long()


class PerScaleEncoder(nn.Module):
    """
    单尺度编码器分支 (Per-Scale Encoder Branch)
    负责处理特定距离范围内的图结构
    """

    def __init__(self, node_dims, edge_dims, n_layers, dropout=0.1):
        super().__init__()
        # 堆叠 GVP-GNN 层
        self.layers = nn.ModuleList([
            GVPConvLayer(
                node_dims=node_dims,
                edge_dims=edge_dims,
                n_message=3,
                drop_rate=dropout,
                residual=True,
                attention=True  # 开启几何感知注意力
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(node_dims[0])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        s, V = x
        for layer in self.layers:
            # GVPConvLayer 处理消息传递
            s_new, V_new = layer((s, V), edge_index, edge_attr)

            # 跨层残差连接
            s = s + self.dropout(s_new)
            V = V + self.dropout(V_new)

        s = self.norm(s)
        return s, V

class ScaleFusion(nn.Module):
    """
    多尺度融合模块 (Scale Fusion)
    使用门控机制 (Gated Sum) 自动学习不同尺度的重要性权重
    """

    def __init__(self, s_dim, v_dim, n_scales):
        super().__init__()
        # 评分网络: 根据节点的特征，决定它更依赖哪个尺度
        self.gate_net = nn.Sequential(
            nn.Linear(s_dim, s_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(s_dim // 2, 1)
        )
        self.v_dim = v_dim

    def forward(self, s_list, V_list):
        # s_list: [Scale0, Scale1, ...] -> Stack -> [S, N, D]
        s_stack = torch.stack(s_list, dim=0)

        # 计算每个尺度的权重: [S, N, 1]
        weights = self.gate_net(s_stack)
        # 在尺度维度 S 上进行 Softmax，确保权重和为 1
        weights = F.softmax(weights, dim=0)

        # 标量加权融合
        s_out = (weights * s_stack).sum(dim=0)  # [N, D]

        # 向量加权融合 (复用标量计算出的权重)
        V_stack = torch.stack(V_list, dim=0)  # [S, N, v_dim, 3]
        weights_v = weights.unsqueeze(-1)  # [S, N, 1, 1]
        V_out = (weights_v * V_stack).sum(dim=0)

        return s_out, V_out

class CrossScaleAttentionFusion(nn.Module):
    """
    【升级版融合】跨尺度注意力融合模块 (Cross-Scale Attention Fusion)

    原理:
    将不同尺度的特征视为一个序列 [Local, Medium, Global]，
    使用 Self-Attention 让不同尺度进行交互，交换信息。
    """

    def __init__(self, s_dim, v_dim, n_scales, n_heads=4, dropout=0.1):
        super().__init__()
        self.s_dim = s_dim
        self.n_scales = n_scales

        # 1. 尺度嵌入 (Scale Embedding)
        # 让模型知道哪个向量来自哪个尺度 (类似 Transformer 的 Positional Encoding)
        self.scale_emb = nn.Parameter(torch.randn(n_scales, s_dim))

        # 2. 多头注意力 (Multi-Head Attention)
        # 这里把 节点(N) 看作 Batch，尺度(S) 看作 Sequence Length
        self.mha = nn.MultiheadAttention(embed_dim=s_dim, num_heads=n_heads,
                                         dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(s_dim)
        self.dropout = nn.Dropout(dropout)

        # 3. 向量融合门控 (Vector Gating)
        # 根据交互后的标量特征，计算每个尺度的向量权重
        self.vector_gate = nn.Sequential(
            nn.Linear(s_dim, n_scales),
            nn.Softmax(dim=-1)
        )

    def forward(self, s_list, V_list):
        # s_list: List of [N, D]
        # V_list: List of [N, V_dim, 3]

        # 1. 堆叠: [N, S, D] (Batch=N, Seq=S, Feature=D)
        # 注意: 之前的 ScaleFusion 是 [S, N, D]，这里为了 MHA 改为 [N, S, D]
        s_stack = torch.stack(s_list, dim=1)
        N, S, D = s_stack.shape

        # 2. 添加尺度嵌入
        # scale_emb: [S, D] -> [1, S, D] -> broadcast to [N, S, D]
        s_emb = s_stack + self.scale_emb.unsqueeze(0)

        # 3. 跨尺度注意力交互 (Cross-Scale Interaction)
        # attn_out: [N, S, D]
        s_attn, _ = self.mha(s_emb, s_emb, s_emb)

        # 残差 + 归一化
        s_interacted = self.norm(s_stack + self.dropout(s_attn))

        # 4. 标量融合 (Scalar Aggregation)
        # 将交互后的尺度特征进行平均池化 (Mean Pooling)
        s_fused = s_interacted.mean(dim=1)  # [N, D]

        # 5. 向量融合 (Vector Aggregation)
        # 利用交互后的标量特征来决定如何融合向量
        # V_stack: [N, S, V_dim, 3]
        V_stack = torch.stack(V_list, dim=1)

        # 计算权重: [N, D] -> [N, S]
        weights = self.vector_gate(s_fused)
        # [N, S, 1, 1]
        weights = weights.view(N, S, 1, 1)

        # 加权求和
        V_fused = (weights * V_stack).sum(dim=1)  # [N, V_dim, 3]

        return s_fused, V_fused


class ChannelWiseGatedFusion(nn.Module):
    """
    【Deep Supervision 的最佳搭档】通道级门控融合

    原理：
    既然每个尺度的特征都已经通过 loss_aux 被训练得很好了，
    我们只需要精细地挑选每个通道的最佳来源即可。
    """

    def __init__(self, s_dim, v_dim, n_scales):
        super().__init__()

        # 1. 标量门控网络 (Scalar Gating Network)
        # 输入: 所有尺度的特征堆叠 [S, N, D]
        # 任务: 为每个通道 d 生成一个权重 w_d
        self.scalar_gate = nn.Sequential(
            nn.Linear(s_dim, s_dim // 2),
            nn.LayerNorm(s_dim // 2),
            nn.GELU(),  # GELU 比 ReLU 更平滑，适合深层网络
            nn.Linear(s_dim // 2, s_dim)  # 输出 D 个权重
        )

        # 2. 向量门控网络 (Vector Gating Network)
        # 输入: 标量特征 [S, N, D]
        # 任务: 为每个尺度生成一个标量权重 w_s (保持向量的旋转等变性，不能做Channel-wise)
        self.vector_gate = nn.Sequential(
            nn.Linear(s_dim, s_dim // 4),
            nn.GELU(),
            nn.Linear(s_dim // 4, 1)  # 输出 1 个权重
        )

    def forward(self, s_list, V_list):
        # s_list: List of [N, D]
        # V_list: List of [N, V_dim, 3]

        # 堆叠: [S, N, D]
        s_stack = torch.stack(s_list, dim=0)

        # --- 1. 标量融合 (Channel-Wise Selection) ---
        # 计算原始 Logits: [S, N, D]
        gate_logits = self.scalar_gate(s_stack)

        # Softmax 归一化 (跨尺度 S)
        # 意义: 对于第 n 个残基的第 d 个特征通道，3个尺度的权重和为1
        weights_s = F.softmax(gate_logits, dim=0)

        # 加权求和
        s_fused = (weights_s * s_stack).sum(dim=0)  # [N, D]

        # --- 2. 向量融合 (Scale-Wise Selection) ---
        # 利用 s_stack 的信息来决定
        V_stack = torch.stack(V_list, dim=0)  # [S, N, V_dim, 3]

        # 计算权重: [S, N, D] -> [S, N, 1]
        v_logits = self.vector_gate(s_stack)
        weights_v = F.softmax(v_logits, dim=0)  # [S, N, 1]

        # 广播权重: [S, N, 1, 1]
        weights_v = weights_v.unsqueeze(-1)

        # 加权求和
        V_fused = (weights_v * V_stack).sum(dim=0)  # [N, V_dim, 3]

        return s_fused, V_fused


class MultiScaleGVPBindingPredictor(nn.Module):
    """
    多尺度 GVP-GNN 结合位点预测器
    """

    def __init__(self,
                 orig_node_dims=(1889, 3),
                 hidden_dim=256,
                 edge_dims=(32, 3),
                 n_layers=3,
                 dropout=0.1,
                 scales=[5, 10, 20],
                 n_heads=4, # [新增参数] Attention 头数
                 task_type = 'RNA'
                 ):
        super().__init__()
        self.scales = scales

        # 1. ESM 专用投影层 (1536 -> 384)
        self.esm_dim = 1536
        self.esm_proj_dim = 192  #384/128
        self.esm_projector = nn.Sequential(
            nn.Linear(self.esm_dim, self.esm_proj_dim),
            nn.LayerNorm(self.esm_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. 其他特征投影 (1886 - 1536 = 350)
        self.other_dim = orig_node_dims[0] - self.esm_dim
        self.other_proj_dim = hidden_dim - self.esm_proj_dim
        self.other_projector = nn.Sequential(
            nn.Linear(self.other_dim, self.other_proj_dim),
            nn.LayerNorm(self.other_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 向量投影: 3 -> 16
        target_v_dim = 16
        self.v_in = nn.Linear(orig_node_dims[1], target_v_dim, bias=False)

        self.node_dims = (hidden_dim, target_v_dim)

        # 2. 多尺度编码器组 (Multi-scale Encoders)
        self.encoders = nn.ModuleList([
            PerScaleEncoder(self.node_dims, edge_dims, n_layers, dropout)
            for _ in scales
        ])

        # 3. 融合模块
        # self.fusion = ScaleFusion(hidden_dim, target_v_dim, len(scales))
        # self.fusion = CrossScaleAttentionFusion(
        #     s_dim=hidden_dim,
        #     v_dim=target_v_dim,
        #     n_scales=len(scales),
        #     n_heads=n_heads,
        #     dropout=dropout
        # )
        self.fusion = ChannelWiseGatedFusion(hidden_dim, target_v_dim, len(scales))

        # 4. 输出头
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, s_orig, V_orig, edge_index, edge_attr, return_feats=False):
        device = s_orig.device
        edge_index = normalize_edge_index(edge_index).to(device)

        # 1. 分体式特征投影
        s_esm = s_orig[:, :self.esm_dim]
        s_other = s_orig[:, self.esm_dim:]

        s_esm_emb = self.esm_projector(s_esm)
        s_other_emb = self.other_projector(s_other)

        s = torch.cat([s_esm_emb, s_other_emb], dim=-1)  # [N, hidden]

        # V投影
        V_trans = V_orig.transpose(-1, -2)
        V = self.v_in(V_trans)
        V = V.transpose(-1, -2)

        # 2. 多尺度并行处理
        s_list = []
        V_list = []

        s_e_all, V_e_all = edge_attr

        for i, cutoff_idx in enumerate(self.scales):
            # Hard Multi-scale Masking via RBF
            rbf_energy = s_e_all[:, :cutoff_idx].sum(dim=-1)
            mask = rbf_energy > 0.1

            if mask.sum() == 0:
                mask = torch.ones_like(rbf_energy, dtype=torch.bool)

            edge_index_sub = edge_index[:, mask]
            s_e_sub = s_e_all[mask]
            V_e_sub = V_e_all[mask]

            s_out, V_out = self.encoders[i]((s, V), edge_index_sub, (s_e_sub, V_e_sub))

            s_list.append(s_out)
            V_list.append(V_out)

        # 3. 跨尺度注意力融合
        s_fused, V_fused = self.fusion(s_list, V_list)

        # 4. 预测
        logits = self.readout(s_fused).squeeze(-1)

        # === 返回中间特征用于可视化 ===
        if self.training:
            aux_logits = []
            for s_scale in s_list:
                # 复用 readout 头对每个尺度的特征进行预测
                # 这样可以强迫每个尺度的特征空间都与最终分类目标对齐
                aux_out = self.readout(s_scale).squeeze(-1)
                aux_logits.append(aux_out)

            # 返回 tuple: (主输出, 辅助输出列表)
            return logits, aux_logits
        else:
            if return_feats:
                return logits, V_fused, {
                    'Initial': s,  # 这是经过 s_in 投影后的初始特征
                    'Scale_Local': s_list[0],
                    'Scale_Medium': s_list[1],
                    'Scale_Global': s_list[2],
                    'Final_Fused': s_fused
                }
            return logits, V_fused