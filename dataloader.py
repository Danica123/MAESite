import torch as trc
import numpy as np
from dgl.data import DGLDataset
import os


def get_node_positional_encoding(seq_len, d_model=16):
    """
    生成节点级的正弦位置编码 (Sinusoidal Positional Encoding)
    :param seq_len: 序列长度 N
    :param d_model: 编码维度，默认为16维（也可以是8, 32等）
    :return: [N, d_model] 的 numpy 数组
    """
    # 初始化位置矩阵 [N, 1]
    position = np.arange(seq_len).reshape(-1, 1)

    # 初始化频率项 [d_model/2]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((seq_len, d_model))

    # 偶数维用 sin，奇数维用 cos
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe

def get_positional_embedding(edge_index, num_embeddings=16, device=None):
    row, col = edge_index
    d = row - col
    frequency = trc.exp(
        trc.arange(0, num_embeddings, 2, dtype=trc.float32, device=device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    pe = trc.cat((trc.cos(angles), trc.sin(angles)), -1)

    return pe

def construct_residue_orientation_vectors(coords_ca, coords_cb=None):
    N = coords_ca.shape[0]
    V = trc.zeros((N, 3, 3), dtype=trc.float32)

    # (i) 主链前向 & 后向方向 (Backbone forward & backward)
    for i in range(N):
        # 前向: Cα(i) -> Cα(i+1)
        if i < N - 1:
            fwd = coords_ca[i + 1] - coords_ca[i]
            norm = trc.norm(fwd)
            if norm > 0:
                V[i, 0] = fwd / norm

        # 后向: Cα(i) -> Cα(i-1)
        if i > 0:
            bwd = coords_ca[i - 1] - coords_ca[i]
            norm = trc.norm(bwd)
            if norm > 0:
                V[i, 1] = bwd / norm

    # (ii) 侧链方向 (Cα -> Cβ) 或 近似叉积
    if coords_cb is not None:
        for i in range(N):
            if i in coords_cb:  # 如果存在真实 Cβ (非甘氨酸)
                if isinstance(coords_cb[i], trc.Tensor):
                    cb_coord = coords_cb[i].clone().detach()
                else:
                    cb_coord = trc.tensor(coords_cb[i], dtype=trc.float32)

                vec = cb_coord - coords_ca[i]
                norm = trc.norm(vec)
                if norm > 0:
                    V[i, 2] = vec / norm
            else:
                # 对于甘氨酸 (没有Cβ)，使用主链向量的叉积作为近似
                # 对应需求描述: "using the cross product of forward and backward vectors"
                if i > 0 and i < N - 1:
                    v1 = coords_ca[i] - coords_ca[i - 1]  # vector from prev
                    v2 = coords_ca[i + 1] - coords_ca[i]  # vector to next
                    if trc.norm(v1) > 0 and trc.norm(v2) > 0:
                        v1 = v1 / trc.norm(v1)
                        v2 = v2 / trc.norm(v2)
                        # 注意：叉积方向取决于坐标系定义，通常使用这两个向量构建法向量
                        side = trc.cross(v1, v2)
                        norm = trc.norm(side)
                        if norm > 0:
                            V[i, 2] = side / norm

    return V

def construct_gvp_edge_frame(coords_ca, node_V, edge_index):
    """
    【需求4实现】: 边向量特征
    虽然需求只说了"unit vectors"，但GVP-GNN通常需要一个完整的局部坐标系(Frame)来处理旋转等变性。
    这里的 e1 就是需求中的 "unit vectors from vi to vj"。
    e2, e3 是正交补，用于构建完整的旋转矩阵。

    Returns:
        edge_V: [E, 3, 3] (3个正交单位向量)
    """
    src, dst = edge_index
    rij = coords_ca[dst] - coords_ca[src]  # [E, 3] Vector from src to dst

    # e1: Unit vector (需求明确要求的边向量)
    e1 = rij / (trc.norm(rij, dim=-1, keepdim=True) + 1e-8)

    # 为了构建坐标系，我们需要一个参考向量。通常使用源节点的第一个向量特征(Forward向量)。
    vj = node_V[src, 0]  # [E, 3]

    # Gram–Schmidt 正交化: 剔除 vj 中在 e1 方向上的分量，得到垂直于 e1 的 e2
    proj = (vj * e1).sum(dim=-1, keepdim=True) * e1
    e2 = vj - proj
    e2 = e2 / (trc.norm(e2, dim=-1, keepdim=True) + 1e-8)

    # e3: e1 和 e2 的叉积，构成右手系
    e3 = trc.cross(e1, e2, dim=-1)

    edge_V = trc.stack([e1, e2, e3], dim=1)  # [E, 3, 3]
    return edge_V


def compute_surface_curvature(coords, k=10):
    """
    计算表面曲率 (Surface Curvature / Variation)
    原理: 利用局部邻域的协方差矩阵的特征值。
    Curvature = min_eigenvalue / sum_eigenvalues
    :param coords: [N, 3] Cα坐标 Tensor
    :param k: 邻居数量 (建议 6-10)
    :return: [N, 1] 曲率特征 (0=平面, 1=尖锐/无序)
    """
    N = coords.shape[0]
    device = coords.device
    k = min(k, N - 1)  # 防止 k 大于节点数

    # 1. 计算距离矩阵
    # dist: [N, N]
    dist_mat = trc.cdist(coords, coords)

    # 2. 找到 K 近邻
    # indices: [N, k]
    _, indices = dist_mat.topk(k + 1, largest=False)  # +1 是因为包含自己
    indices = indices[:, 1:]  # 去掉自己 [N, k]

    # 3. 收集邻居坐标
    # neighbors: [N, k, 3]
    neighbors = coords[indices]

    # 4. 去中心化 (Centering)
    # [N, 1, 3]
    centers = coords.unsqueeze(1)
    centered_neighbors = neighbors - centers

    # 5. 计算协方差矩阵 (Covariance Matrix)
    # [N, 3, k] @ [N, k, 3] -> [N, 3, 3]
    cov = trc.matmul(centered_neighbors.transpose(1, 2), centered_neighbors) / k

    # 6. 特征值分解 (Eigenvalues)
    # L: [N, 3] (从小到大排序)
    try:
        L, _ = trc.linalg.eigh(cov)
    except:
        # 极少数情况可能SVD失败，返回全0
        return trc.zeros(N, 1, device=device)

    # 7. 计算曲率 (Surface Variation)
    # sigma = lambda_0 / (lambda_0 + lambda_1 + lambda_2)
    # L[:, 0] 是最小特征值 (法向量方向的方差，平面则接近0)
    curvature = L[:, 0] / (L.sum(dim=1) + 1e-8)

    return curvature.unsqueeze(1)  # [N, 1]


def normalize_b_factor(b_factors):
    """
    B-factor 标准化 (Z-score)
    :param b_factors: [N] list or array
    :return: [N, 1] tensor
    """
    b = trc.tensor(b_factors, dtype=trc.float32)
    mean = b.mean()
    std = b.std() + 1e-8
    b_norm = (b - mean) / std
    return b_norm.unsqueeze(1)

def get_charge_feature(aa_codes):
    """
    针对 DNA 任务的关键特征：电荷
    R, K -> +1 (正电，吸附DNA)
    D, E -> -1 (负电，排斥DNA)
    H -> +0.5 (弱正电)
    Others -> 0
    """
    charges = []
    # 氨基酸单字母映射
    # 假设 aa_codes 是数字编码，需要先转回字母，或者直接建立数字到电荷的映射
    # 这里假设我们能拿到 PSSM 或者原始序列。
    # 简便方法：根据 PSSM 最大概率对应的氨基酸来估算
    return trc.tensor(charges).unsqueeze(1)


def rbf_expand(dist, D=16, cutoff=16.0, device=None):
    """
    【修正】采用 GVP 原始逻辑的 RBF 扩展
    保证高斯核的宽度 (sigma) 与中心点间距匹配，实现平滑覆盖。
    """
    # 1. 定义中心点 mu
    # GVP原始代码使用 linspace(min, max, count)
    D_min, D_max, D_count = 0., cutoff, D
    mu = trc.linspace(D_min, D_max, D_count, device=device).view(1, D_count)

    # 2. 动态计算带宽 sigma
    # 使得 sigma 等于两个中心点之间的距离
    sigma = (D_max - D_min) / D_count

    # 3. 计算 RBF
    # dist: [E, 1], mu: [1, D] -> [E, D]
    # 公式: exp( -((d - mu) / sigma)^2 )
    return trc.exp(-((dist - mu) / sigma) ** 2)

# =============================================================================
# 数据集类
# =============================================================================

class buildGraph(DGLDataset):
    def __init__(self, indir, strict_length_check=True):
        self.indir = indir
        self.strict_length_check = strict_length_check
        super().__init__(name='buildgraph')

    def sigmoid(self, x):
        """数值稳定的sigmoid，用于归一化pLM等特征"""
        x = np.clip(x, -20, 20)
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = np.zeros_like(x)
        z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        z[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))
        return z

    def validate_and_fix_lengths(self, features_dict, target_len, protein_name):
        """
        【修正】强制所有特征对齐到 target_len (以DSSP长度为准)，不再使用投票机制
        """
        fixed_features = {}
        for name, feat in features_dict.items():
            current_len = feat.shape[0]
            if current_len != target_len:
                if current_len > target_len:
                    # 截断
                    fixed_features[name] = feat[:target_len]
                else:
                    # 填充
                    pad_len = target_len - current_len
                    # 维度判定
                    if feat.ndim == 1:
                        fixed_features[name] = np.pad(feat, (0, pad_len), mode='constant')
                    else:
                        fixed_features[name] = np.pad(feat, ((0, pad_len), (0, 0)), mode='constant')
            else:
                fixed_features[name] = feat

        return target_len, fixed_features

    def load_scalar_features(self, name):
        """
        【需求1实现】: 加载并拼接所有节点标量特征
        """
        tmp_dir = self.indir + '/tmp/'
        input_dir = self.indir + '/input/'

        try:
            features = {}
            # --- (iv) 结构衍生特征 ---
            pdbfeat = np.load(tmp_dir + name + '.feat.npy')  # data.shape=(147, 28)
            features['dssp'] = pdbfeat[:, :26]  # DSSP (26维)
            features['surface_area'] = pdbfeat[:, 27:28]  # 虚拟表面积 (1维)

            ccountfeat = np.load(tmp_dir + name + '.concount.npy')  # 接触数 Shape (147,)
            if ccountfeat.ndim == 1:
                ccountfeat = ccountfeat.reshape(-1, 1)
            features['contact'] = ccountfeat

            features['feat22'] = np.load(tmp_dir + name + '.feat22.npy')  # 结构上下文 [N, 22] Shape (147, 22)
            features['angles'] = np.load(tmp_dir + name + '.feat_angle6.npy')  # sin/cos Dihedrals [N, 6]  (147, 6)

            # --- (ii) 进化特征 (PSSM) ---
            pssmfeat = np.load(tmp_dir + name + '.npy')  # Shape (147, 20)
            if pssmfeat.shape[1] > 20:
                pssmfeat = pssmfeat[:, :20]
            features['pssm'] = pssmfeat  # [N, 20]

            # --- (i) pLM特征 (ESM-2) ---
            # 路径需根据实际情况确认。需求指明维度为 1536 (ESM-2 3B)
            # esm2feat = np.load(self.indir + 'esm3/RNA_495_train/' + name + '.npy') # Shape (147, 1536) 573
            esm2feat = np.load(self.indir + 'esm3/DNA_573_train/' + name + '.npy')  # Shape (147, 1536) 573

            # ESM特征对齐逻辑 (处理 CLS/EOS token)
            seq_len = features['dssp'].shape[0]
            esm2_len = esm2feat.shape[0]
            if esm2_len == seq_len + 2:
                esm2feat = esm2feat[1:-1]
            elif esm2_len > seq_len:
                esm2feat = esm2feat[:seq_len]
            elif esm2_len < seq_len:
                pad_len = seq_len - esm2_len
                esm2feat = np.pad(esm2feat, ((0, pad_len), (0, 0)), mode='constant')

            # features['esm2'] = self.sigmoid(esm2feat)  # [N, 1536]
            mean = esm2feat.mean(axis=0)
            std = esm2feat.std(axis=0) + 1e-6
            features['esm2'] = (esm2feat - mean) / std

            # --- (iii) MSA特征 ---
            singlefeat = np.load(input_dir + name + 'msa_first_row.npy') # Shape (147, 256)
            if singlefeat.shape[0] != seq_len:
                # 简化的对齐逻辑...
                if singlefeat.shape[0] > seq_len:
                    singlefeat = singlefeat[:seq_len]
                else:
                    singlefeat = np.pad(singlefeat, ((0, seq_len - singlefeat.shape[0]), (0, 0)), mode='constant')
            features['msa'] = self.sigmoid(singlefeat)  # [N, 256]

            # --- 结构衍生：相对位置特征 ---
            # 8. 正弦位置编码 (推荐使用 16 维)
            pe_dim = 16
            node_pe = get_node_positional_encoding(seq_len, d_model=pe_dim)
            features['rel_seq'] = node_pe  # 维度变为 [N, 16]
            # 相对空间位置 (占位，稍后用 coords 计算更新)
            features['rel_spatial'] = np.zeros((seq_len, 1))  # [N, 1]

            # === 从 PSSM 计算香农熵 (Conservation) ===
            pssm_safe = features['pssm'] + 1e-8
            # 归一化使其和为1 (以防万一)
            pssm_norm = pssm_safe / pssm_safe.sum(axis=1, keepdims=True)
            # 计算熵: -sum(p * log(p))
            entropy = -np.sum(pssm_norm * np.log2(pssm_norm), axis=1, keepdims=True)  # [N, 1]
            # 归一化熵 (最大熵是 log2(20) ≈ 4.32)
            features['entropy'] = entropy / 4.32

            # 1. 确定基准长度 (Ground Truth Length)
            # 优先使用 DSSP 的长度，因为它直接对应 PDB 的残基
            if 'dssp' in features:
                target_len = features['dssp'].shape[0]
            elif 'dssp_cat' in features:  # 如果您用了拆分版
                target_len = features['dssp_cat'].shape[0]
            else:
                # 兜底：如果没DSSP，随便取第一个特征的长度
                target_len = list(features.values())[0].shape[0]

            # 2. 强制对齐 (传入 target_len)
            final_len, fixed_features = self.validate_and_fix_lengths(features, target_len, name)

            # 拼接
            scalar_features = np.concatenate([
                fixed_features['esm2'],  # 1536 (i)
                fixed_features['pssm'],  # 20   (ii)
                fixed_features['msa'],  # 256  (iii)
                fixed_features['dssp'],  # 25   (iv start...)
                fixed_features['surface_area'],  # 2
                fixed_features['feat22'],  # 22
                fixed_features['contact'],  # 1
                fixed_features['angles'],  # 6
                fixed_features['rel_seq'],  # 1
                fixed_features['rel_spatial'],  # 1
                fixed_features['entropy']
            ], axis=1)  # 总和应为 1870

            return trc.tensor(scalar_features, dtype=trc.float32), final_len

        except Exception as e:
            raise ValueError(f"Feature loading failed for {name}: {e}")

    def compute_relative_spatial_position(self, coords):
        """计算残基到质心的逆欧几里得距离"""
        centroid = coords.mean(dim=0, keepdim=True)
        distances = trc.norm(coords - centroid, dim=1)
        distances = distances + 1e-8
        inv_distances = 1.0 / distances
        # 归一化处理
        max_inv = inv_distances.max()
        if max_inv > 0:
            inv_distances = inv_distances / max_inv
        return inv_distances.unsqueeze(1)

    def process(self):
        self.data_and_label = []
        # ... (文件路径定义保持不变) ...
        trainlist = self.indir + '/input.list'
        label_dir = self.indir + '/labels/'
        node_xyz_dir = self.indir + '/input/'
        edge_dir = self.indir + '/distmaps/'

        with open(trainlist, 'r') as f:
            flines = f.readlines()

        for line in flines:
            tgt = line.strip().split('.')[0]

            # 1. 加载标签 (Label Loading) - 略微简化，保持原逻辑
            try:
                label_file = os.path.join(label_dir, f'{tgt}.label')
                with open(label_file) as labelf:
                    labels = [int(li) for li in labelf.readlines()[0].strip()]
            except:
                continue

            # 2. 加载标量特征 (Scalar Features)
            try:
                scalar_features, feat_len = self.load_scalar_features(tgt)
            except:
                continue

            # 取特征长度和标签长度的最小值
            common_len = min(len(labels), feat_len)

            if common_len == 0:
                print(f"[WARNING] {tgt}: Common length is 0 (Labels={len(labels)}, Feats={feat_len}). Skipping.")
                continue

            # 1. 截断特征 (以防特征比标签长)
            scalar_features = scalar_features[:common_len]
            # 2. 截断标签 (以防标签比特征长)
            labels = labels[:common_len]
            # 3. 更新 min_len，确保后续的坐标(PDB)和边(Dist)也只取前 common_len 个
            min_len = common_len

            # 3. 加载坐标 (Coordinates) - 修正版
            # 初始化：全 0 矩阵
            xyz_ca = np.zeros((min_len, 3), dtype=np.float32)
            xyz_cb = {}  # 字典用于存储稀疏的 Cbeta (甘氨酸没有 Cbeta)
            # [新增] 初始化 B-factor 数组
            b_factors = np.zeros(min_len, dtype=np.float32)

            try:
                pdb_file = os.path.join(node_xyz_dir, f'{tgt}.pdb')
                # 用于临时存储解析出的所有残基坐标: {res_id: {'CA': [x,y,z], 'CB': [x,y,z]}}
                temp_coords = {}
                # [新增] 临时存储 B-factor
                temp_bfactors = {}

                with open(pdb_file, 'r') as xyz_f:
                    for line in xyz_f:
                        if line.startswith("ATOM"):
                            # 提取链 ID (Chain ID) - 可选，如果有特定链需求
                            # chain_id = line[21]
                            # 提取原子类型
                            atom_name = line[12:16].strip()
                            if atom_name not in ['CA', 'CB']: continue

                            # 提取残基编号 (Residue Sequence Number)
                            res_id = int(line[22:26].strip())
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())

                            # [新增] 读取 B-factor (列 60-66)
                            try:
                                bf = float(line[60:66].strip())
                            except:
                                bf = 0.0  # 默认值

                            if res_id not in temp_coords:
                                temp_coords[res_id] = {}

                            if atom_name not in temp_coords[res_id]:
                                temp_coords[res_id][atom_name] = [x, y, z]
                                # 如果是 CA，记录 B-factor
                                if atom_name == 'CA':
                                    temp_bfactors[res_id] = bf


                # 【关键修正】：将 PDB 中的坐标映射到我们的特征列表中
                # 这里假设 PDB 中的残基顺序与您的标量特征（DSSP/ESM等）是严格对应的。
                # 我们获取 temp_coords 中所有排序后的残基 ID
                sorted_ids = sorted(temp_coords.keys())

                # 检查 PDB 残基数量是否足够
                # 注意：这只是一个简单的对齐策略。最严谨的方法是比对序列(Sequence Alignment)。
                # 但在不做序列比对的情况下，通常假设 PDB 文件经过了清洗，与特征文件是一一对应的。

                # 策略：直接按顺序填充前 min_len 个坐标
                # 如果 PDB 残基数少于 min_len，后面的保持为 0 (将被 Mask 掉或报错)
                # 如果 PDB 残基数多于 min_len，只取前 min_len 个

                count = 0
                for res_id in sorted_ids:
                    if count >= min_len:
                        break

                    # 获取 CA
                    if 'CA' in temp_coords[res_id]:
                        xyz_ca[count] = temp_coords[res_id]['CA']
                        if 'CB' in temp_coords[res_id]:
                            xyz_cb[count] = temp_coords[res_id]['CB']

                            # [新增] 填充 B-factor
                        if res_id in temp_bfactors:
                            b_factors[count] = temp_bfactors[res_id]
                        count += 1

            except Exception as e:
                print(f"Error loading PDB {tgt}: {e}")
                continue

            coords = trc.tensor(xyz_ca, dtype=trc.float32)

            # 1. 相对空间位置 (原逻辑)
            rel_spatial_pos = self.compute_relative_spatial_position(coords)  # [N, 1]

            # 2. 表面曲率 (新逻辑)
            curvature = compute_surface_curvature(coords, k=10)  # [N, 1]

            # 3. B-factor (新逻辑)
            b_factor_feat = normalize_b_factor(b_factors)  # [N, 1]


            # 分离 Entropy (最后一列)
            feat_base = scalar_features[:, :1884]  # 去掉 rel_spatial(占位) 和 entropy
            entropy = scalar_features[:, -1:]  # 取出 entropy
            pssm_part = scalar_features[:, 1536:1556]
            pos_charge = pssm_part[:, 1] + pssm_part[:, 11] + 0.5 * pssm_part[:, 8]
            neg_charge = pssm_part[:, 3] + pssm_part[:, 6]
            net_charge = (pos_charge - neg_charge).unsqueeze(1)  # [N, 1]

            # 重新组装
            # 顺序: [Base, Rel_Spatial, Curvature, B_factor, Entropy]
            # 这样物理特征在一起，Entropy 放最后
            s = trc.cat([
                feat_base,  # 基础语义
                rel_spatial_pos,  # 埋藏程度
                curvature,  # 凹凸形状 (DNA关键)
                b_factor_feat,  # 柔性
                net_charge,  # 静电 (DNA关键!)
                entropy  # 守恒性
            ], dim=1)

            # 4. 构建边 (Edges)
            src, dst = [], []
            try:
                dist_file = os.path.join(edge_dir, f'{tgt}.dist')
                with open(dist_file, 'r') as rrfile:
                    for rline in rrfile.readlines()[1:]:
                        ni, nj = int(rline.split()[0]) - 1, int(rline.split()[1]) - 1
                        if ni < min_len and nj < min_len:
                            src += [ni, nj];
                            dst += [nj, ni]  # 双向边
            except:
                continue

            if len(src) == 0: continue
            edge_index = trc.tensor([src, dst], dtype=trc.long)

            # =========================================================
            # 构建 GVP 模型所需的特征
            # =========================================================

            # Node Vector (V): [N, 3, 3]
            # 包含 Forward, Backward, Sidechain
            coords_cb_tensor = {}
            for i in range(min_len):
                if i in xyz_cb: coords_cb_tensor[i] = trc.tensor(xyz_cb[i])
            V = construct_residue_orientation_vectors(coords, coords_cb_tensor)

            # 形状修正
            if V.dim() == 2: V = V.unsqueeze(1)
            if V.shape[1] < 3:
                V_padded = trc.zeros((V.shape[0], 3, 3), dtype=V.dtype)
                V_padded[:, :V.shape[1], :] = V
                V = V_padded

            # === 【核心修改区域 Start】 ===
            src_idx, dst_idx = edge_index

            # 1. 空间距离 RBF (保持不变，对应原始代码的 _rbf)
            dist_vec = coords[dst_idx] - coords[src_idx]
            dist = trc.norm(dist_vec, dim=-1, keepdim=True)
            edge_rbf = rbf_expand(dist, D=16, cutoff=20.0, device=coords.device)

            # 2. 序列距离 Positional Embedding (修正！对应原始代码的 _positional_embeddings)
            # 之前错误地使用了 dist (空间距离)，现在改为 edge_index (序列索引)
            edge_pos = get_positional_embedding(edge_index, num_embeddings=16, device=coords.device)

            # 3. 拼接
            # edge_s: [E, 16] + [E, 16] = [E, 32]
            edge_s = trc.cat([edge_rbf, edge_pos], dim=-1)
            # === 【核心修改区域 End】 ===

            # 边向量特征 (保持不变，虽然原始代码只用单向量，但为了您的模型兼容性，建议保留完整Frame)
            edge_V = construct_gvp_edge_frame(coords, V, edge_index)

            # 最终边属性元组
            edge_attr = (edge_s, edge_V)

            # Label
            labels_tensor = trc.tensor(labels, dtype=trc.float32)

            self.data_and_label.append((s, V, edge_index, edge_attr, labels_tensor))

    def __getitem__(self, i):
        return self.data_and_label[i]

    def __len__(self):
        return len(self.data_and_label)
