import torch as trc
import numpy as np
from dgl.data import DGLDataset
import os


def get_node_positional_encoding(seq_len, d_model=16):
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
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
    for i in range(N):
        if i < N - 1:
            fwd = coords_ca[i + 1] - coords_ca[i]
            norm = trc.norm(fwd)
            if norm > 0: V[i, 0] = fwd / norm
        if i > 0:
            bwd = coords_ca[i - 1] - coords_ca[i]
            norm = trc.norm(bwd)
            if norm > 0: V[i, 1] = bwd / norm

    if coords_cb is not None:
        for i in range(N):
            if i in coords_cb:
                if isinstance(coords_cb[i], trc.Tensor):
                    cb_coord = coords_cb[i].clone().detach()
                else:
                    cb_coord = trc.tensor(coords_cb[i], dtype=trc.float32)
                vec = cb_coord - coords_ca[i]
                norm = trc.norm(vec)
                if norm > 0: V[i, 2] = vec / norm
            else:
                if i > 0 and i < N - 1:
                    v1 = coords_ca[i] - coords_ca[i - 1]
                    v2 = coords_ca[i + 1] - coords_ca[i]
                    if trc.norm(v1) > 0 and trc.norm(v2) > 0:
                        v1 = v1 / trc.norm(v1)
                        v2 = v2 / trc.norm(v2)
                        side = trc.cross(v1, v2)
                        norm = trc.norm(side)
                        if norm > 0: V[i, 2] = side / norm
    return V


def construct_gvp_edge_frame(coords_ca, node_V, edge_index):
    src, dst = edge_index
    rij = coords_ca[dst] - coords_ca[src]
    e1 = rij / (trc.norm(rij, dim=-1, keepdim=True) + 1e-8)
    vj = node_V[src, 0]
    proj = (vj * e1).sum(dim=-1, keepdim=True) * e1
    e2 = vj - proj
    e2 = e2 / (trc.norm(e2, dim=-1, keepdim=True) + 1e-8)
    e3 = trc.cross(e1, e2, dim=-1)
    edge_V = trc.stack([e1, e2, e3], dim=1)
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

def rbf_expand(dist, D=16, cutoff=16.0, device=None):
    D_min, D_max, D_count = 0., cutoff, D
    mu = trc.linspace(D_min, D_max, D_count, device=device).view(1, D_count)
    sigma = (D_max - D_min) / D_count
    return trc.exp(-((dist - mu) / sigma) ** 2)


# =============================================================================
# 2. 数据集类
# =============================================================================

class TestGraphDataset(DGLDataset):
    def __init__(self, indir, strict_length_check=True):
        self.indir = indir
        self.strict_length_check = strict_length_check
        super().__init__(name='testgraph')

    def sigmoid(self, x):
        x = np.clip(x, -20, 20)
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = np.zeros_like(x)
        z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        z[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))
        return z

    def validate_and_fix_lengths(self, features_dict, target_len, protein_name):
        """
        【修正】强制所有特征对齐到 target_len (DSSP长度)，不再使用投票机制
        这能解决 3jcm_B 等样本特征长度不一致导致拼接失败的问题
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
                    if feat.ndim == 1:
                        fixed_features[name] = np.pad(feat, (0, pad_len), mode='constant')
                    else:
                        fixed_features[name] = np.pad(feat, ((0, pad_len), (0, 0)), mode='constant')
            else:
                fixed_features[name] = feat
        return target_len, fixed_features

    def load_scalar_features(self, name):
        """加载标量特征"""
        tmp_dir = os.path.join(self.indir, 'tmp/')
        input_dir = os.path.join(self.indir, 'input/')

        try:
            features = {}
            # 1. 基础特征 (作为长度基准)
            pdbfeat = np.load(os.path.join(tmp_dir, name + '.feat.npy'))
            features['dssp'] = pdbfeat[:, :26]
            features['surface_area'] = pdbfeat[:, 27:28]

            ccountfeat = np.load(os.path.join(tmp_dir, name + '.concount.npy'))
            if ccountfeat.ndim == 1: ccountfeat = ccountfeat.reshape(-1, 1)
            features['contact'] = ccountfeat

            features['feat22'] = np.load(os.path.join(tmp_dir, name + '.feat22.npy'))
            features['angles'] = np.load(os.path.join(tmp_dir, name + '.feat_angle6.npy'))

            pssmfeat = np.load(os.path.join(tmp_dir, name + '.npy'))
            if pssmfeat.shape[1] > 20: pssmfeat = pssmfeat[:, :20]
            features['pssm'] = pssmfeat

            # ESM特征 (需使用 ESM3 路径)
            esm2feat = np.load(os.path.join(self.indir + '/esm3/DNA_test_181/', name + '.npy'))
            # esm2feat = np.load(os.path.join(input_dir, name + '.rep_1280_esm_dbp.npy')) # esmdbp

            # 【基准长度】所有特征必须对齐到这个长度
            seq_len = features['dssp'].shape[0]
            # ESM 对齐逻辑
            esm2_len = esm2feat.shape[0]
            if esm2_len == seq_len + 2:
                esm2feat = esm2feat[1:-1]
            elif esm2_len > seq_len:
                esm2feat = esm2feat[:seq_len]
            elif esm2_len < seq_len:
                pad_len = seq_len - esm2_len
                esm2feat = np.pad(esm2feat, ((0, pad_len), (0, 0)), mode='constant')

            # 【关键修正】ESM 使用均值方差归一化 (匹配训练)
            mean = esm2feat.mean(axis=0)
            std = esm2feat.std(axis=0) + 1e-6
            features['esm2'] = (esm2feat - mean) / std

            singlefeat = np.load(os.path.join(input_dir, name + 'msa_first_row.npy'))
            if singlefeat.shape[0] != seq_len:
                if singlefeat.shape[0] > seq_len:
                    singlefeat = singlefeat[:seq_len]
                else:
                    singlefeat = np.pad(singlefeat, ((0, seq_len - singlefeat.shape[0]), (0, 0)), mode='constant')
            features['msa'] = self.sigmoid(singlefeat)

            # 位置编码
            pe_dim = 16
            node_pe = get_node_positional_encoding(seq_len, d_model=pe_dim)
            features['rel_seq'] = node_pe
            features['rel_spatial'] = np.zeros((seq_len, 1))  # 占位

            # 熵计算
            pssm_safe = features['pssm'] + 1e-8
            pssm_norm = pssm_safe / pssm_safe.sum(axis=1, keepdims=True)
            entropy = -np.sum(pssm_norm * np.log2(pssm_norm), axis=1, keepdims=True)
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
                fixed_features['esm2'],
                fixed_features['pssm'],
                fixed_features['msa'],
                fixed_features['dssp'],
                fixed_features['surface_area'],
                fixed_features['feat22'],
                fixed_features['contact'],
                fixed_features['angles'],
                fixed_features['rel_seq'],
                fixed_features['rel_spatial'],
                fixed_features['entropy']
            ], axis=1)

            return trc.tensor(scalar_features, dtype=trc.float32), final_len

        except Exception as e:
            raise

    def compute_relative_spatial_position(self, coords):
        centroid = coords.mean(dim=0, keepdim=True)
        distances = trc.norm(coords - centroid, dim=1)
        distances = distances + 1e-8
        inv_distances = 1.0 / distances
        max_inv = inv_distances.max()
        if max_inv > 0: inv_distances = inv_distances / max_inv
        return inv_distances.unsqueeze(1)

    def process(self):
        self.data_list = []
        testlist = os.path.join(self.indir, 'input.list')
        node_xyz_dir = os.path.join(self.indir, 'input')
        edge_dir = os.path.join(self.indir, 'distmaps')

        with open(testlist, 'r') as f:
            targets = [line.strip().split('.')[0] for line in f.readlines()]

        for target in targets:
            try:
                # 1. 加载标量特征
                scalar_features, seq_len = self.load_scalar_features(target)

                # 2. 坐标解析
                xyz_ca = np.zeros((seq_len, 3), dtype=np.float32)
                xyz_cb = {}
                b_factors = np.zeros(seq_len, dtype=np.float32)
                try:
                    pdb_file = os.path.join(node_xyz_dir, f'{target}.pdb')
                    temp_coords = {}
                    temp_bfactors = {}
                    if os.path.exists(pdb_file):
                        with open(pdb_file, 'r') as f:
                            for line in f:
                                if not line.startswith("ATOM"): continue
                                atom_name = line[12:16].strip()
                                if atom_name not in ['CA', 'CB']: continue

                                res_id = int(line[22:26].strip())
                                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])

                                # B-factor
                                try:
                                    bf = float(line[60:66].strip())
                                except:
                                    bf = 0.0

                                if res_id not in temp_coords: temp_coords[res_id] = {}
                                if atom_name not in temp_coords[res_id]:
                                    temp_coords[res_id][atom_name] = [x, y, z]
                                    if atom_name == 'CA': temp_bfactors[res_id] = bf

                        sorted_ids = sorted(temp_coords.keys())
                        count = 0
                        for rid in sorted_ids:
                            if count >= seq_len: break
                            if 'CA' in temp_coords[rid]:
                                xyz_ca[count] = temp_coords[rid]['CA']
                                if 'CB' in temp_coords[rid]: xyz_cb[count] = temp_coords[rid]['CB']
                                if rid in temp_bfactors: b_factors[count] = temp_bfactors[rid]
                                count += 1
                except:
                    pass

                coords = trc.tensor(xyz_ca, dtype=trc.float32)

                # 3. 更新空间位置
                rel_spatial_pos = self.compute_relative_spatial_position(coords)
                curvature = compute_surface_curvature(coords, k=10)
                b_factor_feat = normalize_b_factor(b_factors)

                # # === 特征重组 (与训练集严格一致) ===
                # feat_base = scalar_features[:, :-2]
                # entropy = scalar_features[:, -1:]
                # s = trc.cat([feat_base, rel_spatial_pos, curvature, b_factor_feat, entropy], dim=1)

                # 分离 Entropy (最后一列)
                feat_base = scalar_features[:, :1884]  # 去掉 rel_spatial(占位) 和 entropy
                entropy = scalar_features[:, -1:]  # 取出 entropy
                pssm_part = scalar_features[:, 1536:1556]
                pos_charge = pssm_part[:, 1] + pssm_part[:, 11] + 0.5 * pssm_part[:, 8]
                neg_charge = pssm_part[:, 3] + pssm_part[:, 6]
                net_charge = (pos_charge - neg_charge).unsqueeze(1)  # [N, 1]

                s = trc.cat([
                    feat_base,  # 基础语义
                    rel_spatial_pos,  # 埋藏程度
                    curvature,  # 凹凸形状 (DNA关键)
                    b_factor_feat,  # 柔性
                    net_charge,  # 静电 (DNA关键!)
                    entropy  # 守恒性
                ], dim=1)

                # 4. 构建边
                src, dst = [], []
                dist_file = os.path.join(edge_dir, f'{target}.dist')
                if os.path.exists(dist_file):
                    try:
                        with open(dist_file, 'r') as f:
                            lines = f.readlines()[1:]
                            for line in lines:
                                parts = line.split()
                                ni, nj = int(parts[0]) - 1, int(parts[1]) - 1
                                if 0 <= ni < seq_len and 0 <= nj < seq_len:
                                    src.extend([ni, nj])
                                    dst.extend([nj, ni])
                    except:
                        pass

                if len(src) == 0:
                    try:
                        from scipy.spatial import cKDTree
                        coords_np = coords.numpy()
                        k = min(20, seq_len - 1)
                        if k > 0:
                            tree = cKDTree(coords_np)
                            _, indices = tree.query(coords_np, k=k + 1)
                            for i in range(seq_len):
                                for j in indices[i, 1:]:
                                    if j < seq_len:
                                        src.extend([i, j])
                                        dst.extend([j, i])
                    except:
                        pass

                src = np.array(src, dtype=np.int64)
                dst = np.array(dst, dtype=np.int64)

                if len(src) == 0: continue
                # 【修正：UserWarning】使用 stack + from_numpy
                edge_index = trc.from_numpy(np.stack([src, dst])).long()

                # 5. GVP 特征
                coords_cb_tensor = {}
                for i in range(seq_len):
                    if i in xyz_cb: coords_cb_tensor[i] = trc.tensor(xyz_cb[i])
                V = construct_residue_orientation_vectors(coords, coords_cb_tensor)

                if V.dim() == 2: V = V.unsqueeze(1)
                if V.shape[1] < 3:
                    V_padded = trc.zeros((V.shape[0], 3, 3), dtype=V.dtype)
                    V_padded[:, :V.shape[1], :] = V
                    V = V_padded

                src_idx, dst_idx = edge_index
                dist_vec = coords[dst_idx] - coords[src_idx]
                dist = trc.norm(dist_vec, dim=-1, keepdim=True)

                edge_rbf = rbf_expand(dist, D=16, cutoff=14.0, device=coords.device)
                edge_pos = get_positional_embedding(edge_index, num_embeddings=16, device=coords.device)

                edge_s = trc.cat([edge_rbf, edge_pos], dim=-1)
                edge_V = construct_gvp_edge_frame(coords, V, edge_index)
                edge_attr = (edge_s, edge_V)

                self.data_list.append((target, s, V, edge_index, edge_attr))

            except Exception as e:
                print(f"[ERROR] Failed to process {target}: {e}")
                continue

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)


# 兼容性包装器
class buildGraph(TestGraphDataset):
    pass
