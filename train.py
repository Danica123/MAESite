import os
import logging
from datetime import datetime
import warnings
import faulthandler
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, average_precision_score, roc_auc_score,
                             precision_recall_curve)
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# 引用新版模型
from multiscale_gvp import MultiScaleGVPBindingPredictor
from dataloader import buildGraph
from early_stopping import EarlyStopping
from optimized_train_config import get_optimized_config
from losses import CombinedLoss, AdaptiveThresholdLoss, WeightedBCELoss, FocalLoss

faulthandler.enable()
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["OMP_NUM_THREADS"] = "1"


# ========== 日志与绘图工具 (保持不变) ==========
def setup_logger(log_dir, fold):
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    logger = logging.getLogger(f'train_fold{fold}')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(os.path.join(log_dir, f'{FLAGS.name}_fold{fold}.log'))
        ch = logging.StreamHandler()
        fh.setFormatter(formatter);
        ch.setFormatter(formatter)
        logger.addHandler(fh);
        logger.addHandler(ch)
    return logger


def plot_loss_curve(train_losses, val_losses, save_path, fold):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title(f'Training and Validation Loss (Fold {fold})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend();
    plt.grid(True)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{FLAGS.name}_loss_fold{fold}.png'))
    plt.close()


def custom_collate_fn(batch):
    return batch[0]  # 单样本 Batch

def count_class_distribution(dataloader):
    # dataloader yields (s, V, edge_index, edge_attr, labels)
    zero = one = 0
    for sample in dataloader:
        _, _, _, _, y = sample
        # y might be float tensor
        y = y.flatten()
        zero += (y == 0).sum().item()
        one += (y == 1).sum().item()
    return [zero, one]

def find_best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    if best_idx >= len(thresholds):
        return 0.5, f1_scores[best_idx]
    return thresholds[best_idx], f1_scores[best_idx]


def _edge_index_to_device(edge_index, device):
    if torch.is_tensor(edge_index): return edge_index.to(device)
    return edge_index  # 已经是 tensor


# ========== 核心训练/评估循环 ==========

def evaluate_model(model, val_loader, loss_fn, device):
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            # 解包数据
            s, V, edge_index, edge_attr, y = batch

            # 移动到 GPU
            s = s.to(device)
            V = V.to(device)
            edge_index = edge_index.to(device)
            y = y.to(device)

            # 解包 edge_attr (tuple: s_e, V_e)
            s_e = edge_attr[0].to(device)
            V_e = edge_attr[1].to(device)

            with autocast():
                logits, _ = model(s, V, edge_index, (s_e, V_e))
                loss = loss_fn(logits, y.float())

            val_loss += loss.item()
            val_preds.extend(torch.sigmoid(logits).float().cpu().numpy())
            val_labels.extend(y.cpu().numpy())

    return np.array(val_preds), np.array(val_labels), val_loss / len(val_loader)


def train_epoch(model, loss_fn, train_loader, optimizer, scheduler, device, scaler, grad_accum_steps):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        s, V, edge_index, edge_attr, y = batch
        s, V, y = s.to(device), V.to(device), y.to(device)
        edge_index = edge_index.to(device)
        s_e, V_e = edge_attr[0].to(device), edge_attr[1].to(device)

        # === [新增]：训练时对标量特征加入微小的高斯噪声 ===
        if model.training:
            # 强度 0.01 的噪声
            noise = torch.randn_like(s) * 0.01
            s = s + noise
            V_noise = torch.randn_like(V) * 0.1
            V = V + V_noise
            if np.random.rand() < 0.2:
                V = torch.zeros_like(V)

        with autocast():
            logits, _ = model(s, V, edge_index, (s_e, V_e))
            loss = loss_fn(logits, y.float())
            loss = loss / grad_accum_steps

        if torch.isnan(loss): continue

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler: scheduler.step()

        train_loss += loss.item() * grad_accum_steps

    return train_loss / len(train_loader)


# ========== 主程序 ==========
def main(FLAGS):
    dataset = buildGraph(FLAGS.indir, strict_length_check=False)
    kfold = KFold(n_splits=5, shuffle=True, random_state=FLAGS.seed)

    # 路径设置
    log_dir = os.path.join(FLAGS.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(FLAGS.save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    plot_dir = os.path.join(FLAGS.save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 解析 scales 字符串 "5,10,16" -> [5, 10, 16]
    scales_list = [int(x) for x in FLAGS.scales.split(',')]

    thresholds_list = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logger = setup_logger(log_dir, fold + 1)
        logger.info(f"=== Fold {fold + 1} ===")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # 简单计算正负样本比
        tmp_loader = DataLoader(train_subset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
        zero, one = count_class_distribution(tmp_loader)
        samples_per_class = [zero, one]  # [负样本数, 正样本数]
        # logger.info(f"Class counts (train): zero={zero}, one={one}")

        # 计算正样本权重
        if one == 0:
            pos_weight_value = 1.0
            logger.warning("No positive samples in this fold's training set; using pos_weight=1.0")
        else:
            pos_weight_value = float(zero) / max(1.0, float(one))
            # clamp to reasonable range
            pos_weight_value = max(1.0, min(pos_weight_value, 50.0))
        pos_weight = torch.tensor([pos_weight_value], device=FLAGS.device)

        # 初始化模型
        model = MultiScaleGVPBindingPredictor(
            orig_node_dims=(FLAGS.orig_s_dim, FLAGS.orig_v_dim),
            hidden_dim=FLAGS.hidden_nf,
            edge_dims=(FLAGS.edge_s_dim, FLAGS.edge_v_dim),  # (32, 3)
            n_layers=FLAGS.num_layers,
            dropout=FLAGS.dropout,
            scales=scales_list,  # [5, 10, 16]
            # task_type='RNA'
        ).to(FLAGS.device)

        logger.info(f"Model Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        optimizer = optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

        # 学习率调度: Warmup -> Cosine
        # Warmup
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=FLAGS.warmup_epochs * len(
            train_subset) // FLAGS.gradient_accumulation)
        # Cosine
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FLAGS.cosine_epochs * len(
            train_subset) // FLAGS.gradient_accumulation, eta_min=FLAGS.min_lr)
        # 串联
        # 注意: SequentialLR 是按 epoch 还是 step 取决于你怎么调用。通常 step 级调度更平滑。
        # 这里为了简单，我们在 train_epoch 里每一步都 step，所以 total_iters 设为 steps。
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[
            FLAGS.warmup_epochs * len(train_subset) // FLAGS.gradient_accumulation])

        # Loss
        if FLAGS.use_focal_loss:
            loss_fn = FocalLoss(samples_per_class=samples_per_class, gamma=FLAGS.focal_gamma, reduction='mean', device=FLAGS.device)
        else:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        scaler = GradScaler()
        early_stopping = EarlyStopping(patience=FLAGS.patience, delta=FLAGS.early_delta, mode='max')

        best_auprc, best_model_state, best_threshold = 0.0, None, 0.5
        train_losses, val_losses = [], []

        for epoch in range(FLAGS.epochs):
            loss = train_epoch(model, loss_fn,
                               DataLoader(train_subset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn),
                               optimizer, scheduler, FLAGS.device, scaler, FLAGS.gradient_accumulation)

            preds, labels, val_loss = evaluate_model(model,
                                                     DataLoader(val_subset, batch_size=1, collate_fn=custom_collate_fn),
                                                     loss_fn, FLAGS.device)

            # Metrics
            if len(labels) > 0:
                auprc = average_precision_score(labels, preds)
                threshold, f1_at_threshold = find_best_threshold(labels, preds)
                binary_preds = (preds >= threshold).astype(int)
                acc = accuracy_score(labels, binary_preds)
                prec = precision_score(labels, binary_preds, zero_division=0)
                rec = recall_score(labels, binary_preds, zero_division=0)
                f1 = f1_score(labels, binary_preds, zero_division=0)
                mcc = matthews_corrcoef(labels, binary_preds)
                roc_auc = roc_auc_score(labels, preds)
            else:
                auprc, f1, threshold, f1_at_threshold, acc, prec, rec, mcc, roc_auc= 0, 0, 0.5, 0, 0, 0, 0, 0, 0

            train_losses.append(loss)
            val_losses.append(val_loss)

            # 计算特异性
            tn = ((binary_preds == 0) & (labels == 0)).sum()
            fp = ((binary_preds == 1) & (labels == 0)).sum()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            logger.info(
                f'Epoch {epoch + 1:03d}: '
                f'Train={loss:.4f}, Val={val_loss:.4f}, '
                f'LR={optimizer.param_groups[0]["lr"]:.2e}, '
                f'Threshold={threshold:.4f}, Optimal F1={f1_at_threshold:.4f}, '
                f'AUPRC={auprc:.4f}, F1={f1:.4f}, '
                f'Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, '
                f'Spe={specificity:.4f}, MCC={mcc:.4f}, ROC_AUC={roc_auc:.4f}'
            )

            if auprc > best_auprc:
                best_auprc = auprc
                best_model_state = model.state_dict()
                best_threshold = threshold
                best_epoch = epoch + 1
                save_path = os.path.join(model_dir, f'{FLAGS.name}_fold{fold + 1}_best.pt')
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'auprc': best_auprc,
                    'threshold': best_threshold,
                    'fold': fold + 1
                }, save_path)
                logger.info(f"Saved best model to {save_path}")

            if early_stopping(auprc):
                logger.info(f'Early stopping triggered at epoch {epoch + 1}!')
                break

        thresholds_list.append(best_threshold)

        plot_loss_curve(train_losses, val_losses, plot_dir, fold + 1)
        del model, optimizer
        torch.cuda.empty_cache()

    threshold_file_path = os.path.join(FLAGS.save_dir, f'{FLAGS.name}_thresholds.txt')
    with open(threshold_file_path, 'w') as f:
        for i, threshold in enumerate(thresholds_list):
            f.write(f'Fold {i + 1}: {threshold:.4f}\n')


if __name__ == '__main__':
    FLAGS, _ = get_optimized_config()
    FLAGS.name = f"MultiScaleGVP_esm3_S{FLAGS.scales}_H{FLAGS.hidden_nf}_{datetime.now().strftime('%m%d_%H%M')}"
    FLAGS.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(FLAGS)
