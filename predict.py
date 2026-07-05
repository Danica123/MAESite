import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix
)
from torch.utils.data import DataLoader
from multiscale_gvp import MultiScaleGVPBindingPredictor
from dataloader_test import TestGraphDataset
import faulthandler
import glob

faulthandler.enable()
warnings.simplefilter(action='ignore', category=FutureWarning)

# 设置环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
torch.backends.cudnn.benchmark = True


def read_true_labels(label_file):
    """读取真实标签（如果存在）"""
    true_labels = {}
    if not os.path.exists(label_file):
        print(f"[WARNING] Label file not found: {label_file}")
        return true_labels

    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            if lines[i].startswith('>'):
                name = lines[i].strip()[1:]
                if i + 2 < len(lines):
                    seq = lines[i + 1].strip()
                    label_str = lines[i + 2].strip()
                    labels = list(map(int, list(label_str)))
                    true_labels[name] = labels
                    i += 3
                else:
                    i += 1
            else:
                i += 1
        print(f"[INFO] Loaded {len(true_labels)} true labels")
    except Exception as e:
        print(f"[ERROR] Failed to read label file: {e}")

    return true_labels


def load_thresholds(threshold_file):
    """从阈值文件加载阈值"""
    thresholds = {}
    if not os.path.exists(threshold_file):
        print(f"[WARNING] Threshold file not found: {threshold_file}")
        return thresholds

    try:
        with open(threshold_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    key, value = line.split(':')
                    key = key.strip()
                    try:
                        value = float(value.strip())
                        if 'fold' in key.lower():
                            # 格式: "Fold 1: 0.1234"
                            fold_num = int(''.join(filter(str.isdigit, key)))
                            thresholds[fold_num] = value
                        else:
                            thresholds[key] = value
                    except ValueError:
                        continue

        if not thresholds:
            # 尝试直接加载为数组
            thresholds_array = np.loadtxt(threshold_file)
            if thresholds_array.ndim == 0:
                thresholds = {i + 1: float(thresholds_array) for i in range(5)}
            else:
                thresholds = {i + 1: float(thresholds_array[i]) for i in range(len(thresholds_array))}

        print(f"[INFO] Loaded thresholds: {thresholds}")
        return thresholds

    except Exception as e:
        print(f"[ERROR] Failed to load thresholds: {e}")
        return {}


def find_model_files(model_dir, pattern="*fold*best.pt"):
    """查找模型文件"""
    model_files = glob.glob(os.path.join(model_dir, pattern))

    if not model_files:
        # 尝试其他模式
        model_files = glob.glob(os.path.join(model_dir, "*best*.pt"))
        model_files += glob.glob(os.path.join(model_dir, "*fold*.pt"))

    # 按fold编号排序
    model_files.sort(key=lambda x: (
        int(''.join(filter(str.isdigit, os.path.basename(x).split('fold')[1].split('_')[0])))
        if 'fold' in os.path.basename(x).lower() else 0
    ))

    return model_files


def collate_test_fn(batch):
    """测试数据collate函数"""
    # batch是单个样本，直接返回
    if isinstance(batch, list) and len(batch) == 1:
        return batch[0]
    return batch


def predict_single_model(model, dataloader, device, threshold=0.5):
    """使用单个模型进行预测"""
    model.eval()
    all_predictions = {}

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 5:  # (name, s, V, edge_index, edge_attr)
                target_name, s, V, edge_index, edge_attr = batch
            else:
                print(f"[WARNING] Unexpected batch format: {len(batch)} elements")
                continue

            # 移动到设备
            s = s.to(device)
            V = V.to(device)

            # 处理边索引
            if isinstance(edge_index, (list, tuple)):
                if torch.is_tensor(edge_index[0]):
                    edge_index = edge_index[0].to(device)
                else:
                    edge_index = torch.tensor(edge_index[0], dtype=torch.long, device=device)
            else:
                edge_index = edge_index.to(device)

            # 处理边特征
            if isinstance(edge_attr, (list, tuple)) and len(edge_attr) == 2:
                s_e, V_e = edge_attr
                s_e = s_e.to(device) if torch.is_tensor(s_e) else torch.tensor(s_e, device=device)
                V_e = V_e.to(device) if torch.is_tensor(V_e) else torch.tensor(V_e, device=device)
            else:
                print(f"[WARNING] Unexpected edge_attr format for {target_name}")
                continue

            try:
                # 模型预测
                logits, _ = model(s, V, edge_index, (s_e, V_e))
                pred_probs = torch.sigmoid(logits).cpu().numpy()

                # 二值化预测
                pred_labels = (pred_probs >= threshold).astype(int)

                all_predictions[target_name] = {
                    'probs': pred_probs,
                    'labels': pred_labels,
                    'logits': logits.cpu().numpy() if torch.is_tensor(logits) else logits
                }

                # print(f"[INFO] Predicted {target_name}: {len(pred_probs)} residues, "
                #       f"positive rate: {pred_labels.mean():.3f}")

            except Exception as e:
                print(f"[ERROR] Failed to predict {target_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return all_predictions


def ensemble_predictions(all_predictions_list):
    """集成多个模型的预测结果"""
    ensemble_results = {}

    # 收集所有模型的预测
    for fold_idx, predictions in enumerate(all_predictions_list):
        for target_name, pred_dict in predictions.items():
            if target_name not in ensemble_results:
                ensemble_results[target_name] = {
                    'probs_list': [],
                    'labels_list': []
                }
            ensemble_results[target_name]['probs_list'].append(pred_dict['probs'])
            ensemble_results[target_name]['labels_list'].append(pred_dict['labels'])

    # 计算平均值
    for target_name, data in ensemble_results.items():
        probs_array = np.array(data['probs_list'])  # [n_models, n_residues]
        avg_probs = probs_array.mean(axis=0)
        std_probs = probs_array.std(axis=0)

        # 平均阈值进行二值化
        labels_array = np.array(data['labels_list'])
        avg_labels = (labels_array.mean(axis=0) > 0.5).astype(int)

        ensemble_results[target_name] = {
            'avg_probs': avg_probs,
            'std_probs': std_probs,
            'avg_labels': avg_labels,
            'n_models': len(data['probs_list'])
        }

    return ensemble_results


def evaluate_predictions(predictions, true_labels, threshold=0.5):
    """评估预测结果（如果有真实标签）"""
    if not true_labels:
        print("[INFO] No true labels provided, skipping evaluation")
        return None

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for target_name, pred_data in predictions.items():
        if target_name in true_labels:
            y_true = np.array(true_labels[target_name])
            y_prob = pred_data['avg_probs'] if 'avg_probs' in pred_data else pred_data['probs']

            # 确保长度匹配
            min_len = min(len(y_true), len(y_prob))
            if min_len == 0:
                continue

            y_true = y_true[:min_len]
            y_prob = y_prob[:min_len]

            # 二值化
            y_pred = (y_prob >= threshold).astype(int)

            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob)

    if not all_y_true:
        print("[WARNING] No matching targets for evaluation")
        return None

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    # 计算指标
    metrics = {}

    # 基础指标
    metrics['acc'] = accuracy_score(all_y_true, all_y_pred)
    metrics['prec'] = precision_score(all_y_true, all_y_pred, zero_division=0)
    metrics['rec'] = recall_score(all_y_true, all_y_pred, zero_division=0)
    metrics['f1'] = f1_score(all_y_true, all_y_pred, zero_division=0)

    # MCC
    if len(np.unique(all_y_true)) > 1:
        metrics['mcc'] = matthews_corrcoef(all_y_true, all_y_pred)
    else:
        metrics['mcc'] = 0.0

    # AUC指标
    try:
        metrics['auroc'] = roc_auc_score(all_y_true, all_y_prob)
    except:
        metrics['auroc'] = 0.0

    try:
        metrics['auprc'] = average_precision_score(all_y_true, all_y_prob)
    except:
        metrics['auprc'] = 0.0

    # 混淆矩阵
    cm = confusion_matrix(all_y_true, all_y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['tp'] = tp
        metrics['spe'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        metrics['tn'] = metrics['fp'] = metrics['fn'] = metrics['tp'] = 0
        metrics['spe'] = 0.0

    return metrics


def save_predictions(predictions, output_dir, fold=None, ensemble=False):
    """保存预测结果"""
    if fold is not None:
        fold_dir = os.path.join(output_dir, f"fold{fold}")
    else:
        fold_dir = output_dir

    os.makedirs(fold_dir, exist_ok=True)

    for target_name, pred_data in predictions.items():
        if ensemble:
            # 保存ensemble结果
            probs = pred_data['avg_probs']
            labels = pred_data['avg_labels']
            stds = pred_data.get('std_probs', np.zeros_like(probs))

            # 保存概率和标准差
            result = np.column_stack([probs, stds, labels])
            header = "probability,std,label"
            fmt = "%.6f,%.6f,%d"

            filename = os.path.join(fold_dir, f"{target_name}_ensemble_pred.csv")
        else:
            # 保存单个fold结果
            probs = pred_data['probs']
            labels = pred_data['labels']

            result = np.column_stack([probs, labels])
            header = "probability,label"
            fmt = "%.6f,%d"

            filename = os.path.join(fold_dir, f"{target_name}_fold{fold}_pred.csv")

        np.savetxt(filename, result, fmt=fmt, header=header, comments='')

    print(f"[INFO] Predictions saved to {fold_dir}")


def plot_metrics(metrics_dict, output_dir):
    """绘制评估指标"""
    if not metrics_dict:
        return

    # 提取不同fold的指标
    folds = list(metrics_dict.keys())
    metric_names = ['acc', 'prec', 'rec', 'f1', 'mcc', 'auroc', 'auprc', 'spe']

    # 创建指标表格
    data = []
    for fold, metrics in metrics_dict.items():
        row = {'Fold': f"Fold{fold}"}
        for name in metric_names:
            if name in metrics:
                row[name] = metrics[name]
        data.append(row)

    df = pd.DataFrame(data)

    # 保存为CSV
    csv_path = os.path.join(output_dir, "fold_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Metrics saved to {csv_path}")

    # 绘制条形图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(metric_names[:8]):
        if metric in df.columns:
            ax = axes[idx]
            df.plot(x='Fold', y=metric, kind='bar', ax=ax, legend=False)
            ax.set_title(metric.upper())
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "metrics_summary.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"[INFO] Metrics plot saved to {plot_path}")


def main(args):
    """主函数"""
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)

    # 读取真实标签（如果存在）
    true_labels = read_true_labels(args.label_file) if args.label_file else {}

    # 加载阈值
    thresholds = load_thresholds(args.threshold_file) if args.threshold_file else {}

    # 查找模型文件
    model_files = find_model_files(args.model_dir)
    if not model_files:
        print(f"[ERROR] No model files found in {args.model_dir}")
        print(f"[INFO] Looking for files matching pattern: *fold*best.pt")
        return

    print(f"[INFO] Found {len(model_files)} model files:")
    for m in model_files:
        print(f"  - {os.path.basename(m)}")

    # 加载测试数据
    print(f"\n[INFO] Loading test data from {args.indir}")
    try:
        dataset = TestGraphDataset(args.indir)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_test_fn,
            num_workers=min(4, os.cpu_count() // 2)
        )
        print(f"[INFO] Loaded {len(dataset)} test samples")
    except Exception as e:
        print(f"[ERROR] Failed to load test data: {e}")
        return

    # 存储所有fold的预测结果
    all_fold_predictions = []
    all_fold_metrics = {}

    # 对每个fold的模型进行预测
    for fold_idx, model_path in enumerate(model_files):
        fold_num = fold_idx + 1

        print(f"\n{'=' * 60}")
        print(f"Processing Fold {fold_num}")
        print(f"{'=' * 60}")

        # 确定阈值
        threshold = thresholds.get(fold_num, thresholds.get('default', 0.5))
        print(f"[INFO] Using threshold: {threshold:.4f}")

        # 初始化模型
        model = MultiScaleGVPBindingPredictor(
            orig_node_dims=(1889, 3),
            hidden_dim= 256,
            edge_dims=(32, 3),  # 这里的边特征维度固定为 32 (16 RBF + 16 Pos)
            n_layers=3,
            dropout=0.0,  # 推理时不需要 dropout
            scales=[5, 10, 16]
        ).to(device)

        # 加载模型权重
        try:
            checkpoint = torch.load(model_path, map_location=device)

            # 处理不同的checkpoint格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # 移除可能的"module."前缀（如果使用DataParallel训练）
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v

                model.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

            print(f"[INFO] Successfully loaded model: {os.path.basename(model_path)}")

        except Exception as e:
            print(f"[ERROR] Failed to load model {model_path}: {e}")
            continue

        # 进行预测
        fold_predictions = predict_single_model(
            model, dataloader, device, threshold=threshold
        )

        if not fold_predictions:
            print(f"[WARNING] No predictions for fold {fold_num}")
            continue

        # 保存单个fold的预测结果
        save_predictions(fold_predictions, args.outdir, fold=fold_num, ensemble=False)

        # 评估（如果有真实标签）
        if true_labels:
            fold_metrics = evaluate_predictions(fold_predictions, true_labels, threshold)
            if fold_metrics:
                all_fold_metrics[fold_num] = fold_metrics

                print(f"\nFold {fold_num} Metrics:")
                print(f"  Accuracy:    {fold_metrics['acc']:.4f}")
                print(f"  Precision:   {fold_metrics['prec']:.4f}")
                print(f"  Recall:      {fold_metrics['rec']:.4f}")
                print(f"  Specificity: {fold_metrics['spe']:.4f}")
                print(f"  F1 Score:    {fold_metrics['f1']:.4f}")
                print(f"  MCC:         {fold_metrics['mcc']:.4f}")
                print(f"  AUPRC:       {fold_metrics['auprc']:.4f}")
                print(f"  AUROC:       {fold_metrics['auroc']:.4f}")

        # 存储fold预测结果用于ensemble
        all_fold_predictions.append(fold_predictions)

        # 清理GPU内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Ensemble预测（如果有多于一个fold）
    if len(all_fold_predictions) > 1:
        print(f"\n{'=' * 60}")
        print(f"Ensemble Prediction")
        print(f"{'=' * 60}")

        ensemble_results = ensemble_predictions(all_fold_predictions)

        if ensemble_results:
            # 保存ensemble结果
            save_predictions(ensemble_results, args.outdir, fold=None, ensemble=True)

            # 评估ensemble结果
            if true_labels:
                ensemble_threshold = 0.5  # ensemble使用0.5阈值
                ensemble_metrics = evaluate_predictions(
                    ensemble_results, true_labels, threshold=ensemble_threshold
                )

                if ensemble_metrics:
                    all_fold_metrics['ensemble'] = ensemble_metrics

                    print(f"\nEnsemble Metrics:")
                    print(f"  Accuracy:    {ensemble_metrics['acc']:.4f}")
                    print(f"  Precision:   {ensemble_metrics['prec']:.4f}")
                    print(f"  Recall:      {ensemble_metrics['rec']:.4f}")
                    print(f"  Specificity: {ensemble_metrics['spe']:.4f}")
                    print(f"  F1 Score:    {ensemble_metrics['f1']:.4f}")
                    print(f"  MCC:         {ensemble_metrics['mcc']:.4f}")
                    print(f"  AUPRC:       {ensemble_metrics['auprc']:.4f}")
                    print(f"  AUROC:       {ensemble_metrics['auroc']:.4f}")

    # 绘制和保存指标
    if all_fold_metrics:
        plot_metrics(all_fold_metrics, args.outdir)

        # 计算平均指标
        if len(all_fold_metrics) > 1:
            metric_keys = ['acc', 'prec', 'rec', 'f1', 'mcc', 'auroc', 'auprc', 'spe']
            fold_keys = [k for k in all_fold_metrics.keys() if isinstance(k, int)]

            if fold_keys:
                print(f"\n{'=' * 60}")
                print(f"Average Metrics across Folds")
                print(f"{'=' * 60}")

                for metric in metric_keys:
                    values = [all_fold_metrics[fold][metric] for fold in fold_keys if metric in all_fold_metrics[fold]]
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        print(f"{metric.upper():8s}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"\n{'=' * 60}")
    print(f"Prediction Complete!")
    print(f"Results saved to: {args.outdir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict protein-nucleic acid binding sites using trained multiscale GVP-GNN model')

    parser.add_argument('--model_dir', type=str, default="/home2/2023/23dhh2/GTSite/model/DNA/models",
                        # /home2/dhh/GTSite/MAGGN/model/RNA/models
                        help='Directory containing trained model weights (*fold*best.pt files)')
    parser.add_argument('--indir', type=str,
                        default="/pubssd/dhh/GTSite/PreprocessingDNA/DNA_test_181_Preprocessing_using_native",
                        # /pubssd/dhh/GTSitedata/PreprocessingRNA/RNA_test_117_Preprocessing_using_native
                        help='Input directory with test data (must contain input/ and distmaps/ subdirectories)')
    parser.add_argument('--outdir', type=str,
                        default="/home2/2023/23dhh2/GTSite/predict_result/DNA_native/DNA_181/MultiScaleGVP_esm3_S5,10,16_H256_0127_1116",
                        # /home2/dhh/GTSite/MAGGN/results/RNA_native/RNA_117/
                        help='Output directory for predictions')
    parser.add_argument('--label_file', type=str, default='/pubssd/dhh/GTSite/PreprocessingDNA/DNA-181_Test.txt',
                        # /pubssd/dhh/GTSitedata/PreprocessingRNA/RNA-117_Test.txt
                        help='Path to true label file (optional, for evaluation only)')
    parser.add_argument('--threshold_file', type=str,
                        default='/home2/2023/23dhh2/GTSite/model/DNA/MultiScaleGVP_esm3_S5,10,16_H256_0127_1116_thresholds.txt',
                        # /home2/dhh/GTSite/MAGGN/model/RNA
                        help='Path to threshold file from training (optional)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, or cpu)')

    args = parser.parse_args()

    # 检查必需参数
    if not os.path.exists(args.model_dir):
        print(f"[ERROR] Model directory not found: {args.model_dir}")
        sys.exit(1)

    if not os.path.exists(args.indir):
        print(f"[ERROR] Input directory not found: {args.indir}")
        sys.exit(1)

    main(args)