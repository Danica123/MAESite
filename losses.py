import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    通用 Focal Loss 实现
    支持二分类和多分类，支持手动 alpha 或自动 alpha
    """

    def __init__(self, gamma=2.0, alpha=None, samples_per_class=None, beta=0.999, reduction='mean', device='cuda:0'):
        """
        :param gamma: 聚焦参数
        :param alpha: 正样本权重 (float, e.g. 0.25) 或 类别权重列表 (list/tensor)
                      如果为 None 且提供了 samples_per_class，则自动计算。
        :param samples_per_class: [neg_count, pos_count] 用于自动计算 alpha
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device

        # 处理 Alpha
        if alpha is not None:
            # 手动指定 Alpha
            if isinstance(alpha, (float, int)):
                # 二分类: alpha 是正类权重，1-alpha 是负类权重
                self.alpha = torch.tensor([1 - alpha, alpha], dtype=torch.float32, device=device)
            else:
                self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        elif samples_per_class is not None:
            # 自动计算类别平衡权重 (Class Balanced)
            effective_num = 1.0 - np.power(beta, samples_per_class)
            weights = (1.0 - beta) / (effective_num + 1e-8)
            weights = weights / np.sum(weights) * len(samples_per_class)
            self.alpha = torch.tensor(weights, dtype=torch.float32, device=device)
        else:
            # 不使用权重
            self.alpha = None

    def forward(self, logits, targets):
        logits = logits.to(self.device)
        targets = targets.to(self.device)

        # 形状对齐: [N]
        if logits.dim() > 1: logits = logits.view(-1)
        if targets.dim() > 1: targets = targets.view(-1)

        # 过滤无效标签
        valid_mask = targets != -1
        logits = logits[valid_mask]
        targets = targets[valid_mask].float()

        if len(logits) == 0:
            return torch.tensor(0.0, device=self.device)

        # 计算概率 pt
        probs = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, probs, 1 - probs)  # 正样本取p, 负样本取1-p

        # Focal Term: (1 - pt)^gamma
        focal_weight = (1 - pt).pow(self.gamma)

        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        loss = focal_weight * bce_loss

        # Alpha Weighting
        if self.alpha is not None:
            # targets 0 -> alpha[0], targets 1 -> alpha[1]
            alpha_t = torch.where(targets >= 0.5, self.alpha[1], self.alpha[0])
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedBCELoss(nn.Module):
    """简单的加权BCE损失"""

    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__()
        # pos_weight 应该是一个 tensor [1]
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        if logits.dim() > 1: logits = logits.view(-1)
        if targets.dim() > 1: targets = targets.view(-1)

        return F.binary_cross_entropy_with_logits(
            logits, targets.float(),
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class CombinedLoss(nn.Module):
    """组合损失：BCE + Focal"""

    def __init__(self, pos_weight=None, focal_alpha=0.25, focal_gamma=2.0, bce_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.bce = WeightedBCELoss(pos_weight=pos_weight)
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.w_bce = bce_weight
        self.w_focal = focal_weight

    def forward(self, logits, targets):
        return self.w_bce * self.bce(logits, targets) + self.w_focal * self.focal(logits, targets)


class AdaptiveThresholdLoss(nn.Module):
    """自适应阈值损失"""

    def __init__(self, base_loss_fn, threshold_margin=0.1):
        super().__init__()
        self.base_loss = base_loss_fn
        self.margin = threshold_margin

    def forward(self, logits, targets):
        base_loss = self.base_loss(logits, targets)

        probs = torch.sigmoid(logits)
        targets = targets.float()

        # 惩罚那些在 (0.5 - margin, 0.5 + margin) 之间模糊不清的预测
        # 鼓励正样本 > 0.5+m, 负样本 < 0.5-m
        pos_mask = targets > 0.5
        neg_mask = targets < 0.5

        margin_loss = 0.0
        if pos_mask.any():
            margin_loss += F.relu(0.5 + self.margin - probs[pos_mask]).mean()
        if neg_mask.any():
            margin_loss += F.relu(probs[neg_mask] - (0.5 - self.margin)).mean()

        return base_loss + 0.1 * margin_loss
