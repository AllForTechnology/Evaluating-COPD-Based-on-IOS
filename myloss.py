from sklearn.metrics import roc_curve, auc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from scipy.stats import norm
import time


class FeatureAndClassWeightedBCELoss(nn.Module):
    def __init__(self, weight_positive, weight_negative, reduction='mean'):
        super(FeatureAndClassWeightedBCELoss, self).__init__()
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative
        self.reduction = reduction

    def forward(self, inputs, targets, feature_weights):
        # 确保 inputs 和 targets 的形状匹配，并处理 (batch_size, 1) 的情况
        if inputs.dim() > 1 and inputs.shape[-1] == 1:
            inputs = inputs.squeeze(-1)

        # 确保 inputs, targets, feature_weights 形状一致
        if inputs.dim() != 2 or inputs.shape[1] != 2:
            raise ValueError(f"Inputs must have shape (batch_size, 2). Got {inputs.shape}")
        if targets.dim() != 1 or targets.shape[0] != inputs.shape[0]:
            raise ValueError(f"Targets must have shape (batch_size,). Got {targets.shape}")
        if feature_weights.dim() != 1 or feature_weights.shape[0] != inputs.shape[0]:
            raise ValueError(f"Feature_weights must have shape (batch_size,). Got {feature_weights.shape}")

        for i in range(len(feature_weights)):
            if feature_weights[i] < 70:
                feature_weights[i] = norm.pdf(feature_weights[i].cpu(), 70, 20) * 100
            else:
                feature_weights[i] = 1
        per_sample_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 2. 为每个样本计算基于类别的权重
        # 使用 torch.where 根据 targets (0.0 或 1.0) 选择对应的类别权重
        # 确保权重 tensor 和 targets 在同一个设备上
        class_weights_per_sample = torch.where(
            targets == 1,  # 如果是阳性样本 (标签为 1.0)
            torch.tensor(self.weight_positive, device=targets.device),  # 使用阳性权重
            torch.tensor(self.weight_negative, device=targets.device)  # 否则使用阴性权重
        )

        # 3. 计算每个样本的总权重：类别权重 * 特征权重
        # 确保 feature_weights 是正数，避免负权重或零权重导致问题
        if torch.any(feature_weights <= 0):
             print("Warning: feature_weights should be positive. Clamping to a small positive value.")
             feature_weights = torch.clamp(feature_weights, min=1e-6)  # 避免零或负权重

        final_sample_weights = class_weights_per_sample * feature_weights

        # 4. 将基础损失乘以总权重
        weighted_per_sample_loss = per_sample_loss * final_sample_weights

        # 5. 应用最终的 reduction
        if self.reduction == 'mean':
            return torch.mean(weighted_per_sample_loss)
        elif self.reduction == 'sum':
            return torch.sum(weighted_per_sample_loss)
        else:
            return weighted_per_sample_loss  # 返回每个样本的加权损失



