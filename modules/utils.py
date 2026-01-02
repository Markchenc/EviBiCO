"""
工具函数模块 - 权重计算、评估函数
"""

import torch
import numpy as np
from typing import List, Optional
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score


def calculate_accuracy_detailed(predictions, targets, dimension: Optional[int] = None):
    """
    计算多标签分类的准确率（返回总体和每个维度的准确率）

    Args:
        predictions: 模型预测
        targets: 真实标签
        dimension: 指定维度（可选）

    Returns:
        accuracy: 准确率或详细准确率
    """
    pred_classes = torch.argmax(predictions, dim=2)
    target_reshaped = targets.view(targets.size(0), 5, 3)
    target_classes = torch.argmax(target_reshaped, dim=2)

    correct_per_dimension = (pred_classes == target_classes)

    overall_accuracy = correct_per_dimension.float().mean().item()

    accuracy_per_dimension = correct_per_dimension.float().mean(dim=0)

    accuracy_per_dimension = [float(acc) for acc in accuracy_per_dimension]

    if dimension is not None:
        return float(accuracy_per_dimension[dimension])
    else:
        return {
            'overall_accuracy': overall_accuracy,
            'accuracy_per_dimension': accuracy_per_dimension,
        }


def calculate_evidence_f1_score(true_evidence, pred_evidence):
    """
    计算单个图的证据句F1分数和准确率

    Args:
        true_evidence: 真实证据句标签，格式为 {'openness': [0, 2], 'conscientiousness': [1], ...}
        pred_evidence: 预测证据句结果，格式为 {'openness': 0/1张量, 'conscientiousness': 0/1张量, ...}
                      每个维度对应的是一个长度为对话话语数量的一维张量

    Returns:
        dict: 包含每个维度的F1分数、准确率和总体指标
    """
    personality_dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

    dimension_metrics = {}

    first_dim = personality_dims[0]
    if first_dim not in pred_evidence:
        print("预测结果为空，返回零值")
        return {
            'dimension_metrics': {dim: {'f1': 0.0, 'weighted_f1': 0.0, 'accuracy': 0.0} for dim in personality_dims},
            'avg_f1_score': 0.0,
            'avg_accuracy': 0.0
        }

    num_utterances = pred_evidence[first_dim].size(0)
    device = pred_evidence[first_dim].device

    all_predictions = []
    all_labels = []

    for dim_name in personality_dims:
        true_tensor = torch.zeros(num_utterances, dtype=torch.long, device=device)
        evidence_indices = true_evidence.get(dim_name, [])

        for idx_str in evidence_indices:
            if not idx_str or idx_str.strip() == '':
                continue
            for idx_part in idx_str.split(','):
                try:
                    idx_part = idx_part.strip()
                    if idx_part:
                        idx_int = int(idx_part) - 1
                        if 0 <= idx_int < num_utterances:
                            true_tensor[idx_int] = 1
                except (ValueError, TypeError):
                    continue

        pred_tensor = pred_evidence.get(dim_name, torch.zeros(num_utterances, dtype=torch.long, device=device))

        true_np = true_tensor.cpu().numpy()
        pred_np = pred_tensor.cpu().numpy()

        accuracy = accuracy_score(true_np, pred_np)
        f1_macro = f1_score(true_np, pred_np, average='binary', zero_division=0)
        f1_weighted = f1_score(true_np, pred_np, average='weighted', zero_division=0)

        dimension_metrics[dim_name] = {
            'f1': f1_macro,
            'weighted_f1': f1_weighted,
            'accuracy': accuracy
        }

        all_predictions.extend(pred_np.tolist())
        all_labels.extend(true_np.tolist())

    avg_f1 = np.mean([m['f1'] for m in dimension_metrics.values()])
    avg_weighted_f1 = np.mean([m['weighted_f1'] for m in dimension_metrics.values()])
    avg_accuracy = np.mean([m['accuracy'] for m in dimension_metrics.values()])

    overall_weighted_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    overall_accuracy = accuracy_score(all_labels, all_predictions)

    return {
        'dimension_metrics': dimension_metrics,
        'dimension_f1_scores': {dim: m['f1'] for dim, m in dimension_metrics.items()},
        'avg_f1_score': avg_f1,
        'avg_weighted_f1': avg_weighted_f1,
        'avg_accuracy': avg_accuracy,
        'overall_weighted_f1': overall_weighted_f1,
        'overall_accuracy': overall_accuracy
    }
