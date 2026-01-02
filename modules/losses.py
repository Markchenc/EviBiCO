"""
损失函数模块 - BigFiveLoss和EvidenceWeightLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EvidenceWeightLoss(nn.Module):
    """
    证据句损失函数

    核心思路：
    - 对每个人格维度的每个发言节点，视为一个独立的二分类任务
    - 使用BCE损失函数计算每个二分类任务的损失
    - 确保每个二分类任务都能得到梯度更新
    """

    def __init__(self, weight, pos_weight=1.0, use_dynamic_pos_weight=False, max_pos_weight=10.0):
        """
        初始化证据句损失函数

        Args:
            weight: 证据句损失的权重系数
            pos_weight: 正样本权重（固定权重模式），>1表示对正样本更重视
            use_dynamic_pos_weight: 是否使用动态正样本权重（按批次自动计算neg/pos比例）
            max_pos_weight: 动态权重的上限，避免过大导致训练不稳定
        """
        super().__init__()
        self.weight = weight
        self.pos_weight = float(pos_weight)
        self.use_dynamic_pos_weight = bool(use_dynamic_pos_weight)
        self.max_pos_weight = float(max_pos_weight)

    def forward(self, evidence_predictions_raw, true_evidence_labels):
        """
        计算证据句损失

        Args:
            evidence_predictions_raw: 列表,包含5个一维张量,每个张量对应一个人格维度的证据句预测概率
                                     顺序为['开放性', '尽责性', '外向性', '宜人性', '神经质性']
                                     每个张量形状为 [num_utterance_nodes], 值为预测概率
            true_evidence_labels: 列表,包含5个一维张量,每个张量对应一个人格维度的真实标签
                                 顺序为['开放性', '尽责性', '外向性', '宜人性', '神经质性']
                                 每个张量形状为 [num_utterance_nodes], 值为0或1

        Returns:
            total_loss: 总损失值
            metrics: 指标字典,包含损失统计信息
        """
        assert len(evidence_predictions_raw) == 5, f"预测列表长度应为5,实际为{len(evidence_predictions_raw)}"
        assert len(true_evidence_labels) == 5, f"标签列表长度应为5,实际为{len(true_evidence_labels)}"

        dimension_losses = []
        personality_dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

        for dim_idx, dim_name in enumerate(personality_dims):
            pred_probs = evidence_predictions_raw[dim_idx]
            true_labels = true_evidence_labels[dim_idx].float()

            if pred_probs.device != true_labels.device:
                true_labels = true_labels.to(pred_probs.device)

            if self.use_dynamic_pos_weight:
                pos_count = true_labels.sum().item()
                neg_count = true_labels.numel() - pos_count
                if pos_count > 0:
                    pos_weight_dim = min(max(neg_count / pos_count, 1.0), self.max_pos_weight)
                else:
                    pos_weight_dim = 1.0
            else:
                pos_weight_dim = self.pos_weight

            weight_tensor = torch.ones_like(true_labels, dtype=torch.float, device=pred_probs.device)
            weight_tensor = weight_tensor + (pos_weight_dim - 1.0) * true_labels

            device_type = 'cuda' if pred_probs.device.type == 'cuda' else 'cpu'
            with torch.autocast(device_type=device_type, enabled=False):
                bce_loss = F.binary_cross_entropy(
                    pred_probs.float().clamp(1e-7, 1.0 - 1e-7),
                    true_labels.float(),
                    weight=weight_tensor,
                    reduction='mean'
                )

            dimension_losses.append(bce_loss)

        total_loss = torch.stack(dimension_losses).mean()

        metrics = {
            'evidence_loss': total_loss.item(),
            'dimension_losses': {dim_name: loss.item() for dim_name, loss in zip(personality_dims, dimension_losses)}
        }

        return self.weight * total_loss, metrics


class BigFiveLoss(nn.Module):
    """大五人格损失函数，集成证据句权重损失"""

    def __init__(self, use_focal_loss: bool = False, pos_weight: Optional[float] = None, evidence_weight: float = 1.0):
        """
        初始化损失函数

        Args:
            use_focal_loss: 是否使用焦点损失
            pos_weight: 正样本权重
            evidence_weight: 证据句损失权重
        """
        super(BigFiveLoss, self).__init__()
        self.use_focal_loss = use_focal_loss
        self.pos_weight = pos_weight if pos_weight is not None else 1.0
        self.evidence_weight = evidence_weight

        self.evidence_loss_fn = EvidenceWeightLoss(
            weight=evidence_weight,
            pos_weight=self.pos_weight,
            use_dynamic_pos_weight=False,
            max_pos_weight=10.0
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, evidence_predictions_raw=None,
                true_evidence=None, focus_trait: Optional[int] = None, return_metrics: bool = False):
        """
        损失计算，支持证据句权重损失

        Args:
            predictions: 模型预测 [batch_size, 5, num_classes] 或 [5, num_classes] (非批处理)
            targets: 真实标签 [batch_size, 15] 或 [15] (非批处理)
            evidence_predictions_raw: GNN返回的证据句预测概率列表,包含5个一维张量
            true_evidence: 证据句标注字典 {dim: [indices]} 或 None
            focus_trait: 关注的人格维度
            return_metrics: 是否返回详细指标

        Returns:
            loss: 总损失（如果return_metrics=False）
            loss, metrics: 总损失和详细指标（如果return_metrics=True）
        """
        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, num_traits, num_classes = predictions.shape
        target_reshaped = targets.view(batch_size, 5, 3)

        if focus_trait is not None:
            pred_i = predictions[:, focus_trait, :]
            target_i = target_reshaped[:, focus_trait, :]
            target_indices = torch.argmax(target_i, dim=1)
            personality_loss = F.cross_entropy(pred_i, target_indices)
        else:
            losses = []
            for i in range(5):
                pred_i = predictions[:, i, :]
                target_i = target_reshaped[:, i, :]
                target_indices = torch.argmax(target_i, dim=1)
                loss_i = F.cross_entropy(pred_i, target_indices)
                losses.append(loss_i)
            personality_loss = torch.stack(losses).mean()

        total_loss = personality_loss
        metrics = {'personality_loss': personality_loss.item()}

        if evidence_predictions_raw is not None and true_evidence is not None:
            true_evidence_labels = self._convert_evidence_dict_to_tensors(
                true_evidence,
                evidence_predictions_raw[0].size(0),
                evidence_predictions_raw[0].device
            )

            evidence_loss, evidence_metrics = self.evidence_loss_fn(
                evidence_predictions_raw,
                true_evidence_labels
            )

            total_loss = total_loss + evidence_loss
            metrics.update(evidence_metrics)

        if squeeze_output:
            total_loss = total_loss.squeeze()

        if return_metrics:
            return total_loss, metrics
        else:
            return total_loss

    def _convert_evidence_dict_to_tensors(self, true_evidence, num_utterance_nodes, device):
        """
        将证据句字典格式转换为张量列表格式

        Args:
            true_evidence: 字典格式 {'开放性': [0, 2], '尽责性': [1], ...}
            num_utterance_nodes: 发言节点数量
            device: 设备

        Returns:
            张量列表,包含5个一维张量,每个张量形状为 [num_utterance_nodes]
        """
        personality_dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        true_evidence_labels = []

        for dim_name in personality_dims:
            true_vector = torch.zeros(num_utterance_nodes, dtype=torch.long, device=device)
            evidence_indices = true_evidence.get(dim_name, [])

            for idx_str in evidence_indices:
                if not idx_str or idx_str.strip() == '':
                    continue

                for idx_part in idx_str.split(','):
                    try:
                        idx_part = idx_part.strip()
                        if idx_part:
                            idx_int = int(idx_part) - 1
                            if 0 <= idx_int < num_utterance_nodes:
                                true_vector[idx_int] = 1
                    except (ValueError, TypeError):
                        continue

            true_evidence_labels.append(true_vector)

        return true_evidence_labels
