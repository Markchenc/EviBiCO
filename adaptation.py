"""
测试时适应(Test-Time Adaptation)脚本
使用大模型的人格判断结果作为伪标签指导GNN模型参数更新

核心思路：
1. 加载预训练的GNN模型
2. 加载大模型对测试集的人格判断结果
3. 在测试时，使用大模型预测作为伪标签进行参数更新
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import os
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import copy
from sklearn.metrics import f1_score
import numpy as np

from modules import (
    BigFiveGNN,
    BigFiveDataset,
    ModelEvaluator,
    calculate_accuracy_detailed,
    download_model_from_modelscope,
    MODELSCOPE_MODEL_PATH
)

class LLMPseudoLabelLoader:
    """
    大模型伪标签加载器
    将大模型的人格判断结果转换为可用于训练的伪标签
    """

    DIM_ORDER = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

    def __init__(self, llm_results_path: str):
        """
        初始化加载器

        Args:
            llm_results_path: 大模型判断结果JSON文件路径
        """
        self.llm_results_path = llm_results_path
        self.results_dict = {}
        self._load_results()

    def _load_results(self):
        """加载并解析大模型判断结果"""
        print(f"正在加载大模型判断结果: {self.llm_results_path}")

        with open(self.llm_results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        detailed_results = data.get('detailed_results', [])

        for item in detailed_results:
            character_name = item['character_name']
            dialogue_id = item['dialogue_id']
            dimensions = item['dimensions']

            key = (character_name, str(dialogue_id))

            self.results_dict[key] = {}
            for dim_name, dim_info in dimensions.items():
                predicted = dim_info['predicted']
                self.results_dict[key][dim_name] = predicted

        print(f"加载完成，共 {len(self.results_dict)} 条大模型判断结果")

    def get_pseudo_label(self, character_name: str, dialogue_id: str) -> Optional[Dict[str, str]]:
        """
        获取指定对话的大模型伪标签

        Args:
            character_name: 角色名
            dialogue_id: 对话ID

        Returns:
            伪标签字典 {dim_name: predicted_level} 或 None
        """
        key = (character_name, str(dialogue_id))
        return self.results_dict.get(key, None)

    def convert_to_label_vector(self, pseudo_label: Dict[str, str]) -> torch.Tensor:
        """
        将大模型预测转换为15维标签向量

        Args:
            pseudo_label: {dim_name: predicted_level}

        Returns:
            15维one-hot标签张量
        """

        dim_order = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

        level_mapping = {
            '高': 0,
            '低': 1,
            '无法判断': 2
        }

        label_vector = torch.zeros(15, dtype=torch.float)

        for i, dim_name in enumerate(dim_order):
            predicted_level = pseudo_label.get(dim_name, '无法判断')
            level_idx = level_mapping.get(predicted_level, 2)

            label_vector[i * 3 + level_idx] = 1.0

        return label_vector

    def get_undetermined_dimensions(self, pseudo_label: Dict[str, str]) -> List[str]:
        """
        获取伪标签中"无法判断"的维度列表

        Args:
            pseudo_label: {dim_name: predicted_level}

        Returns:
            "无法判断"的维度名称列表
        """
        undetermined_dims = []
        for dim_name, predicted_level in pseudo_label.items():
            if predicted_level == '无法判断':
                undetermined_dims.append(dim_name)
        return undetermined_dims

def convert_utter_ids_to_evidence_labels(utter_ids: Dict, num_utterance_nodes: int, device: str = 'cpu') -> torch.Tensor:
    """
    将utter_ids字典转换为证据句标签张量

    Args:
        utter_ids: 原始证据句信息字典 {dim_name: [indices]}
        num_utterance_nodes: 发言节点数量
        device: 设备

    Returns:
        evidence_labels: 张量 [num_utterance_nodes, 5]
    """
    dim_order = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    evidence_labels_list = []

    for dim_name in dim_order:

        true_vector = torch.zeros(num_utterance_nodes, dtype=torch.float, device=device)

        evidence_indices = utter_ids.get(dim_name, [])

        for idx_str in evidence_indices:
            if not idx_str or (isinstance(idx_str, str) and idx_str.strip() == ''):
                continue

            if isinstance(idx_str, str):
                parts = idx_str.split(',')
            else:
                parts = [str(idx_str)]

            for idx_part in parts:
                try:
                    idx_part = str(idx_part).strip()
                    if idx_part:

                        idx_int = int(idx_part) - 1
                        if 0 <= idx_int < num_utterance_nodes:
                            true_vector[idx_int] = 1
                except (ValueError, TypeError):
                    continue

        evidence_labels_list.append(true_vector)

    evidence_labels = torch.stack(evidence_labels_list, dim=1)
    return evidence_labels

class PersonalityOnlyLoss(nn.Module):
    """
    仅人格分类损失函数
    只计算人格特征水平的交叉熵损失，不包含证据句损失
    """

    def __init__(self):
        super(PersonalityOnlyLoss, self).__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                focus_trait: Optional[int] = None) -> torch.Tensor:
        """
        计算人格分类损失

        Args:
            predictions: 模型预测 [batch_size, 5, 3] 或 [5, 3]
            targets: 伪标签 [batch_size, 15] 或 [15]
            focus_trait: 关注的人格维度（可选）

        Returns:
            loss: 人格分类损失
        """

        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)

        batch_size = predictions.shape[0]

        target_reshaped = targets.view(batch_size, 5, 3)

        if focus_trait is not None:

            pred_i = predictions[:, focus_trait, :]
            target_i = target_reshaped[:, focus_trait, :]
            target_indices = torch.argmax(target_i, dim=1)
            loss = F.cross_entropy(pred_i, target_indices)
        else:

            losses = []
            for i in range(5):
                pred_i = predictions[:, i, :]
                target_i = target_reshaped[:, i, :]
                target_indices = torch.argmax(target_i, dim=1)
                loss_i = F.cross_entropy(pred_i, target_indices)
                losses.append(loss_i)
            loss = torch.stack(losses).mean()

        return loss

def zero_undetermined_evidence(evidence_predictions: Dict[str, torch.Tensor],
                                undetermined_dims: List[str]) -> Dict[str, torch.Tensor]:
    """
    Args:
        evidence_predictions: 证据句预测字典 {dim_name: tensor}
        undetermined_dims: "无法判断"的维度名称列表

    Returns:
        处理后的证据句预测字典
    """
    zeroed_predictions = {}

    for dim_name, pred_tensor in evidence_predictions.items():
        if dim_name in undetermined_dims:

            zeroed_predictions[dim_name] = torch.zeros_like(pred_tensor)
        else:

            zeroed_predictions[dim_name] = pred_tensor

    return zeroed_predictions

class TestTimeAdaptationDynamicThreshold:
    """
    测试时适应类
    在测试时使用大模型伪标签进行模型参数更新
    """

    def __init__(self,
                 model_path: str,
                 llm_results_path: str,
                 data_path: str,
                 device: str = 'cuda',
                 adaptation_lr: float = 1e-5,
                 adaptation_steps: int = 1,
                 update_bert: bool = False,
                 threshold_predictor_lr: Optional[float] = None):
        """
        初始化测试时适应

        Args:
            model_path: 预训练模型路径
            llm_results_path: 大模型判断结果路径
            data_path: 数据集路径
            device: 设备
            adaptation_lr: 适应学习率（用于GCN和分类器）
            adaptation_steps: 每个样本的适应步数
            update_bert: 是否更新BERT参数
            threshold_predictor_lr: 阈值预测器的学习率（如果为None，则使用adaptation_lr）
        """
        self.model_path = model_path
        self.llm_results_path = llm_results_path
        self.data_path = data_path
        self.device = device
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps
        self.update_bert = update_bert
        self.threshold_predictor_lr = threshold_predictor_lr if threshold_predictor_lr is not None else adaptation_lr

        self.pseudo_label_loader = LLMPseudoLabelLoader(llm_results_path)
        self.criterion = PersonalityOnlyLoss()

        self.model = self._load_model()
        self.original_state_dict = copy.deepcopy(self.model.state_dict())

        self.dataset = BigFiveDataset(data_path=data_path, use_bert=True)

    def _load_model(self) -> BigFiveGNN:
        """加载预训练模型"""
        print(f"正在加载预训练的动态阈值模型: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']

        model = BigFiveGNN(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            num_layers=config['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        print("动态阈值模型加载完成")
        return model

    def _get_adaptation_optimizer(self) -> torch.optim.Optimizer:
        """
        获取适应阶段的优化器

        为不同的模块使用不同的学习率：
        - BERT: 如果update_bert为True，使用adaptation_lr
        - GCN和分类器: adaptation_lr
        - ThresholdPredictor: threshold_predictor_lr
        """
        param_groups = []

        if self.update_bert:

            bert_params = []
            for name, param in self.model.named_parameters():
                if 'bert_model' in name:
                    bert_params.append(param)
            if bert_params:
                param_groups.append({'params': bert_params, 'lr': self.adaptation_lr})
        else:

            for name, param in self.model.named_parameters():
                if 'bert_model' in name:
                    param.requires_grad = False

        gcn_classifier_params = []
        for name, param in self.model.named_parameters():
            if 'bert_model' not in name and 'threshold_predictor' not in name:
                gcn_classifier_params.append(param)
        if gcn_classifier_params:
            param_groups.append({'params': gcn_classifier_params, 'lr': self.adaptation_lr})

        threshold_params = []
        for name, param in self.model.named_parameters():
            if 'threshold_predictor' in name:
                threshold_params.append(param)
        if threshold_params:
            param_groups.append({'params': threshold_params, 'lr': self.threshold_predictor_lr})

        optimizer = torch.optim.Adam(param_groups)
        return optimizer

    def _reset_model(self):
        """重置模型到原始状态"""
        self.model.load_state_dict(copy.deepcopy(self.original_state_dict))

        if not self.update_bert:
            for name, param in self.model.named_parameters():
                if 'bert_model' in name:
                    param.requires_grad = True

    def _evaluate_before_adaptation(self, test_loader):
        """
        在微调前评估模型的证据句预测性能

        Args:
            test_loader: 测试数据加载器
        """
        self.model.eval()

        all_evidence_predictions = []
        all_evidence_labels = []
        all_evidence_masks = []

        all_predictions = []
        all_labels = []

        all_dynamic_thresholds = []

        print(f"开始评估原始模型（共 {len(test_loader)} 个样本）...")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                batch_data = batch_data.to(self.device)

                output, evidence_pred, _ = self.model(
                    batch_data.input_ids,
                    batch_data.attention_mask,
                    batch_data.personality_mask,
                    batch_data.adjacency_matrix,
                    batch_data.batch
                )

                dynamic_thresholds = self.model._compute_dynamic_thresholds(output)
                all_dynamic_thresholds.append(dynamic_thresholds.detach())

                all_predictions.append(output.detach())
                all_labels.append(batch_data.labels.view_as(output).detach())

                if evidence_pred is not None and isinstance(evidence_pred, dict):
                    dim_order = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
                    evidence_tensors = []
                    for dim_name in dim_order:
                        if dim_name in evidence_pred:
                            evidence_tensors.append(evidence_pred[dim_name])

                    if evidence_tensors:
                        evidence_tensor = torch.stack(evidence_tensors, dim=1)
                        all_evidence_predictions.append(evidence_tensor.detach())

                        if hasattr(batch_data, 'utter_ids') and batch_data.utter_ids:
                            utter_ids = batch_data.utter_ids
                            num_utterance_nodes = evidence_tensor.shape[0]
                            evidence_labels = convert_utter_ids_to_evidence_labels(
                                utter_ids, num_utterance_nodes, self.device
                            )
                            all_evidence_labels.append(evidence_labels)

        evidence_metrics = self._calculate_evidence_accuracy(
            all_evidence_predictions, all_evidence_labels, all_evidence_masks
        )

        all_pred_tensor = torch.cat(all_predictions, dim=0)
        all_label_tensor = torch.cat(all_labels, dim=0)
        evaluator = ModelEvaluator(num_classes=15, device=self.device)
        personality_metrics = evaluator.evaluate(all_pred_tensor, all_label_tensor)

        print("\n" + "=" * 80)
        print("【微调前评估结果】")
        print("=" * 80)

        if evidence_metrics and evidence_metrics.get('total_sentences', 0) > 0:
            print("\n证据句预测指标:")
            print(f"  总体 Accuracy: {evidence_metrics['accuracy']:.4f}")
            print(f"  平均 F1 (weighted): {evidence_metrics.get('avg_f1', 0.0):.4f}")
            print(f"  评估句子数: {evidence_metrics['total_sentences']}")

            per_dim = evidence_metrics.get('per_dimension', {})
            if per_dim:
                trait_names_en = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
                trait_names_cn = ['开放性', '尽责性', '外向性', '宜人性', '神经质性']
                print("\n  各维度证据句指标:")
                for en_name, cn_name in zip(trait_names_en, trait_names_cn):
                    if en_name in per_dim:
                        dim_metrics = per_dim[en_name]
                        print(f"    {cn_name}: Acc={dim_metrics['accuracy']:.4f}, "
                              f"Weighted-F1={dim_metrics.get('weighted_f1', 0.0):.4f}")
        else:
            print("\n证据句预测指标:")
            print("  无证据句标签数据，跳过评估")

        if all_dynamic_thresholds:
            all_dynamic_thresholds = torch.cat(all_dynamic_thresholds, dim=0)
            print(f"\n动态阈值统计:")
            print(f"  均值: {all_dynamic_thresholds.mean():.3f}")
            print(f"  最小值: {all_dynamic_thresholds.min():.3f}")
            print(f"  最大值: {all_dynamic_thresholds.max():.3f}")
            print(f"  标准差: {all_dynamic_thresholds.std():.3f}")

            dim_names = ['开放性', '尽责性', '外向性', '宜人性', '神经质性']
            for i, dim_name in enumerate(dim_names):
                print(f"  {dim_name}: {all_dynamic_thresholds[:, i].mean():.3f} ± {all_dynamic_thresholds[:, i].std():.3f}")

        return evidence_metrics, personality_metrics

    def adapt_and_predict(self, batch_data, pseudo_label: torch.Tensor,
                         pseudo_label_dict: Dict[str, str] = None) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        对单个样本进行适应并预测

        Args:
            batch_data: PyG批次数据
            pseudo_label: 大模型伪标签（15维向量）
            pseudo_label_dict: 大模型伪标签字典（用于判断"无法判断"维度）

        Returns:
            output: 模型预测输出
            evidence_predictions: 证据句预测
            adapt_info: 适应过程信息
        """
        self.model.train()
        optimizer = self._get_adaptation_optimizer()

        adapt_info = {
            'losses': [],
            'initial_pred': None,
            'final_pred': None,
            'initial_thresholds': None,
            'final_thresholds': None
        }

        pseudo_label = pseudo_label.to(self.device)

        for step in range(self.adaptation_steps):
            optimizer.zero_grad()

            output, _, _ = self.model(
                batch_data.input_ids,
                batch_data.attention_mask,
                batch_data.personality_mask,
                batch_data.adjacency_matrix,
                batch_data.batch
            )

            if step == 0:
                adapt_info['initial_pred'] = output.detach().clone()

                with torch.no_grad():
                    adapt_info['initial_thresholds'] = self.model._compute_dynamic_thresholds(output).detach().clone()

            loss = self.criterion(output, pseudo_label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            adapt_info['losses'].append(loss.item())

        self.model.eval()
        with torch.no_grad():
            output, evidence_predictions, _ = self.model(
                batch_data.input_ids,
                batch_data.attention_mask,
                batch_data.personality_mask,
                batch_data.adjacency_matrix,
                batch_data.batch
            )
            adapt_info['final_pred'] = output.detach().clone()

            adapt_info['final_thresholds'] = self.model._compute_dynamic_thresholds(output).detach().clone()

        if pseudo_label_dict is not None and evidence_predictions is not None:
            undetermined_dims = self.pseudo_label_loader.get_undetermined_dimensions(pseudo_label_dict)
            if undetermined_dims:
                evidence_predictions = zero_undetermined_evidence(evidence_predictions, undetermined_dims)

        return output, evidence_predictions, adapt_info

    def run_adaptation(self,
                       reset_per_sample: bool = True,
                       focus_trait: Optional[int] = None,
                       output_path: Optional[str] = None):
        """
        运行测试时适应

        Args:
            reset_per_sample: 每个样本后是否重置模型
            focus_trait: 是否只关注某个特定维度（0-4）
            output_path: 输出结果路径
        """
        print("=" * 80)
        print("开始测试时适应（动态阈值模型）")
        print("=" * 80)
        print(f"配置:")
        print(f"  学习率: {self.adaptation_lr}")
        print(f"  阈值预测器学习率: {self.threshold_predictor_lr}")
        print(f"  适应步数: {self.adaptation_steps}")
        print(f"  更新BERT: {self.update_bert}")
        print(f"  每样本重置: {reset_per_sample}")
        if focus_trait is not None:
            trait_names = ['开放性', '尽责性', '外向性', '宜人性', '神经质性']
            print(f"  关注维度: {trait_names[focus_trait]}")
        print()

        test_indices = [i for i, item in enumerate(self.dataset.data)
                        if item.get('split') == 'test']
        print(f"测试集样本数: {len(test_indices)}")

        test_subset = Subset(self.dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

        self._evaluate_before_adaptation(test_loader)
        print()

        evaluator = ModelEvaluator(num_classes=15, device=self.device)

        all_predictions = []
        all_labels = []
        all_pseudo_labels = []
        all_dialogue_ids = []
        all_character_names = []
        adaptation_stats = []

        all_evidence_predictions = []
        all_evidence_labels = []
        all_evidence_masks = []

        threshold_changes = []

        samples_with_pseudo_label = 0
        samples_without_pseudo_label = 0

        for batch_idx, batch_data in enumerate(test_loader):
            batch_data = batch_data.to(self.device)

            dialogue_id = batch_data.dialogue_id[0] if hasattr(batch_data, 'dialogue_id') else str(batch_idx)
            character_name = batch_data.target_character[0] if hasattr(batch_data, 'target_character') else "unknown"

            pseudo_label_dict = self.pseudo_label_loader.get_pseudo_label(character_name, dialogue_id)

            if pseudo_label_dict is not None:

                pseudo_label = self.pseudo_label_loader.convert_to_label_vector(pseudo_label_dict)
                output, evidence_pred, adapt_info = self.adapt_and_predict(
                    batch_data, pseudo_label, pseudo_label_dict
                )
                samples_with_pseudo_label += 1

                all_pseudo_labels.append(pseudo_label)

                if 'initial_thresholds' in adapt_info and 'final_thresholds' in adapt_info:
                    initial_th = adapt_info['initial_thresholds'].squeeze().cpu().numpy()
                    final_th = adapt_info['final_thresholds'].squeeze().cpu().numpy()

                    threshold_change = np.abs(final_th - initial_th)
                    threshold_changes.append(threshold_change)

                adaptation_stats.append({
                    'dialogue_id': dialogue_id,
                    'character_name': character_name,
                    'losses': adapt_info['losses'],
                    'adapted': True
                })

                if reset_per_sample:
                    self._reset_model()
            else:

                self.model.eval()
                with torch.no_grad():
                    output, evidence_pred, _ = self.model(
                        batch_data.input_ids,
                        batch_data.attention_mask,
                        batch_data.personality_mask,
                        batch_data.adjacency_matrix,
                        batch_data.batch
                    )
                samples_without_pseudo_label += 1

                adaptation_stats.append({
                    'dialogue_id': dialogue_id,
                    'character_name': character_name,
                    'adapted': False
                })

            all_predictions.append(output.detach())
            all_labels.append(batch_data.labels.view_as(output).detach())
            all_dialogue_ids.append(dialogue_id)
            all_character_names.append(character_name)

            if evidence_pred is not None and isinstance(evidence_pred, dict):
                dim_order = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
                evidence_tensors = []
                for dim_name in dim_order:
                    if dim_name in evidence_pred:
                        evidence_tensors.append(evidence_pred[dim_name])

                if evidence_tensors:
                    evidence_tensor = torch.stack(evidence_tensors, dim=1)
                    all_evidence_predictions.append(evidence_tensor.detach())

                    if hasattr(batch_data, 'utter_ids') and batch_data.utter_ids:
                        utter_ids = batch_data.utter_ids
                        num_utterance_nodes = evidence_tensor.shape[0]
                        evidence_labels = convert_utter_ids_to_evidence_labels(
                            utter_ids, num_utterance_nodes, self.device
                        )
                        all_evidence_labels.append(evidence_labels.detach())

                        utterance_mask = torch.ones(num_utterance_nodes, dtype=torch.bool, device=self.device)
                        all_evidence_masks.append(utterance_mask)

            if (batch_idx + 1) % 50 == 0:
                print(f"处理进度: {batch_idx + 1}/{len(test_loader)}")

        print(f"\n适应统计:")
        print(f"  有伪标签样本: {samples_with_pseudo_label}")
        print(f"  无伪标签样本: {samples_without_pseudo_label}")

        all_pred_tensor = torch.cat(all_predictions, dim=0)
        all_label_tensor = torch.cat(all_labels, dim=0)

        evidence_metrics = self._calculate_evidence_accuracy(
            all_evidence_predictions, all_evidence_labels, all_evidence_masks
        )

        metrics = evaluator.evaluate(all_pred_tensor, all_label_tensor, focus_trait=focus_trait)
        metrics['evidence_metrics'] = evidence_metrics

        if threshold_changes:
            threshold_changes_array = np.array(threshold_changes)
            trait_names = ['开放性', '尽责性', '外向性', '宜人性', '神经质性']
            print(f"\n动态阈值变化统计:")
            print(f"  平均变化量: {threshold_changes_array.mean():.4f}")
            print(f"  最大变化量: {threshold_changes_array.max():.4f}")
            print(f"  最小变化量: {threshold_changes_array.min():.4f}")

            for i, dim_name in enumerate(trait_names):
                print(f"  {dim_name}平均变化: {threshold_changes_array[:, i].mean():.4f}")

        self._print_results(metrics, focus_trait)

        if output_path:
            self._save_results(output_path, metrics, adaptation_stats,
                              all_dialogue_ids, all_character_names,
                              all_pred_tensor, all_label_tensor,
                              all_evidence_predictions, all_evidence_labels,
                              threshold_changes)

        return metrics, adaptation_stats

    def _calculate_evidence_accuracy(self,
                                      evidence_predictions: List[torch.Tensor],
                                      evidence_labels: List[torch.Tensor],
                                      evidence_masks: List[torch.Tensor]) -> Dict:
        """
        计算证据句预测的准确率

        Args:
            evidence_predictions: 证据句预测列表
            evidence_labels: 证据句真实标签列表
            evidence_masks: 有效句子mask列表

        Returns:
            证据句评估指标字典
        """
        if not evidence_predictions or not evidence_labels:
            return {
                'accuracy': 0.0,
                'avg_f1': 0.0,
                'total_sentences': 0,
                'correct_predictions': 0,
                'per_dimension': {}
            }

        dim_names = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

        sample_f1_scores = []

        total_correct = 0
        total_count = 0

        per_dim_stats = {dim: {'correct': 0, 'total': 0} for dim in dim_names}

        per_dim_predictions = {dim: [] for dim in dim_names}
        per_dim_labels = {dim: [] for dim in dim_names}

        for pred, label, mask in zip(evidence_predictions, evidence_labels, evidence_masks):

            if pred.dim() == 3:
                pred = pred.squeeze(0)
            if label.dim() == 3:
                label = label.squeeze(0)
            if mask.dim() == 2:
                mask = mask.squeeze(0)

            valid_indices = mask.bool()
            if valid_indices.sum() == 0:
                continue

            pred_binary = pred.float()
            label_binary = label.float()

            sample_dim_f1_scores = []

            for dim_idx, dim_name in enumerate(dim_names):
                pred_dim = pred_binary[valid_indices, dim_idx]
                label_dim = label_binary[valid_indices, dim_idx]

                correct = (pred_dim == label_dim).sum().item()
                count = valid_indices.sum().item()

                per_dim_stats[dim_name]['correct'] += correct
                per_dim_stats[dim_name]['total'] += count

                total_correct += correct
                total_count += count

                if len(pred_dim) > 0 and len(label_dim) > 0:
                    per_dim_predictions[dim_name].extend(pred_dim.cpu().numpy().tolist())
                    per_dim_labels[dim_name].extend(label_dim.cpu().numpy().tolist())

                if len(pred_dim) > 0 and len(label_dim) > 0:
                    dim_f1 = f1_score(
                        label_dim.cpu().numpy(),
                        pred_dim.cpu().numpy(),
                        average='weighted',
                        zero_division=0
                    )
                    sample_dim_f1_scores.append(dim_f1)

            if sample_dim_f1_scores:
                sample_avg_f1 = np.mean(sample_dim_f1_scores)
                sample_f1_scores.append(sample_avg_f1)

        avg_f1 = np.mean(sample_f1_scores) if sample_f1_scores else 0.0

        overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

        per_dimension = {}
        for dim_name in dim_names:
            stats = per_dim_stats[dim_name]
            dim_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

            dim_weighted_f1 = 0.0
            if len(per_dim_predictions[dim_name]) > 0 and len(per_dim_labels[dim_name]) > 0:
                dim_weighted_f1 = f1_score(
                    per_dim_labels[dim_name],
                    per_dim_predictions[dim_name],
                    average='weighted',
                    zero_division=0
                )

            per_dimension[dim_name] = {
                'accuracy': dim_acc,
                'weighted_f1': dim_weighted_f1,
                'total': stats['total'],
                'correct': stats['correct']
            }

        return {
            'accuracy': overall_accuracy,
            'avg_f1': avg_f1,
            'total_sentences': total_count // 5,
            'correct_predictions': total_correct,
            'per_dimension': per_dimension
        }

    def _print_results(self, metrics: Dict, focus_trait: Optional[int] = None):
        """打印评估结果"""
        print("\n" + "=" * 80)
        print("测试时适应结果")
        print("=" * 80)

        if focus_trait is not None:
            trait_names = ['开放性', '尽责性', '外向性', '宜人性', '神经质性']
            print(f"关注维度: {trait_names[focus_trait]}")
            print(f"  Trait Accuracy: {metrics['overall_metrics']['trait_accuracy']:.4f}")
            print(f"  Trait F1: {metrics['overall_metrics']['trait_f1']:.4f}")
        else:
            print("整体指标")
        evidence_metrics = metrics.get('evidence_metrics', {})
        if evidence_metrics and evidence_metrics.get('total_sentences', 0) > 0:
            print("\n【证据句预测指标】")
            print(f"  平均 F1 (weighted): {evidence_metrics.get('avg_f1', 0.0):.4f}")
            print(f"  评估句子数: {evidence_metrics['total_sentences']}")

            per_dim = evidence_metrics.get('per_dimension', {})
            if per_dim:
                trait_names_en = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
                trait_names_cn = ['开放性', '尽责性', '外向性', '宜人性', '神经质性']
                print("\n  各维度证据句指标:")
                for en_name, cn_name in zip(trait_names_en, trait_names_cn):
                    if en_name in per_dim:
                        dim_metrics = per_dim[en_name]
                        print(f"    {cn_name}: Acc={dim_metrics['accuracy']:.4f}, "
                              f"Weighted-F1={dim_metrics.get('weighted_f1', 0.0):.4f}")
        else:
            print("\n【证据句预测指标】")
            print("  无证据句标签数据，跳过评估")

        print("=" * 80)

    def _save_results(self, output_path: str, metrics: Dict, adaptation_stats: List,
                      dialogue_ids: List, character_names: List,
                      predictions: torch.Tensor, labels: torch.Tensor,
                      evidence_predictions: List[torch.Tensor] = None,
                      evidence_labels: List[torch.Tensor] = None,
                      threshold_changes: List = None):
        """保存结果到JSON文件"""

        pred_classes = torch.argmax(predictions, dim=2).cpu().tolist()
        label_reshaped = labels.view(-1, 5, 3)
        label_classes = torch.argmax(label_reshaped, dim=2).cpu().tolist()

        level_names = ['高', '低', '无法判断']
        dim_names = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

        detailed_predictions = []
        for i in range(len(dialogue_ids)):
            sample_result = {
                'dialogue_id': dialogue_ids[i],
                'character_name': character_names[i],
                'adapted': adaptation_stats[i]['adapted'],
                'dimensions': {}
            }

            for j, dim_name in enumerate(dim_names):
                sample_result['dimensions'][dim_name] = {
                    'predicted': level_names[pred_classes[i][j]],
                    'ground_truth': level_names[label_classes[i][j]],
                    'correct': pred_classes[i][j] == label_classes[i][j]
                }

            if evidence_predictions and i < len(evidence_predictions):
                ev_pred = evidence_predictions[i]
                if ev_pred.dim() == 3:
                    ev_pred = ev_pred.squeeze(0)
                ev_pred_binary = ev_pred.int().cpu().tolist()

                ev_label = None
                if evidence_labels and i < len(evidence_labels):
                    ev_label = evidence_labels[i]
                    if ev_label.dim() == 3:
                        ev_label = ev_label.squeeze(0)
                    ev_label = ev_label.cpu().tolist()

                evidence_info = []
                for sent_idx in range(len(ev_pred_binary)):
                    sent_evidence = {}
                    for dim_idx, dim_name in enumerate(dim_names):
                        pred_val = int(ev_pred_binary[sent_idx][dim_idx])
                        label_val = int(ev_label[sent_idx][dim_idx]) if ev_label else None
                        sent_evidence[dim_name] = {
                            'predicted': pred_val,
                            'ground_truth': label_val,
                            'correct': pred_val == label_val if label_val is not None else None
                        }
                    evidence_info.append(sent_evidence)

                sample_result['evidence_sentences'] = evidence_info

            detailed_predictions.append(sample_result)

        output_data = {
            'config': {
                'adaptation_lr': self.adaptation_lr,
                'threshold_predictor_lr': self.threshold_predictor_lr,
                'adaptation_steps': self.adaptation_steps,
                'update_bert': self.update_bert,
                'model_path': self.model_path,
                'llm_results_path': self.llm_results_path
            },
            'metrics': {
                'overall': metrics['overall_metrics'],
                'per_trait': metrics['trait_metrics'],
                'evidence': metrics.get('evidence_metrics', {})
            },
            'adaptation_stats': adaptation_stats,
            'detailed_predictions': detailed_predictions
        }

        if threshold_changes:
            threshold_changes_array = np.array(threshold_changes)
            trait_names = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
            output_data['threshold_statistics'] = {
                'mean_change': float(threshold_changes_array.mean()),
                'max_change': float(threshold_changes_array.max()),
                'min_change': float(threshold_changes_array.min()),
                'std_change': float(threshold_changes_array.std()),
                'per_dimension': {
                    trait_names[i]: {
                        'mean_change': float(threshold_changes_array[:, i].mean()),
                        'std_change': float(threshold_changes_array[:, i].std())
                    }
                    for i in range(5)
                }
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='测试时适应')
    parser.add_argument('--model_path', type=str, default='best_bigfive.pth', help='预训练模型路径')
    parser.add_argument('--llm_results', type=str,
                        default='',
                        help='大模型判断结果JSON文件路径')
    parser.add_argument('--data_path', type=str, default='data/multilabel_dataset.json',
                        help='数据集路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--lr', type=float, default=1e-5, help='适应学习率')
    parser.add_argument('--threshold_lr', type=float, default=None,
                        help='阈值预测器学习率（如果不指定，使用--lr）')
    parser.add_argument('--steps', type=int, default=1, help='每个样本的适应步数')
    parser.add_argument('--update_bert', action='store_true', help='是否更新BERT参数')
    parser.add_argument('--no_reset', action='store_true', help='不重置模型（持续适应）')
    parser.add_argument('--focus_trait', type=int, default=None, choices=[0, 1, 2, 3, 4],
                        help='只关注某个特定维度 (0:开放性, 1:尽责性, 2:外向性, 3:宜人性, 4:神经质性)')
    parser.add_argument('--output', type=str, default=None, help='输出结果路径')

    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f'tta_dynamic_threshold_results_{timestamp}.json'

    tta = TestTimeAdaptationDynamicThreshold(
        model_path=args.model_path,
        llm_results_path=args.llm_results,
        data_path=args.data_path,
        device=args.device,
        adaptation_lr=args.lr,
        adaptation_steps=args.steps,
        update_bert=args.update_bert,
        threshold_predictor_lr=args.threshold_lr
    )

    tta.run_adaptation(
        reset_per_sample=not args.no_reset,
        focus_trait=args.focus_trait,
        output_path=args.output
    )

if __name__ == '__main__':
    main()
