"""
评估模块 - 模型评估器
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

try:
    from data.BigFive_definition import definition
except ImportError:
    definition = {
        '开放性': '开放性是指对新经验、新想法的接受程度',
        '尽责性': '尽责性是指做事认真负责、有条理的程度',
        '外向性': '外向性是指与他人互动、寻求刺激的倾向',
        '宜人性': '宜人性是指与他人和谐相处、关心他人的程度',
        '神经质性': '神经质性是指情绪不稳定、容易焦虑的倾向'
    }


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, num_classes: int = 15, device: str = 'cpu', output_json_path: str = None):
        """
        初始化评估器

        Args:
            num_classes: 类别数量
            device: 设备
            output_json_path: 输出JSON路径
        """
        self.num_classes = num_classes
        self.device = device
        self.output_json_path = output_json_path

    def _vector_to_labels(self, label_vector):
        """
        将15维标签向量转换为可读的标签字典

        Args:
            label_vector: 15维标签向量

        Returns:
            labels: 标签字典
        """
        labels = {}
        big_five_traits = [
            ("开放性", ["高", "低", "无法判断"]),
            ("尽责性", ["高", "低", "无法判断"]),
            ("外向性", ["高", "低", "无法判断"]),
            ("宜人性", ["高", "低", "无法判断"]),
            ("神经质性", ["高", "低", "无法判断"])
        ]

        for i, (trait, levels) in enumerate(big_five_traits):
            for j, level in enumerate(levels):
                if label_vector[i * 3 + j] == 1:
                    labels[trait] = level
                    break
            else:
                labels[trait] = "无法判断"
        return labels

    def evaluate(self, predictions: torch.Tensor, true_labels: torch.Tensor,
                 dialogue_ids: list = None, original_texts: list = None, character_name: Optional[list] = None,
                 focus_trait: Optional[int] = None, evidence_f1_scores: Optional[list] = None,
                 evidence_predictions: Optional[list] = None, true_evidence_labels: Optional[list] = None):
        """
        评估模型

        Args:
            predictions: 预测结果
            true_labels: 真实标签
            dialogue_ids: 对话ID列表
            original_texts: 原始文本列表
            character_name: 角色名称列表
            focus_trait: 关注的人格维度
            evidence_f1_scores: 证据句F1分数列表
            evidence_predictions: 证据句预测列表
            true_evidence_labels: 真实证据句标签列表

        Returns:
            metrics: 评估指标字典
        """
        predictions_np = predictions.cpu().numpy()
        true_labels_np = true_labels.cpu().numpy()

        if predictions_np.shape != true_labels_np.shape:
            if true_labels_np.ndim == 1 and predictions_np.ndim == 2:
                expected_size = predictions_np.shape[0] * predictions_np.shape[1]
                if true_labels_np.size == expected_size:
                    true_labels_np = true_labels_np.reshape(predictions_np.shape)
                else:
                    print(f"无法处理形状不匹配: 预测 {predictions_np.shape}, 标签 {true_labels_np.shape}")
            else:
                raise ValueError("预测和标签形状不匹配且无法自动处理")

        pred_reshaped = predictions_np.reshape(-1, 5, 3)
        true_reshaped = true_labels_np.reshape(-1, 5, 3)

        trait_names = ['开放性', '尽责性', '外向性', '宜人性', '神经质性']

        if focus_trait is not None:
            pred_class = np.argmax(pred_reshaped[:, focus_trait, :], axis=1)
            true_class = np.argmax(true_reshaped[:, focus_trait, :], axis=1)
            accuracy = accuracy_score(true_class, pred_class)
            f1 = f1_score(true_class, pred_class, average='weighted', zero_division=0)

            metrics = {
                'overall_metrics': {
                    'trait_accuracy': accuracy,
                    'trait_f1': f1
                },
                'trait_metrics': {
                    f'trait_{focus_trait}': {
                        'trait_name': trait_names[focus_trait],
                        'accuracy': accuracy,
                        'f1_score': f1
                    }
                }
            }

            if evidence_f1_scores is not None:
                avg_evidence_f1 = float(np.mean(evidence_f1_scores)) if evidence_f1_scores else 0.0
                metrics['overall_metrics']['avg_evidence_f1'] = avg_evidence_f1
        else:
            pred_classes = []
            true_classes = []
            for i in range(5):
                pred_class = np.argmax(pred_reshaped[:, i, :], axis=1)
                true_class = np.argmax(true_reshaped[:, i, :], axis=1)
                pred_classes.append(pred_class)
                true_classes.append(true_class)

            sample_correct_counts = np.zeros(len(predictions_np))
            for i in range(5):
                sample_correct_counts += (pred_classes[i] == true_classes[i]).astype(int)

            sample_accuracy = sample_correct_counts / 5.0
            overall_sample_accuracy = float(np.mean(sample_accuracy))
            fully_correct_accuracy = float(np.mean(sample_correct_counts == 5))

            trait_metrics = {}
            for i in range(5):
                accuracy = accuracy_score(true_classes[i], pred_classes[i])
                f1 = f1_score(true_classes[i], pred_classes[i], average='weighted', zero_division=0)
                trait_metrics[f'trait_{i}'] = {
                    'trait_name': trait_names[i],
                    'accuracy': accuracy,
                    'f1_score': f1
                }

            avg_trait_accuracy = np.mean([m['accuracy'] for m in trait_metrics.values()])
            avg_trait_f1 = np.mean([m['f1_score'] for m in trait_metrics.values()])

            metrics = {
                'overall_metrics': {
                    'sample_accuracy': overall_sample_accuracy,
                    'fully_correct_accuracy': fully_correct_accuracy,
                    'avg_trait_accuracy': avg_trait_accuracy,
                    'avg_trait_f1': avg_trait_f1
                },
                'trait_metrics': trait_metrics
            }

            if evidence_f1_scores is not None:
                avg_evidence_f1 = float(np.mean(evidence_f1_scores)) if evidence_f1_scores else 0.0
                metrics['overall_metrics']['avg_evidence_f1'] = avg_evidence_f1

        if self.output_json_path and dialogue_ids and original_texts and character_name:
            output_data = []
            for i in range(len(dialogue_ids)):
                if focus_trait is not None:
                    pred_class = int(np.argmax(pred_reshaped[i, focus_trait, :]))
                    true_class = int(np.argmax(true_reshaped[i, focus_trait, :]))
                    item_data = {
                        "dialogue_id": dialogue_ids[i],
                        "original_text": original_texts[i],
                        "character_name": character_name[i],
                        "trait_index": int(focus_trait),
                        "trait_name": trait_names[focus_trait],
                        "predicted_class": pred_class,
                        "true_class": true_class
                    }

                    if evidence_predictions and i < len(evidence_predictions):
                        item_data["evidence_predictions"] = evidence_predictions[i]

                    if true_evidence_labels and i < len(true_evidence_labels):
                        item_data["true_evidence_labels"] = true_evidence_labels[i]
                    if evidence_f1_scores and i < len(evidence_f1_scores):
                        item_data["evidence_f1_score"] = float(evidence_f1_scores[i])
                    output_data.append(item_data)
                else:
                    pred_class_vector = []
                    true_class_vector = []
                    for j in range(5):
                        pc = int(np.argmax(pred_reshaped[i, j, :]))
                        tc = int(np.argmax(true_reshaped[i, j, :]))
                        pred_class_vector.append(pc)
                        true_class_vector.append(tc)
                    pred_one_hot = np.zeros(15)
                    true_one_hot = np.zeros(15)
                    for j in range(5):
                        pred_one_hot[j*3 + pred_class_vector[j]] = 1
                        true_one_hot[j*3 + true_class_vector[j]] = 1
                    pred_labels = self._vector_to_labels(pred_one_hot.tolist())
                    true_labels_readable = self._vector_to_labels(true_one_hot.tolist())
                    item_data = {
                        "dialogue_id": dialogue_ids[i],
                        "original_text": original_texts[i],
                        "character_name": character_name[i],
                        "predicted_class_vector": pred_class_vector,
                        "predicted_labels": pred_labels,
                        "true_class_vector": true_class_vector,
                        "true_labels": true_labels_readable
                    }

                    if evidence_predictions and i < len(evidence_predictions):
                        item_data["evidence_predictions"] = evidence_predictions[i]

                    if true_evidence_labels and i < len(true_evidence_labels):
                        item_data["true_evidence_labels"] = true_evidence_labels[i]
                    if evidence_f1_scores and i < len(evidence_f1_scores):
                        item_data["evidence_f1_score"] = float(evidence_f1_scores[i])
                    output_data.append(item_data)

            os.makedirs(os.path.dirname(self.output_json_path), exist_ok=True)
            with open(self.output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            print(f"预测结果已保存到: {self.output_json_path}")

        return metrics
