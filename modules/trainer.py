"""
训练模块 - TrainingLogger, train_model, evaluate_on_test_set
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Optional
from torch_geometric.loader import DataLoader

from .models import BigFiveGNN
from .losses import BigFiveLoss
from .utils import calculate_accuracy_detailed, calculate_evidence_f1_score

from .evaluator import ModelEvaluator


class TrainingLogger:
    """训练日志记录器，同时输出到控制台和文件"""

    def __init__(self, log_dir="logs", log_filename=None):
        """
        初始化日志记录器

        Args:
            log_dir: 日志目录
            log_filename: 日志文件名
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"training_{timestamp}.log"

        self.log_path = os.path.join(log_dir, log_filename)

        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        file_handler = logging.FileHandler(self.log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.original_print = print

        self.logger.info(f"训练日志已初始化，日志文件: {self.log_path}")
        print(f"📝 训练日志将保存到: {self.log_path}")

    def log_print(self, *args, **kwargs):
        """替代print函数，同时输出到控制台和日志文件"""
        self.original_print(*args, **kwargs)
        message = ' '.join(str(arg) for arg in args)

        if any(keyword in message.lower() for keyword in ['error', '错误', 'warning', '警告']):
            level = logging.WARNING
        elif any(keyword in message.lower() for keyword in ['epoch', 'loss', 'acc', 'f1', '准确', '损失']):
            level = logging.INFO
        else:
            level = logging.INFO

        self.logger.log(level, message)

    def log_training_info(self, model_info, data_info, training_params):
        """记录训练基本信息"""
        self.logger.info("=" * 80)
        self.logger.info("开始训练 - 大五人格图神经网络模型")
        self.logger.info("=" * 80)

        self.logger.info("模型配置:")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")

        self.logger.info("数据集信息:")
        for key, value in data_info.items():
            self.logger.info(f"  {key}: {value}")

        self.logger.info("训练参数:")
        for key, value in training_params.items():
            self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 80)

    def log_epoch_metrics(self, epoch, train_metrics, val_metrics, evidence_metrics=None):
        """记录每个epoch的详细指标"""
        self.logger.info(f"Epoch {epoch + 1} 详细指标:")

        self.logger.info("  训练集:")
        for key, value in train_metrics.items():
            self.logger.info(f"    {key}: {value:.6f}")

        self.logger.info("  验证集:")
        for key, value in val_metrics.items():
            self.logger.info(f"    {key}: {value:.6f}")

        if evidence_metrics:
            self.logger.info("  证据句:")
            for key, value in evidence_metrics.items():
                self.logger.info(f"    {key}: {value:.6f}")

    def log_model_save(self, save_path, metrics, is_best=False):
        """记录模型保存信息"""
        status = "最佳模型" if is_best else "检查点模型"
        self.logger.info(f"保存{status}: {save_path}")
        self.logger.info(f"保存时指标: {metrics}")

    def log_training_complete(self, final_metrics, total_time):
        """记录训练完成信息"""
        self.logger.info("=" * 80)
        self.logger.info("训练完成!")
        self.logger.info(f"总训练时间: {total_time:.2f} 秒")

        self.logger.info("最终指标:")
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 80)

    def replace_print(self):
        """替换全局print函数"""
        import builtins
        builtins.print = self.log_print

    def restore_print(self):
        """恢复原始print函数"""
        import builtins
        builtins.print = self.original_print


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = 50, learning_rate: float = 0.001,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                save_path: str = 'best_bigfive_yuzhi.pth',
                focus_trait: Optional[int] = None):
    """
    训练模型

    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        save_path: 保存路径
        focus_trait: 关注维度
    """
    model = model.to(device)
    if device == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = torch.amp.GradScaler('cpu')

    bert_params = list(model.bert_model.parameters())

    shared_params = []
    for gcn_block in model.gcn_blocks:
        shared_params.extend(list(gcn_block.parameters()))
    classifier_params = [list(h.parameters()) for h in model.personality_classifiers]
    flat_classifier_params = []
    for group in classifier_params:
        flat_classifier_params += group

    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': learning_rate, 'weight_decay': 0.01},
        {'params': shared_params, 'lr': 5e-4, 'weight_decay': 1e-4},
        {'params': flat_classifier_params, 'lr': 5e-4, 'weight_decay': 1e-4}
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
    )

    base_bert_lr = learning_rate
    base_nonbert_lr = 5e-4
    total_steps = max(1, len(train_loader) * num_epochs)
    warmup_steps = max(1, int(total_steps * 0.1))

    for i, g in enumerate(optimizer.param_groups):
        if i == 0:
            g['lr'] = base_bert_lr
        else:
            g['lr'] = 0.0

    criterion = BigFiveLoss(use_focal_loss=False, pos_weight=5.0, evidence_weight=1.0)
    print(f"使用正样本权重: {criterion.pos_weight} (对证据句加权{criterion.pos_weight}倍)")
    print(f"证据句损失权重: {criterion.evidence_weight} (优先优化证据句预测)")

    evaluator = ModelEvaluator(num_classes=15, device=device)

    print("开始训练...")
    if focus_trait is not None:
        trait_name = ['开放性', '尽责性', '外向性', '宜人性', '神经质性'][focus_trait]
        print(f"单维度模式，当前维度: {trait_name}")

    best_val_f1 = 0.0
    current_step = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_train_logits = []
        all_train_labels = []
        train_evidence_metrics = []

        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            device_type = "cuda" if ("cuda" in str(device).lower()) else "cpu"

            with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                output, evidence_predictions_dict, evidence_predictions_raw = model(
                    batch_data.input_ids,
                    batch_data.attention_mask,
                    batch_data.personality_mask,
                    batch_data.adjacency_matrix,
                    batch_data.batch
                )

                if hasattr(batch_data, 'utter_ids'):
                    true_evidence = batch_data.utter_ids[0] if isinstance(batch_data.utter_ids, (list, tuple)) else batch_data.utter_ids

                    loss, loss_metrics = criterion(
                        output,
                        batch_data.labels.view_as(output),
                        evidence_predictions_raw=evidence_predictions_raw,
                        true_evidence=true_evidence,
                        focus_trait=focus_trait,
                        return_metrics=True
                    )

                    train_evidence_metrics.append(loss_metrics)
                else:
                    print("[警告] 当前批次缺少证据句标签，回退到原有损失计算方式。")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            all_train_logits.append(output.detach())
            all_train_labels.append(batch_data.labels.view_as(output).detach())

            current_step += 1
            if current_step < warmup_steps:
                scale = float(current_step) / float(warmup_steps)
                for i, g in enumerate(optimizer.param_groups):
                    if i == 0:
                        g['lr'] = base_bert_lr
                    else:
                        g['lr'] = base_nonbert_lr * scale
            else:
                for i, g in enumerate(optimizer.param_groups):
                    if i == 0:
                        g['lr'] = base_bert_lr
                    else:
                        g['lr'] = base_nonbert_lr

        avg_train_loss = train_loss / len(train_loader)
        all_train_pred_tensor = torch.cat(all_train_logits, dim=0)
        all_train_label_tensor = torch.cat(all_train_labels, dim=0)

        if focus_trait is not None:
            train_acc = calculate_accuracy_detailed(all_train_pred_tensor, all_train_label_tensor, focus_trait)
        else:
            train_metrics = calculate_accuracy_detailed(all_train_pred_tensor, all_train_label_tensor)
            train_acc = train_metrics['overall_accuracy']

        model.eval()
        val_loss = 0.0
        all_logits = []
        all_labels = []

        val_evidence_f1_scores = []
        val_evidence_metrics = []
        val_evidence_predictions_list = []
        val_true_evidence_list = []
        val_evidence_detailed_metrics = []

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
                with torch.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                    output, evidence_predictions_dict, evidence_predictions_raw = model(
                        batch_data.input_ids,
                        batch_data.attention_mask,
                        batch_data.personality_mask,
                        batch_data.adjacency_matrix,
                        batch_data.batch
                    )

                    if hasattr(batch_data, 'utter_ids'):
                        true_evidence = batch_data.utter_ids[0] if isinstance(batch_data.utter_ids, (list, tuple)) else batch_data.utter_ids

                        loss, loss_metrics = criterion(
                            output,
                            batch_data.labels.view_as(output),
                            evidence_predictions_raw=evidence_predictions_raw,
                            true_evidence=true_evidence,
                            focus_trait=focus_trait,
                            return_metrics=True
                        )

                        val_evidence_metrics.append(loss_metrics)
                        val_evidence_predictions_list.append(evidence_predictions_dict)
                        val_true_evidence_list.append(true_evidence)
                    else:
                        print("[警告] 当前验证批次缺少证据句标签或不是单个图，回退到原有损失计算方式。")

                val_loss += loss.item()
                all_logits.append(output)
                all_labels.append(batch_data.labels.view_as(output).float())

        avg_val_loss = val_loss / len(val_loader)
        all_val_pred_tensor = torch.cat(all_logits, dim=0)
        all_val_label_tensor = torch.cat(all_labels, dim=0)

        if focus_trait is not None:
            val_acc = calculate_accuracy_detailed(all_val_pred_tensor, all_val_label_tensor, focus_trait)
        else:
            val_metrics = calculate_accuracy_detailed(all_val_pred_tensor, all_val_label_tensor)
            val_acc = val_metrics['overall_accuracy']

        if val_evidence_predictions_list and val_true_evidence_list:
            for pred_evidence, true_evidence in zip(val_evidence_predictions_list, val_true_evidence_list):
                evidence_f1_result = calculate_evidence_f1_score(true_evidence, pred_evidence)
                val_evidence_f1_scores.append(evidence_f1_result['avg_f1_score'])
                val_evidence_detailed_metrics.append(evidence_f1_result)

            avg_val_evidence_f1 = sum(val_evidence_f1_scores) / len(val_evidence_f1_scores) if val_evidence_f1_scores else 0.0
        else:
            avg_val_evidence_f1 = 0.0

        if val_evidence_metrics:
            evidence_losses = [m.get('evidence_loss', 0.0) for m in val_evidence_metrics if 'evidence_loss' in m]
            avg_val_evidence_loss = sum(evidence_losses) / len(evidence_losses) if evidence_losses else 0.0
        else:
            avg_val_evidence_loss = 0.0

        if all_logits:
            all_pred_tensor = torch.cat(all_logits, dim=0)
            all_label_tensor = torch.cat(all_labels, dim=0)
            val_metrics = evaluator.evaluate(all_pred_tensor, all_label_tensor, focus_trait=focus_trait)

            if focus_trait is not None:
                val_metrics = {
                    'trait_accuracy': val_metrics['overall_metrics']['trait_accuracy'],
                    'trait_f1': val_metrics['overall_metrics']['trait_f1']
                }
            else:
                val_metrics = {
                    'sample_accuracy': val_metrics['overall_metrics']['sample_accuracy'],
                    'fully_correct_accuracy': val_metrics['overall_metrics']['fully_correct_accuracy'],
                    'avg_trait_accuracy': val_metrics['overall_metrics']['avg_trait_accuracy'],
                    'avg_trait_f1': val_metrics['overall_metrics']['avg_trait_f1'],
                    'trait_accuracies': [val_metrics['trait_metrics'][f'trait_{i}']['accuracy'] for i in range(5)],
                    'trait_f1_scores': [val_metrics['trait_metrics'][f'trait_{i}']['f1_score'] for i in range(5)]
                }
        else:
            print("[警告] 当前验证集中没有预测结果，无法计算详细指标。")

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1:2d}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if focus_trait is not None:
            print(f"  Trait Accuracy: {val_metrics['trait_accuracy']:.4f}")
            print(f"  Trait F1: {val_metrics['trait_f1']:.4f}")
        else:
            print(f"  Sample Accuracy: {val_metrics['sample_accuracy']:.4f}")
            print(f"  Fully Correct Accuracy: {val_metrics['fully_correct_accuracy']:.4f}")
            print(f"  Average Trait Accuracy: {val_metrics['avg_trait_accuracy']:.4f}")
            print(f"  Average Trait F1: {val_metrics['avg_trait_f1']:.4f}")

        if val_evidence_f1_scores:
            print(f"  证据句平均F1分数: {avg_val_evidence_f1:.4f}")

            if val_evidence_detailed_metrics:
                personality_dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
                trait_names_cn = ['开放性', '尽责性', '外向性', '宜人性', '神经质性']

                dim_avg_metrics = {}
                for dim in personality_dims:
                    dim_weighted_f1_list = [m['dimension_metrics'][dim]['weighted_f1'] for m in val_evidence_detailed_metrics]
                    dim_acc_list = [m['dimension_metrics'][dim]['accuracy'] for m in val_evidence_detailed_metrics]

                    dim_avg_metrics[dim] = {
                        'weighted_f1': np.mean(dim_weighted_f1_list),
                        'accuracy': np.mean(dim_acc_list)
                    }

                overall_weighted_f1_list = [m['overall_weighted_f1'] for m in val_evidence_detailed_metrics]
                overall_accuracy_list = [m['overall_accuracy'] for m in val_evidence_detailed_metrics]

                overall_avg_weighted_f1 = np.mean(overall_weighted_f1_list)
                overall_avg_accuracy = np.mean(overall_accuracy_list)

                print(f"    总体 Weighted F1: {overall_avg_weighted_f1:.4f}, 总体 Acc: {overall_avg_accuracy:.4f}")
                print(f"    维度指标 - ", end="")
                for en_name, cn_name in zip(personality_dims, trait_names_cn):
                    metrics = dim_avg_metrics[en_name]
                    print(f"{cn_name}:{metrics['weighted_f1']:.3f}/{metrics['accuracy']:.3f} ", end="")
                print()

        if val_evidence_metrics:
            print(f"  证据句损失: {avg_val_evidence_loss:.4f}")

        trait_names = ['开放性', '尽责性', '外向性', '宜人性', '神经质性']
        for i, trait_name in enumerate(trait_names):
            print(f"  {trait_name} - Accuracy: {val_metrics['trait_accuracies'][i]:.4f}, F1: {val_metrics['trait_f1_scores'][i]:.4f}")

        if avg_val_evidence_f1 > best_val_f1:
            best_val_f1 = avg_val_evidence_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'metrics': val_metrics,
                'config': {
                    'input_dim': model.input_dim,
                    'hidden_dim': model.hidden_dim,
                    'output_dim': model.output_dim,
                    'num_layers': model.num_layers
                }
            }, save_path)

            if focus_trait is not None:
                print(f"  保存最佳模型 (Trait Accuracy")
            else:
                print(f"  保存最佳模型 (Fully Evidence F1: {best_val_f1:.4f})")

    print("训练完成！")
    return model


def evaluate_on_test_set(model_path, test_loader, device, test_dataset, output_json_path, focus_trait: Optional[int] = None):
    """
    在测试集上评估最佳模型

    Args:
        model_path: 模型文件路径
        test_loader: 测试数据加载器
        device: 设备
        test_dataset: 测试数据集
        output_json_path: 输出JSON路径
        focus_trait: 关注的人格维度
    """
    print("\n" + "="*80)
    print("在测试集上评估最佳模型...")

    if not os.path.exists(model_path):
        print(f"错误:模型文件不存在 - {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint['config']

    model = BigFiveGNN(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        output_dim=model_config['output_dim'],
        num_layers=model_config['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    evaluator = ModelEvaluator(num_classes=15, device=device, output_json_path=output_json_path)

    all_predictions = []
    all_labels = []
    all_dialogue_ids = []
    all_original_texts = []
    all_character_names = []

    test_evidence_f1_scores = []
    test_evidence_predictions_list = []
    test_true_evidence_list = []
    test_evidence_detailed_metrics = []

    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
            with torch.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                output, evidence_predictions_dict, evidence_predictions_raw = model(
                    batch_data.input_ids,
                    batch_data.attention_mask,
                    batch_data.personality_mask,
                    batch_data.adjacency_matrix,
                    batch_data.batch
                )

            all_predictions.append(output)
            all_labels.append(batch_data.labels.view_as(output))
            all_dialogue_ids.extend(batch_data.dialogue_id)
            all_original_texts.extend(batch_data.dialogue)
            all_character_names.extend(batch_data.target_character)

            if hasattr(batch_data, 'utter_ids'):
                true_evidence = batch_data.utter_ids[0] if isinstance(batch_data.utter_ids, (list, tuple)) else batch_data.utter_ids
                evidence_f1_result = calculate_evidence_f1_score(true_evidence, evidence_predictions_dict)

                test_evidence_f1_scores.append(evidence_f1_result['avg_f1_score'])
                test_evidence_detailed_metrics.append(evidence_f1_result)

                evidence_pred_serializable = {}
                for dim_name, pred_tensor in evidence_predictions_dict.items():
                    evidence_pred_serializable[dim_name] = pred_tensor.cpu().tolist()
                test_evidence_predictions_list.append(evidence_pred_serializable)

                personality_dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
                num_utterances = evidence_predictions_dict['openness'].size(0)

                true_evidence_serializable = {}
                for dim_name in personality_dims:
                    true_vector = [0] * num_utterances
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
                                        true_vector[idx_int] = 1
                            except (ValueError, TypeError):
                                continue

                    true_evidence_serializable[dim_name] = true_vector

                test_true_evidence_list.append(true_evidence_serializable)

    if not all_predictions:
        print("测试集为空,无法评估。")
        return

    all_pred_tensor = torch.cat(all_predictions, dim=0)
    all_label_tensor = torch.cat(all_labels, dim=0)

    test_metrics = evaluator.evaluate(
        all_pred_tensor,
        all_label_tensor,
        all_dialogue_ids,
        all_original_texts,
        all_character_names,
        focus_trait=focus_trait,
        evidence_f1_scores=test_evidence_f1_scores if test_evidence_f1_scores else None,
        evidence_predictions=test_evidence_predictions_list if test_evidence_predictions_list else None,
        true_evidence_labels=test_true_evidence_list if test_true_evidence_list else None
    )

    print("\n测试集评估结果:")
    if focus_trait is not None:
        trait_name = ['开放性', '尽责性', '外向性', '宜人性', '神经质性'][focus_trait]
        print(f" 当前维度: {trait_name}")
        print(f"  Trait Accuracy: {test_metrics['overall_metrics']['trait_accuracy']:.4f}")
        print(f"  Trait F1: {test_metrics['overall_metrics']['trait_f1']:.4f}")
    else:
        print(f"  Sample Accuracy: {test_metrics['overall_metrics']['sample_accuracy']:.4f}")
        print(f"  Fully Correct Accuracy: {test_metrics['overall_metrics']['fully_correct_accuracy']:.4f}")
        print(f"  Average Trait Accuracy: {test_metrics['overall_metrics']['avg_trait_accuracy']:.4f}")
        print(f"  Average Trait F1: {test_metrics['overall_metrics']['avg_trait_f1']:.4f}")

        trait_names = ['开放性', '尽责性', '外向性', '宜人性', '神经质性']
        print("\n  每个特质的准确率和F1分数:")
        for i, trait_name in enumerate(trait_names):
            print(f"    {trait_name} - Accuracy: {test_metrics['trait_metrics'][f'trait_{i}']['accuracy']:.4f}, F1: {test_metrics['trait_metrics'][f'trait_{i}']['f1_score']:.4f}")

    if test_evidence_f1_scores:
        avg_evidence_f1 = test_metrics['overall_metrics'].get('avg_evidence_f1', 0.0)
        print(f"\n  证据句平均F1分数: {avg_evidence_f1:.4f}")

        if test_evidence_detailed_metrics:
            personality_dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
            trait_names_cn = ['开放性', '尽责性', '外向性', '宜人性', '神经质性']

            dim_avg_metrics = {}
            for dim in personality_dims:
                dim_f1_list = [m['dimension_metrics'][dim]['f1'] for m in test_evidence_detailed_metrics]
                dim_weighted_f1_list = [m['dimension_metrics'][dim]['weighted_f1'] for m in test_evidence_detailed_metrics]
                dim_acc_list = [m['dimension_metrics'][dim]['accuracy'] for m in test_evidence_detailed_metrics]

                dim_avg_metrics[dim] = {
                    'f1': np.mean(dim_f1_list),
                    'weighted_f1': np.mean(dim_weighted_f1_list),
                    'accuracy': np.mean(dim_acc_list)
                }

            overall_weighted_f1_list = [m['overall_weighted_f1'] for m in test_evidence_detailed_metrics]
            overall_accuracy_list = [m['overall_accuracy'] for m in test_evidence_detailed_metrics]
            avg_weighted_f1_list = [m['avg_weighted_f1'] for m in test_evidence_detailed_metrics]
            avg_accuracy_list = [m['avg_accuracy'] for m in test_evidence_detailed_metrics]

            overall_avg_weighted_f1 = np.mean(overall_weighted_f1_list)
            overall_avg_accuracy = np.mean(overall_accuracy_list)
            dim_avg_weighted_f1 = np.mean(avg_weighted_f1_list)
            dim_avg_accuracy = np.mean(avg_accuracy_list)

            print(f"\n  【证据句详细指标】")
            print(f"  总体 Weighted F1: {overall_avg_weighted_f1:.4f}")
            print(f"  总体 Accuracy: {overall_avg_accuracy:.4f}")
            print(f"  维度平均 Weighted F1: {dim_avg_weighted_f1:.4f}")
            print(f"  维度平均 Accuracy: {dim_avg_accuracy:.4f}")

            print("\n  各维度证据句指标:")
            for en_name, cn_name in zip(personality_dims, trait_names_cn):
                metrics = dim_avg_metrics[en_name]
                print(f"    {cn_name}: Acc={metrics['accuracy']:.4f}, "
                      f"F1={metrics['f1']:.4f}, "
                      f"Weighted-F1={metrics['weighted_f1']:.4f}")

    print("="*80)
