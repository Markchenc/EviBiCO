"""
大五人格图神经网络模型 - 主入口
使用PyG DataLoader + BERT嵌入
"""

import os
import argparse
import statistics
import torch
from datetime import datetime
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from modules import (
    BigFiveDataset,
    BigFiveGNN,
    TrainingLogger,
    train_model,
    evaluate_on_test_set,
    _compute_scene_weights_for_subset
)


def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--focus_trait', type=str, default=None)
    args = parser.parse_args()

    def _map_trait(trait):
        if trait is None:
            return None
        names_cn = ['开放性', '尽责性', '外向性', '宜人性', '神经质性']
        names_en = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        try:
            idx = int(trait)
            if 0 <= idx < 5:
                return idx
        except Exception:
            pass
        if trait in names_cn:
            return names_cn.index(trait)
        t = str(trait).lower()
        if t in names_en:
            return names_en.index(t)
        return None

    focus_trait = _map_trait(args.focus_trait)

    config = {
        'data_path': "data/multilabel_dataset.json",
        'model_path': "best_bigfive.pth",
        'use_bert': True,
        'batch_size': 1,
        'epochs': 50,
        'learning_rate': 2e-5,
        'hidden_dim': 256,
        'val_size': 0.15,
        'test_size': 0.15,
        'random_seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_weighted_scene_sampler': False,
        'scene_weight_alpha': 0.5,
        'scene_weight_clip_min': 1.0,
        'scene_weight_clip_max': 3.0,
        'samples_per_epoch_factor': 1.0,
        'focus_trait': focus_trait
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "enhanced"
    log_filename = f"training_{model_name}_{timestamp}.log"

    if config['focus_trait'] is not None:
        trait_name = ['开放性', '尽责性', '外向性', '宜人性', '神经质性'][config['focus_trait']]
        log_filename = f"training_{model_name}_{trait_name}_{timestamp}.log"

    training_logger = TrainingLogger(log_dir="logs", log_filename=log_filename)
    training_logger.replace_print()

    model_info = {
        "模型类型": "大五人格图神经网络模型",
        "BERT嵌入": "启用" if config['use_bert'] else "禁用",
        "增强分类器": "启用",
        "证据句损失权重": "0.8 (优先优化证据句预测)"
    }

    data_info = {
        "数据路径": config['data_path'],
        "验证集比例": config['val_size'],
        "测试集比例": config['test_size'],
        "随机种子": config['random_seed']
    }

    training_params = {
        "批次大小": config['batch_size'],
        "训练轮数": config['epochs'],
        "学习率": config['learning_rate'],
        "隐藏维度": config['hidden_dim'],
        "设备": config['device']
    }

    if config['focus_trait'] is not None:
        trait_name = ['开放性', '尽责性', '外向性', '宜人性', '神经质性'][config['focus_trait']]
        training_params["训练模式"] = f"单维度训练 - {trait_name}"

    training_logger.log_training_info(model_info, data_info, training_params)

    print("="*80)
    print("大五人格图神经网络模型")
    print(f"BERT嵌入: {'启用' if config['use_bert'] else '禁用'}")
    print(f"设备: {config['device']}")
    print(f"数据路径: {config['data_path']}")
    if config['focus_trait'] is not None:
        trait_name = ['开放性', '尽责性', '外向性', '宜人性', '神经质性'][config['focus_trait']]
        print(f"单维度训练维度: {trait_name}")

    if not os.path.exists(config['data_path']):
        print(f"错误：数据文件不存在 - {config['data_path']}")
        return

    try:
        dataset = BigFiveDataset(
            data_path=config['data_path'],
            use_bert=config['use_bert']
        )

        print("\n正在根据数据集split字段进行数据划分...")

        train_indices = [i for i, item in enumerate(dataset.data) if item.get('split') == 'train']
        val_indices = [i for i, item in enumerate(dataset.data) if item.get('split') == 'valid']
        test_indices = [i for i, item in enumerate(dataset.data) if item.get('split') == 'test']

        print(f"数据划分统计：")
        print(f"  训练集: {len(train_indices)} 样本")
        print(f"  验证集: {len(val_indices)} 样本")
        print(f"  测试集: {len(test_indices)} 样本")
        print(f"  总计: {len(train_indices) + len(val_indices) + len(test_indices)} 样本")

        if len(train_indices) + len(val_indices) + len(test_indices) != len(dataset):
            print(f"\n警告：数据划分不完整！")
            print(f"  数据集总数: {len(dataset)}")
            print(f"  已划分总数: {len(train_indices) + len(val_indices) + len(test_indices)}")

            all_split_values = set(item.get('split', 'unknown') for item in dataset.data)
            print(f"  数据集中的split值: {all_split_values}")

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        if config.get('use_weighted_scene_sampler', False):
            weights = _compute_scene_weights_for_subset(
                dataset,
                train_indices,
                alpha=config.get('scene_weight_alpha', 0.5),
                w_min=config.get('scene_weight_clip_min', 1.0),
                w_max=config.get('scene_weight_clip_max', 3.0)
            )

            num_samples = int(len(train_indices) * config.get('samples_per_epoch_factor', 1.0))
            sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)

            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                sampler=sampler,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )

            w_min_value = min(weights) if weights else 0.0
            w_max_value = max(weights) if weights else 0.0
            w_median = statistics.median(weights) if weights else 0.0
            w_mean = float(sum(weights) / len(weights)) if weights else 0.0
            print("\n启用场景频率加权采样：")
            print(
                f"  alpha={config.get('scene_weight_alpha', 0.5)}, "
                f"clip=[{config.get('scene_weight_clip_min', 1.0)}, {config.get('scene_weight_clip_max', 3.0)}], "
                f"samples/epoch={num_samples}, replacement=True"
            )
            print(
                f"  权重统计 -> min={w_min_value:.3f}, median={w_median:.3f}, "
                f"mean={w_mean:.3f}, max={w_max_value:.3f}"
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        input_dim = 768
        model = BigFiveGNN(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            output_dim=15,
            num_layers=2,
            dropout=0.3
        ).to(config['device'])

        if os.path.exists(config['model_path']):
            print(f"\n检测到已存在的模型文件: {config['model_path']}，跳过训练阶段，直接加载模型进行评估。")
        else:
            trained_model = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config['epochs'],
                learning_rate=config['learning_rate'],
                device=config['device'],
                save_path=config['model_path'],
                focus_trait=config['focus_trait']
            )

            print(f"\n" + "="*80)
            print("训练完成！")
            print(f"最佳模型已保存至: {config['model_path']}")

        output_json_path = os.path.abspath(os.path.join(os.path.dirname(config['model_path']), 'prediction_visualization.json'))
        evaluate_on_test_set(config['model_path'], test_loader, config['device'], test_dataset, output_json_path, focus_trait=config['focus_trait'])

        final_metrics = {
            "训练状态": "完成",
            "模型保存路径": config['model_path'],
            "评估结果路径": output_json_path
        }

        training_logger.log_training_complete(final_metrics, 0)

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'training_logger' in locals():
            training_logger.restore_print()
            print(f"\n📝 训练日志已保存到: {training_logger.log_path}")


if __name__ == "__main__":
    main()
