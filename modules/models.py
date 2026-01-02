"""
模型模块 - GCNBlock和BigFiveGNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from .gcn import GCN as CustomGCN

from .config import get_model_path


class GCNBlock(nn.Module):
    """
    GCN块：包含GCN层 + LayerNorm + Activation + Dropout

    将原来的分散处理组合成一个统一的块，提供：
    1. 更清晰的模块化设计
    2. 统一的参数管理
    3. 更好的可维护性
    4. 支持不同的配置（dropout率等）
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        """
        初始化GCN块

        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            dropout: dropout率
        """
        super(GCNBlock, self).__init__()

        self.gcn = CustomGCN(in_features, out_features, bias=True)
        self.layer_norm = nn.LayerNorm(out_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        """重置参数"""
        self.gcn.reset_parameters()

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            adj: 邻接矩阵 [num_nodes, num_nodes] 或 [num_edges, 2]
            x: 输入特征 [num_nodes, in_features]

        Returns:
            output: 输出特征 [num_nodes, out_features]
        """
        x = self.gcn(adj, x)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class BigFiveGNN(nn.Module):
    """改进的图神经网络模型 - 使用GCNBlock进行模块化设计"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 15,
                 num_layers: int = 3, dropout: float = 0.3):
        """
        初始化模型

        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（15维：5个人格特质×3个类别）
            num_layers: GCN层数
            dropout: dropout率
        """
        super(BigFiveGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_classes = 3

        try:
            model_path = get_model_path()
            print(f"从本地路径加载BERT模型: {model_path}")
            self.bert_model = BertModel.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"BERT模型加载失败: {e}")
            raise RuntimeError(f"无法加载BERT模型: {e}")

        self.gcn_blocks = nn.ModuleList()

        first_block_dropout = dropout * 0.5
        self.gcn_blocks.append(GCNBlock(
            self.bert_model.config.hidden_size,
            hidden_dim,
            first_block_dropout
        ))

        for i in range(num_layers - 1):
            block_dropout = dropout * (0.5 + 0.5 * (i + 1) / (num_layers - 1)) if num_layers > 1 else dropout
            self.gcn_blocks.append(GCNBlock(
                hidden_dim,
                hidden_dim,
                block_dropout
            ))

        self.personality_classifiers = nn.ModuleList([
            self._create_enhanced_classifier(hidden_dim) for _ in range(5)
        ])

        self.threshold_predictor = nn.Sequential(
            nn.Linear(self.num_classes, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

        self._initialize_weights()

    def _create_enhanced_classifier(self, input_dim: int):
        """
        创建分类器

        Args:
            input_dim: 输入特征维度

        Returns:
            nn.Sequential: 分类器
        """
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(input_dim // 2, 3)
        )

    def _initialize_weights(self):
        """初始化模型权重"""
        for gcn_block in self.gcn_blocks:
            gcn_block.reset_parameters()

        for classifier in self.personality_classifiers:
            for layer in classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        for layer in self.threshold_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, input_ids, attention_mask, personality_mask, adjacency_matrix, batch=None):
        """
        前向传播 - 使用GCNBlock

        Args:
            input_ids: 节点文本的token ids [num_nodes, max_length]
            attention_mask: 节点文本的注意力mask [num_nodes, max_length]
            personality_mask: 人格节点掩码 [num_nodes]
            adjacency_matrix: 邻接矩阵 [num_nodes, num_nodes]
            batch: 批次索引 [num_nodes] (用于批处理)

        Returns:
            output: 输出 [batch_size, 5, num_classes] 或 [5, num_classes] (非批处理)
            evidence_predictions: 证据句预测结果
        """
        L_max = int(attention_mask.sum(dim=1).max().item()) if attention_mask.dim() == 2 else attention_mask.size(0)
        L_max = max(L_max, 1)
        input_ids = input_ids[:, :L_max]
        attention_mask = attention_mask[:, :L_max]
        bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        h = bert_outputs.last_hidden_state[:, 0, :]

        adj_dense = self._create_weighted_adjacency_matrix(h, adjacency_matrix)

        for gcn_block in self.gcn_blocks:
            h = gcn_block(adj_dense, h)

        personality_indices = torch.where(personality_mask)[0]
        personality_features = h[personality_indices]

        if batch is not None:
            batch_size = batch.max().item() + 1
            personality_batch = batch[personality_indices]
            batch_personality_features = torch.zeros(batch_size, 5, h.shape[-1], device=h.device)

            for b in range(batch_size):
                mask = personality_batch == b
                current_personality_features = personality_features[mask]
                assert current_personality_features.shape[0] == 5, f"Expected 5 personality nodes for batch {b}, got {current_personality_features.shape[0]}"
                batch_personality_features[b] = current_personality_features

            personality_outputs = []
            for i in range(5):
                trait_features = batch_personality_features[:, i, :]
                trait_output = self.personality_classifiers[i](trait_features)
                personality_outputs.append(trait_output)

            output = torch.stack(personality_outputs, dim=1)
            dynamic_thresholds = self._compute_dynamic_thresholds(output)
            evidence_predictions, evidence_predictions_raw = self._predict_evidence_sentences(
                adjacency_matrix, adj_dense, dynamic_thresholds, batch
            )

        else:
            assert personality_features.shape[0] == 5, f"Expected 5 personality nodes, got {personality_features.shape[0]}"

            personality_outputs = []
            for i in range(5):
                trait_output = self.personality_classifiers[i](personality_features[i])
                personality_outputs.append(trait_output)

            output = torch.stack(personality_outputs, dim=0)
            dynamic_thresholds = self._compute_dynamic_thresholds(output)
            evidence_predictions, evidence_predictions_raw = self._predict_evidence_sentences(
                adjacency_matrix, adj_dense, dynamic_thresholds, batch=None
            )

        return output, evidence_predictions, evidence_predictions_raw

    def _create_weighted_adjacency_matrix(self, embeddings, adjacency_matrix):
        """
        创建带权重的密集邻接矩阵

        Args:
            embeddings: 节点embeddings [num_nodes, embedding_dim]
            adjacency_matrix: 原始0/1邻接矩阵 [num_nodes, num_nodes]

        Returns:
            adj_dense: 带权重的密集邻接矩阵 [num_nodes, num_nodes]
        """
        device = adjacency_matrix.device
        adj_dense = torch.zeros_like(adjacency_matrix, dtype=torch.float, device=device)
        row, col = adjacency_matrix.nonzero(as_tuple=True)

        if len(row) > 0:
            edge_weights = self._compute_similarity_weights(embeddings, row, col)
            adj_dense[row, col] = edge_weights

        return adj_dense

    def _compute_similarity_weights(self, embeddings, row, col):
        """
        计算相似度权重，注意看相似度计算是怎么处理row和col这两个一维张量的

        Args:
            embeddings: 节点嵌入
            row: 源节点索引
            col: 目标节点索引

        Returns:
            edge_weights: 边权重
        """
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        src_embeddings = embeddings_norm[row]
        dst_embeddings = embeddings_norm[col]
        edge_weights = torch.sum(src_embeddings * dst_embeddings, dim=1)
        edge_weights = (edge_weights + 1.0) / 2.0
        edge_weights = torch.clamp(edge_weights, min=0.05, max=0.95)
        return edge_weights

    def _compute_dynamic_thresholds(self, personality_logits):
        """
        根据人格预测logits计算动态阈值

        Args:
            personality_logits: 人格预测logits [batch_size, 5, 3] 或 [5, 3]

        Returns:
            dynamic_thresholds: 动态阈值 [batch_size, 5] 或 [5]，范围[0.2, 0.75]
        """
        original_shape = personality_logits.shape
        is_batched = len(original_shape) == 3

        if is_batched:
            batch_size, num_traits, num_classes = personality_logits.shape
            logits_reshaped = personality_logits.view(-1, num_classes)
        else:
            num_traits, num_classes = personality_logits.shape
            logits_reshaped = personality_logits

        raw_thresholds = self.threshold_predictor(logits_reshaped).squeeze(-1)
        dynamic_thresholds = 0.2 + 0.55 * torch.sigmoid(raw_thresholds)

        if is_batched:
            dynamic_thresholds = dynamic_thresholds.view(batch_size, num_traits)

        return dynamic_thresholds

    def _predict_evidence_sentences(self, adjacency_matrix, adj_dense, dynamic_thresholds=None, batch=None):
        """
        预测证据句：基于连接到五个人格节点的边权重与动态阈值比较来标记证据句
        【动态阈值版本】根据人格预测动态调整判断阈值

        Args:
            adjacency_matrix: 原始邻接矩阵 (num_nodes, num_nodes)
            adj_dense: 加权邻接矩阵 (num_nodes, num_nodes)
            dynamic_thresholds: 动态阈值 [batch_size, 5] 或 [5]，范围[0.2, 0.75]
            batch: 批次索引 [num_nodes]（用于批处理模式）

        Returns:
            evidence_predictions: 字典，key为人格维度名称，value为证据句标记的一维张量
            evidence_predictions_raw: 原始证据句权重列表
        """
        if dynamic_thresholds is None:
            if batch is not None:
                batch_size = batch.max().item() + 1
                dynamic_thresholds = torch.full((batch_size, 5), 0.5, device=adjacency_matrix.device)
            else:
                dynamic_thresholds = torch.full((5,), 0.5, device=adjacency_matrix.device)

        num_nodes = adjacency_matrix.size(0)
        personality_dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

        if batch is not None:
            batch_size = batch.max().item() + 1
            evidence_predictions = {dim: [] for dim in personality_dims}
            evidence_predictions_raw = [[] for _ in range(5)]

            for b in range(batch_size):
                batch_mask = (batch == b)
                sample_indices = torch.where(batch_mask)[0]

                sample_num_nodes = batch_mask.sum().item()
                sample_num_utterance_nodes = sample_num_nodes - 5

                for dim_idx, dim_name in enumerate(personality_dims):
                    personality_node_local_idx = sample_num_utterance_nodes + dim_idx
                    personality_node_global_idx = sample_indices[personality_node_local_idx]
                    threshold = dynamic_thresholds[b, dim_idx].item()

                    evidence_tensor = torch.zeros(sample_num_utterance_nodes, dtype=torch.long, device=adjacency_matrix.device)
                    evidence_tensor_raw = torch.zeros(sample_num_utterance_nodes, dtype=torch.float, device=adjacency_matrix.device)

                    connected_edges = adjacency_matrix[:, personality_node_global_idx].nonzero(as_tuple=True)[0]

                    for row_idx in connected_edges:
                        if batch[row_idx] == b:
                            local_row_idx = (batch[:row_idx+1] == b).sum().item() - 1

                            if local_row_idx < sample_num_utterance_nodes:
                                edge_weight = adj_dense[row_idx, personality_node_global_idx]
                                evidence_tensor_raw[local_row_idx] = edge_weight

                                if edge_weight > threshold:
                                    evidence_tensor[local_row_idx] = 1

                    evidence_predictions[dim_name].append(evidence_tensor)
                    evidence_predictions_raw[dim_idx].append(evidence_tensor_raw)

            evidence_predictions_raw = [torch.cat(dim_tensors, dim=0) for dim_tensors in evidence_predictions_raw]

            for dim_name in personality_dims:
                evidence_predictions[dim_name] = torch.cat(evidence_predictions[dim_name], dim=0)

        else:
            num_utterance_nodes = num_nodes - 5
            evidence_predictions = {}
            evidence_predictions_raw = []

            for dim_idx, dim_name in enumerate(personality_dims):
                personality_col_idx = num_utterance_nodes + dim_idx
                threshold = dynamic_thresholds[dim_idx].item()

                evidence_tensor = torch.zeros(num_utterance_nodes, dtype=torch.long, device=adjacency_matrix.device)
                evidence_tensor_raw = torch.zeros(num_utterance_nodes, dtype=torch.float, device=adjacency_matrix.device)

                connected_edges = adjacency_matrix[:, personality_col_idx].nonzero(as_tuple=True)[0]

                for row_idx in connected_edges:
                    if row_idx < num_utterance_nodes:
                        edge_weight = adj_dense[row_idx, personality_col_idx]
                        evidence_tensor_raw[row_idx] = edge_weight

                        if edge_weight > threshold:
                            evidence_tensor[row_idx] = 1

                evidence_predictions[dim_name] = evidence_tensor
                evidence_predictions_raw.append(evidence_tensor_raw)

        return evidence_predictions, evidence_predictions_raw
