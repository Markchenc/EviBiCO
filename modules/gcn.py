"""
GCN模块 - 图卷积网络层
"""

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GCN(nn.Module):
    """图卷积网络层"""

    def __init__(self, in_features, out_features, bias=True):
        """
        初始化GCN层

        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            bias: 是否使用偏置
        """
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            torch.empty([in_features, out_features], dtype=torch.float),
            requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty([out_features], dtype=torch.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """重置参数"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, inputs, identity=False):
        """
        前向传播

        Args:
            adj: 邻接矩阵
            inputs: 输入特征
            identity: 是否使用单位矩阵模式

        Returns:
            输出特征
        """
        if identity:
            return torch.matmul(adj, self.weight)
        return torch.matmul(adj, torch.matmul(inputs, self.weight))
