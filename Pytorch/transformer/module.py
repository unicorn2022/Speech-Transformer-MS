import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    '''
    实现Position Encoding函数
        PE(pos, 2i)   = sin(pos / (10000^(2i/d_model)))
        PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
    使用sin/cos的优点:
        1.泛化能力较强
        2.具有对称性
        3.具有唯一性: 每个位置的embedding是确定的
    :param d_model: embedding向量的维数
    :param max_len: 序列的最大长度
    '''

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 位置编码矩阵 = pos * div_term
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        
        # 构造 pos 矩阵: max_len × 1列向量, 值为 0 ~ max_len-1
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # 构造div_term矩阵: 1 × d_model行向量, 值为 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 扩展一维, 用于一个 batch 的训练
        pe = pe.unsqueeze(0)
        
        # 将 pe_embedding_table保存下来, 并设置为不可学习参数
        self.register_buffer('pe', pe)

    def forward(self, input):
        '''
        :param input: N x T x D
        '''
        length = input.size(1)
        return self.pe[:, :length]


class PositionwiseFeedForward(nn.Module):
    '''
    实现 position-wise feedforward 子层
        FFN(x) = max(0, xW1 + b1)W2 + b2
    '''

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


# Another implementation
class PositionwiseFeedForwardUseConv(nn.Module):
    '''
    two-feed-forward-layer模块
    '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForwardUseConv, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
