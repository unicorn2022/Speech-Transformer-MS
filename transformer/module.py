import mindspore
import mindspore.nn as nn
import mindspore.numpy as np
from mindspore.ops import functional as F

class PositionalEncoding(nn.Cell):
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
        pe_embedding_table = F.zeros((max_len, d_model), mindspore.float32)
        
        # 构造 pos 矩阵: max_len × 1列向量, 值为 0 ~ max_len-1
        pos_mat = np.arange(0, max_len).reshape((-1, 1)).float()
        # 构造div_term矩阵: 1 × d_model行向量, 值为 10000^(2i/d_model)
        div_term = np.arange(0, d_model, 2).float() / d_model
        div_term = F.pow(10000.0, div_term)
        # 计算位置编码
        pe_embedding_table[:, 0::2] = mindspore.ops.sin(pos_mat / div_term)
        pe_embedding_table[:, 1::2] = mindspore.ops.cos(pos_mat / div_term)
        # 扩展一维, 用于一个 batch 的训练
        pe_embedding_table = pe_embedding_table.expand_dims(0)
        
        # 将 pe_embedding_table保存下来
        self.pe_embedding_table = pe_embedding_table
        # self.register_buffer('pe_embedding_table', pe_embedding_table)

    def construct(self, input):
        '''
        :param input: N x T x D
        '''
        length = input.size(1)
        return self.pe_embedding_table[:, :length]


class PositionwiseFeedForward(nn.Cell):
    '''
    实现 position-wise feedforward 子层
        FFN(x) = max(0, xW1 + b1)W2 + b2
    '''

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Dense(d_model, d_ff)
        self.w_2 = nn.Dense(d_ff, d_model)
        self.dropout = nn.Dropout(1 - dropout)
        self.layer_norm = nn.LayerNorm([d_model])

    def construct(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


# Another implementation
class PositionwiseFeedForwardUseConv(nn.Cell):
    '''
    two-feed-forward-layer模块
    '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForwardUseConv, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm([d_in])
        self.dropout = nn.Dropout(1 - dropout)

    def construct(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
