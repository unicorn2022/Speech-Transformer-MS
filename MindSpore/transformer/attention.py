import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, Normal, XavierUniform

class ScaledDotProductAttention(nn.Cell):
    ''' 
    Scaled Dot-Product Attention计算:
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(1.0 - attn_dropout)
        self.softmax = nn.Softmax(axis=2)
        self.batmatmul = ops.BatchMatMul()

    def construct(self, q, k, v, mask=None):
        # MatMul: 计算注意力权重 Q*K^T
        attn = self.batmatmul(q, k.transpose((0, 2, 1)))
        
        # Scale: 将attn除以其维度的平方根
        attn = attn / self.temperature

        # Mask: 若mask中某一位为1, 则将attn中对应位置设置为-inf
        if mask is not None:
            attn = attn.masked_fill(mask.bool(), -np.inf)

        # Softmax: 对attn进行softmax
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        
        # MatMul: 计算最终的输出 attn * V
        output = self.batmatmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Cell):
    ''' 
    Multi-Head Attention module 多头注意力模块
    :param n_head:  Q、K、V向下投影的次数
    :param d_model: 词向量的维数
    :param d_k:     Q、K向下投影的维数
    :param d_v:     V向下投影的维数
    '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 将q, k, v向下投影, 维数为d_k/d_v, 投影n_head次
        self.w_qs = nn.Dense(d_model, n_head * d_k)
        self.w_ks = nn.Dense(d_model, n_head * d_k)
        self.w_vs = nn.Dense(d_model, n_head * d_v)
        # 初始化权重, 使其满足正态分布
        self.w_qs.weight = initializer(
            Normal(mean=0, sigma=np.sqrt(2.0 / (d_model + d_k))), 
            shape=self.w_qs.weight.shape, 
            dtype=self.w_qs.weight.dtype
        )
        self.w_ks.weight = initializer(
            Normal(mean=0, sigma=np.sqrt(2.0 / (d_model + d_k))), 
            shape=self.w_ks.weight.shape, 
            dtype=self.w_ks.weight.dtype
        )
        self.w_vs.weight = initializer(
            Normal(mean=0, sigma=np.sqrt(2.0 / (d_model + d_v))), 
            shape=self.w_vs.weight.shape, 
            dtype=self.w_vs.weight.dtype
        )
        
        # 计算注意力
        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5),
            attn_dropout=dropout
        )
        # 归一化层
        self.layer_norm = nn.LayerNorm([d_model])
        
        # 全连接层, 初始化权重满足 XavierUniform 分布
        self.fc = nn.Dense(n_head * d_v, d_model)
        self.fc.weight = initializer(
            XavierUniform(), 
            shape=self.fc.weight.shape, 
            dtype=self.fc.weight.dtype
        )

        self.dropout = nn.Dropout(1.0 - dropout)

    def construct(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # 计算q, k, v的维度: batch大小, 序列长度, 词向量维度
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # 残差连接
        residual = q

        # 将q, k, v向下投影, 并将结果划分为n_head份
        q = self.w_qs(q).view((sz_b, len_q, n_head, d_k))
        k = self.w_ks(k).view((sz_b, len_k, n_head, d_k))
        v = self.w_vs(v).view((sz_b, len_v, n_head, d_v))

        # 将结果转置, 以便进行注意力计算
        q = q.transpose((2, 0, 1, 3)).contiguous().view((-1, len_q, d_k))  # (n_head * batch_size) x len_q x d_k
        k = k.transpose((2, 0, 1, 3)).contiguous().view((-1, len_k, d_k))  # (n_head * batch_size) x len_k x d_k
        v = v.transpose((2, 0, 1, 3)).contiguous().view((-1, len_v, d_v))  # (n_head * batch_size) x len_v x d_v

        # 构造Mask: 将mask复制(n_head * batch_size)次
        if mask is not None:
            mask = np.tile(mask, [n_head, 1, 1])  # (n_head * batch_size) x .. x ..

        # 计算注意力
        output, attn = self.attention(q, k, v, mask=mask)

        # 将结果转置, 以便进行全连接层计算
        output = output.view((n_head, sz_b, len_q, d_v))
        output = output.transpose((1, 2, 0, 3)).contiguous().view((sz_b, len_q, -1))  # batch_size x len_q x (n_head * d_v)
        output = self.dropout(self.fc(output))
        
        # 归一化层
        output = self.layer_norm(output + residual)

        return output, attn