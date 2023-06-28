import numpy as np
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    ''' 
    Scaled Dot-Product Attention计算:
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # bmm: 计算注意力权重 Q*K^T
        attn = torch.bmm(q, k.transpose(1, 2))
        
        # Scale: 将attn除以其维度的平方根
        attn = attn / self.temperature

        # Mask: 若mask中某一位为1, 则将attn中对应位置设置为-inf
        if mask is not None:
            attn = attn.masked_fill(mask.bool(), -np.inf)

        # Softmax: 对attn进行softmax
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        
        # bmm: 计算最终的输出 attn * V
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
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
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        # 初始化权重, 使其满足正态分布
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        # 计算注意力
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                   attn_dropout=dropout)
        # 归一化层
        self.layer_norm = nn.LayerNorm(d_model)

        # 全连接层, 初始化权重满足 XavierUniform 分布
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # 计算q, k, v的维度: batch大小, 序列长度, 词向量维度
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # 残差连接
        residual = q

        # 将q, k, v向下投影, 并将结果划分为n_head份
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # 将结果转置, 以便进行注意力计算
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # 构造Mask: 将mask复制(n_head * batch_size)次
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        # 计算注意力
        output, attn = self.attention(q, k, v, mask=mask)

        # 将结果转置, 以便进行全连接层计算
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)
        output = self.dropout(self.fc(output))

        # 归一化层
        output = self.layer_norm(output + residual)

        return output, attn
