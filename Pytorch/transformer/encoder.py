import torch.nn as nn

from .attention import MultiHeadAttention
from .module import PositionalEncoding, PositionwiseFeedForward
from .utils import get_non_pad_mask, get_attn_pad_mask


class Encoder(nn.Module):
    '''
    Transformer的编码器, 包括自注意力和前馈网络
    :param d_input:         输入的维度
    :param n_layers:        编码器的层数
    :param n_head:          多头注意力的向下投影的次数
    :param d_k:             多头注意力中Q、K的维度
    :param d_v:             多头注意力中V的维度
    :param d_model:         词向量的维度
    :param d_inner:         前馈网络的维度
    :param dropout:         dropout的概率
    :param pe_maxlen:       位置编码的最大长度
    '''

    def __init__(self, d_input=320, n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_model=512, d_inner=2048, dropout=0.1, pe_maxlen=5000):
        super(Encoder, self).__init__()
        # 超参数设置
        self.d_input = d_input
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout_rate = dropout
        self.pe_maxlen = pe_maxlen

        # 使用Linear和LayerNorm化来替换embedding
        self.linear_in = nn.Linear(d_input, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        # 编码器层, 一共有n_layers个
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, padded_input, input_lengths, return_attns=False):
        """
        :param padded_input:    N x T x D,  经过padding后的输入序列
        :param input_lengths:   N,          输入序列的长度
        :param return_attns:    bool,       是否返回注意力权重
        :return: enc_output:    N x T x H,  编码器的输出
        """
        enc_slf_attn_list = []

        # 计算由于padding产生的mask
        non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
        length = padded_input.size(1)
        
        # 计算自注意力的mask
        slf_attn_mask = get_attn_pad_mask(padded_input, input_lengths, length)

        ### 前向计算
        # 将输入通过 Embedding 和 PositionalEncoding
        enc_output = self.dropout(
            self.layer_norm_in(self.linear_in(padded_input)) +
            self.positional_encoding(padded_input))

        # 通过N个Encoder层
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            # 记录 attention 权重
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        # 返回编码器的输出, attention权重
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class EncoderLayer(nn.Module):
    '''
    Encoder 层 = MultiHeadAttention 层 + PositionwiseFeedForward 层
    :param d_model:         词向量的维度
    :param d_inner:         前馈网络的维度
    :param n_head:          多头注意力的向下投影的次数
    :param d_k:             多头注意力中Q、K的维度
    :param d_v:             多头注意力中V的维度
    :param dropout:         dropout的概率
    '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # 多头注意力层
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        # 前馈网络层
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout
        )

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        # 将输入经过 MultiHeadAttention 层
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        # 使用由于padding产生的mask
        enc_output *= non_pad_mask

        # 经过前馈网络层
        enc_output = self.pos_ffn(enc_output)
        
        # 使用由于padding产生的mask
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
