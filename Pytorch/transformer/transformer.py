import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class Transformer(nn.Module):
    '''
    Transformer网络
    :param encoder: Transformer的编码器
    :param decoder: Transformer的解码器
    '''

    def __init__(self, encoder=None, decoder=None):
        super(Transformer, self).__init__()

        # 定义encoder和decoder, 并初始化参数
        if encoder is not None and decoder is not None:
            self.encoder = encoder
            self.decoder = decoder

            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            self.encoder = Encoder()
            self.decoder = Decoder()

    def forward(self, padded_input, input_lengths, padded_target):
        '''
        训练过程
        :param padded_input:    N x T_input x D, 经过对齐后的输入
        :param input_lengths:   N,               每个batch包含的输入序列个数    
        :param padded_targets:  N x T_output,    经过对齐后的目标输出   
        '''
        # 将输入送入encoder, 得到encoder的输出
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        # pred is score before softmax
        # 将decoder的目标输出、encoder的输出送入decoder, 得到decoder的下一个输出
        # 长度最长为 input_lengths
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs, input_lengths)
        return pred, gold

    def recognize(self, input, input_length, char_list, args):
        '''
        Sequence-to-Sequence beam search, decode one utterence now.
        预测过程: 使用seq2seq beam search算法, 一次预测一个句子
        :param input:           T x D, 输入序列
        :param input_length:    输入序列的长度
        :param char_list:       字符列表
        :param args:            args.beam
        :return nbest_hyps:     n个最优的解码结果
        '''
        # 将输入送入encoder, 得到encoder的输出
        # 要将input扩展一维, 因为encoder的输入是三维的
        encoder_outputs, *_ = self.encoder(input.unsqueeze(0), input_length)
        # 将encoder的输出、字典、beam search的参数送入decoder, 得到解码结果
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0], char_list, args)
        # 返回最优的n个解码结果
        return nbest_hyps
