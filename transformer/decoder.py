import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer, Normal, XavierUniform


import config
from .attention import MultiHeadAttention
from .module import PositionalEncoding, PositionwiseFeedForward
from .utils import get_attn_key_pad_mask, get_attn_pad_mask, get_non_pad_mask, get_subsequent_mask, pad_list

IGNORE_ID = config.cfg.IGNORE_ID

class Decoder(nn.Cell):
    '''
    Transformer的解码器, 包含自注意力 + 编码器-解码器注意力 + 前馈网络
    :param sos_id:          <start_of_seq>的id
    :param eos_id:          <end_of_seq>的id
    :param n_tgt_vocab:     目标语言词表的大小
    :param d_word_vec:      词向量的维度
    :param n_layers:        编码器的层数
    :param n_head:          多头注意力的向下投影的次数
    :param d_k:             多头注意力中Q、K的维度
    :param d_v:             多头注意力中V的维度
    :param d_model:         词向量的维度
    :param d_inner:         前馈网络的维度
    :param dropout:         dropout的概率
    :param tgt_emb_prj_weight_sharing: 是否共享目标词向量层和线性层的权重
    :param pe_maxlen:       位置编码的最大长度
    '''

    def __init__(
            self, sos_id=0, eos_id=1,
            n_tgt_vocab=4335, d_word_vec=512,
            n_layers=6, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=2048, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            pe_maxlen=5000):
        super(Decoder, self).__init__()
        
        # 超参数
        self.sos_id = sos_id  # start_of_seq
        self.eos_id = eos_id  # end_of_seq
        self.n_tgt_vocab = n_tgt_vocab
        self.d_word_vec = d_word_vec
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.tgt_emb_prj_weight_sharing = tgt_emb_prj_weight_sharing
        self.pe_maxlen = pe_maxlen

        # Embedding层
        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)
        
        # 位置编码层
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        # 解码器层
        self.layer_stack = nn.CellList(
            [DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)]
        )

        # 全连接层, 初始化权重满足 XavierUniform 分布
        self.tgt_word_prj = nn.Dense(d_model, n_tgt_vocab, has_bias=False)
        self.tgt_word_prj.weight = initializer(
            XavierUniform(), 
            self.tgt_word_prj.weight.shape,
            dtype=mindspore.float32
        )

        # 是否共享目标词向量层和全连接层的权重
        if tgt_emb_prj_weight_sharing:
            # 共享目标词向量层和全连接层的权重
            self.tgt_word_prj.weight = self.tgt_word_emb.embedding_table
            # 权重缩放比为 1/sqrt(d_model)
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def preprocess(self, padded_input):
        '''
        预处理输入序列 padded_input
            添加 <sos> 到 decoder 的输入, 添加 <eos> 到 decoder 的输出标签
        :param padded_input:    N x Ti, 经过padding后的输入序列
        :return ys_in_pad:      N x To, 填充<sos>后的输入序列
        :return ys_out_pad:     N x To, 填充IGNORE_ID后的输出序列
        '''
        # 删除padded_input中需要忽略的数据
        ys = [y[y != IGNORE_ID] for y in padded_input]  
        
        # input中的每个句子添加<sos>，output中的每个句子添加<eos>
        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [np.concatenate([sos, y], axis=0) for y in ys]
        ys_out = [np.concatenate([y, eos], axis=0) for y in ys]
        
        # 将input中的空白位置填充为 <eos>
        ys_in_pad = pad_list(ys_in, self.eos_id)
        # 将output中的空白位置填充为 IGNORE_ID
        ys_out_pad = pad_list(ys_out, IGNORE_ID)
        
        assert ys_in_pad.size() == ys_out_pad.size()
        
        return ys_in_pad, ys_out_pad

    def construct(self, padded_input, encoder_padded_outputs,
                encoder_input_lengths, return_attns=False):
        '''
        :param padded_input:            N x To, 经过padding后的输入序列
        :param encoder_padded_outputs:  N x Ti x H, 编码器的输出
        :param encoder_input_lengths:   N, 编码器输入序列的长度
        :param return_attns:            是否返回注意力权重
        :return: pred:                  N x To x V, 预测的输出序列
        :return: gold:                  N x To, 真实的输出序列
        '''
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # 获取预处理后的decoder的输入和输出序列
        ys_in_pad, ys_out_pad = self.preprocess(padded_input)

        # 计算由于由于padding产生的mask
        non_pad_mask = get_non_pad_mask(ys_in_pad, pad_idx=self.eos_id)

        # 计算自注意力的mask = 由于时间产生的mask + 由于padding产生的mask
        slf_attn_mask_subseq = get_subsequent_mask(ys_in_pad)
        slf_attn_mask_keypad = get_attn_key_pad_mask(
            seq_k=ys_in_pad,
            seq_q=ys_in_pad,
            pad_idx=self.eos_id
        )
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).greater(0)

        # 输出序列的长度
        output_length = ys_in_pad.size(1)
        
        # 计算编码器-解码器注意力的mask
        dec_enc_attn_mask = get_attn_pad_mask(
            encoder_padded_outputs,
            encoder_input_lengths,
            output_length
        )

        ### 前向计算
        # 将输入通过 Embedding 和 PositionalEncoding
        dec_output = self.dropout(
            self.tgt_word_emb(ys_in_pad) * self.x_logit_scale +
            self.positional_encoding(ys_in_pad)
        )
        # 通过N个Decoder层
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, 
                encoder_padded_outputs,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask
            )
            # 记录 attention 权重
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]
        # 经过全连接层
        seq_logit = self.tgt_word_prj(dec_output)

        # 返回: 预测结果pred、Decoder的输出gold
        pred, gold = seq_logit, ys_out_pad
        if return_attns:
            return pred, gold, dec_slf_attn_list, dec_enc_attn_list
        return pred, gold

    def recognize_beam(self, encoder_outputs, char_list, args):
        '''
        使用beam search算法生成目标序列
        :param encoder_outputs:     T x H, 编码器的输出
        :param char_list:           字典列表
        :param args:                参数
        :return: nbest_hyps:        n个最优预测结果
        '''
        # 参数
        beam = args.beam_size
        nbest = args.nbest
        if args.decode_max_len == 0:
            maxlen = encoder_outputs.size(0)
        else:
            maxlen = args.decode_max_len

        encoder_outputs = encoder_outputs.unsqueeze(0)

        # 初始化输入序列为: [<sos>]
        ys = ops.ones(1, 1).fill(self.sos_id).astype(encoder_outputs.dtype).long()

        # yseq: 1xT
        # 初始化预测结果: score=0.0, yseq=[<sos>]
        hyp = {'score': 0.0, 'yseq': ys}
        hyps = [hyp]
        ended_hyps = []

        # 重复执行最多 maxlen 次预测
        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                # 获取当前已经预测出来的序列yseq:  1 x i
                ys = hyp['yseq']
                # 计算由于由于padding产生的mask: 1xix1
                non_pad_mask = ops.ones_like(ys).float().unsqueeze(-1) 
                slf_attn_mask = get_subsequent_mask(ys)

                ### Decoder的前向计算
                # 将输入通过 Embedding 和 PositionalEncoding
                dec_output = self.dropout(
                    self.tgt_word_emb(ys) * self.x_logit_scale +
                    self.positional_encoding(ys)
                )
                # 通过N个Decoder层
                for dec_layer in self.layer_stack:
                    dec_output, _, _ = dec_layer(
                        dec_output, encoder_outputs,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask,
                        dec_enc_attn_mask=None
                    )
                # 经过全连接层
                seq_logit = self.tgt_word_prj(dec_output[:, -1])
                # 经过softmax层, 得到当前时刻的输出结果
                log_softmax = nn.LogSoftmax(axis=1)
                local_scores = log_softmax(seq_logit)
                # 得到前k个scores, 以及对应的预测结果
                local_best_scores, local_best_ids = ops.TopK(sorted=True)(local_scores, beam)

                # 将当前时刻的预测结果与之前的预测结果拼接起来
                for j in range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = ops.ones((1, (1 + ys.size(1))), mindspore.float32).astype(encoder_outputs.dtype).long()
                    new_hyp['yseq'][:, :ys.size(1)] = hyp['yseq']
                    new_hyp['yseq'][:, ys.size(1)] = int(local_best_ids[0, j])
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept,
                    key=lambda x: x['score'],
                    reverse=True)[:beam]
            # end for hyp in hyps
            hyps = hyps_best_kept

            # 如果进行了maxlen次预测, 则向预测结果中添加<eos>, 防止预测结果没有终结符
            if i == maxlen - 1:
                for hyp in hyps:
                    hyp['yseq'] = np.concatenate(
                        [hyp['yseq'], ops.ones((1, 1), mindspore.float32).fill(self.eos_id).astype(encoder_outputs.dtype).long()],
                        axis=1
                    )

            # 添加已经预测出来的序列到ended_hyps中
            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][0, -1] == self.eos_id:
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            hyps = remained_hyps
            # if len(hyps) > 0:
            #     print('remeined hypothes: ' + str(len(hyps)))
            # else:
            #     print('no hypothesis. Finish decoding.')
            #     break
            #
            # for hyp in hyps:
            #     print('hypo: ' + ''.join([char_list[int(x)]
            #                               for x in hyp['yseq'][0, 1:]]))
        # end for i in range(maxlen)
       
        # 根据score对预测结果进行排序
        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), nbest)]
        
        # compitable with LAS implementation
        # 将结果转换为list
        for hyp in nbest_hyps:
            hyp['yseq'] = hyp['yseq'][0].cpu().numpy().tolist()
        return nbest_hyps


class DecoderLayer(nn.Cell):
    '''
    Decoder 层 = Self-Attention 层 + Encoder-Attention 层 + PositionwiseFeedForward 层
    :param d_model:         词向量的维度
    :param d_inner:         前馈网络的维度
    :param n_head:          多头注意力的向下投影的次数
    :param d_k:             多头注意力中Q、K的维度
    :param d_v:             多头注意力中V的维度
    :param dropout:         dropout的概率
    '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 自注意力层
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # 编码器-解码器注意力层
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # 前馈全连接层
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def construct(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        # 将解码器的输入通过自注意力层
        dec_output, dec_slf_attn = self.self_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask
        )
        
        # 应用由于padding产生的mask
        dec_output *= non_pad_mask

        # 通过编码器-解码器注意力层
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask
        )
        
        # 使用由于padding产生的mask
        dec_output *= non_pad_mask

        # 通过前馈全连接层
        dec_output = self.pos_ffn(dec_output)
        
        # 使用由于padding产生的mask
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
