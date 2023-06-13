import sys
import mindspore
import numpy as np

def pad_list(xs, pad_value):
    '''
    将不满一行的数据补齐
    :param xs:          待填充batch序列
    :param pad_value:   填充值
    '''
    # 计算batch_size
    n_batch = len(xs)
    # 计算当前batch中所有句子的最大长度
    max_len = max(x.size(0) for x in xs)
    # 创建一个新的tensor, 用于存放填充后的数据
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill(pad_value)
    # 将数据填充到新的tensor中
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


def process_dict(dict_path):
    '''
    读取文件，创建字典，返回字典列表，以及<sos>和<eos>的id
    '''
    with open(dict_path, 'rb') as f:
        dictionary = f.readlines()
    char_list = [entry.decode('utf-8').split(' ')[0] for entry in dictionary]
    sos_id = char_list.index('<sos>')
    eos_id = char_list.index('<eos>')
    return char_list, sos_id, eos_id


if __name__ == "__main__":
    path = sys.argv[1]
    char_list, sos_id, eos_id = process_dict(path)
    # 输出字典列表，以及<sos>和<eos>的id
    print(char_list, sos_id, eos_id)


# =================================================#
# ================== 预测相关函数 ==================#
# =================================================#
def parse_hypothesis(hyp, char_list):
    '''
    分析预测结果
    :param list hyp:         预测结果
    :param list char_list:   字典列表
    :return string: text       预测出的文本
    :return string: token      预测出的token
    :return string: tokenid    预测tokenid
    :return float : score      预测分数
    '''
    # 删除预测ID序列中的<sos>
    tokenid_as_list = list(map(int, hyp['yseq'][1:]))
    # 根据char_list, 获取ID对应的token
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    # 获取预测分数
    score = float(hyp['score'])

    # 将tokenid和token转换为字符串
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, score


def add_results_to_json(js, nbest_hyps, char_list):
    '''
    将最好的N个预测结果添加到json中
    :param dict js:          原始json
    :param list nbest_hyps:  N个预测结果
    :param list char_list:   字典列表
    :return dict new_js:     添加了N个预测结果的json
    '''
    
    # 复制原始json信息
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    new_js['output'] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # 解析预测结果, 将预测结果转换为字符串
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp, char_list)

        # 复制原始json中的ground-truth信息
        out_dic = dict(js['output'][0].items())

        # 更新name字段
        out_dic['name'] += '[%d]' % n

        # 添加预测结果: text, token, tokenid, score
        out_dic['rec_text'] = rec_text
        out_dic['rec_token'] = rec_token
        out_dic['rec_tokenid'] = rec_tokenid
        out_dic['score'] = score

        # 将预测结果添加到new_js中
        new_js['output'].append(out_dic)

        # 输出最优的预测结果
        if n == 1:
            print('groundtruth: %s' % out_dic['text'])
            print('prediction : %s' % out_dic['rec_text'])

    return new_js


# =================================================#
# ============== Transformer 相关函数 ==============#
# =================================================#
def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    '''
    根据 input_lengths/pad_idx 计算由于padding产生的mask, padding位对应的mask设置为1
    :param Tensor padded_input:     填充后的输入序列
    :param Tensor input_lengths:    输入序列的长度
    :param int pad_idx:             padding的位置
    :return Tensor non_pad_mask:    由于padding产生的mask
    '''
    # 如果input_lengths和pad_idx都为None, 则抛出AssertionError
    assert input_lengths is not None or pad_idx is not None
    
    # 如果 input_lengths 不为空 
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)
        # 默认mask的值全为1
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
        # 将非padding位置的mask设置为0
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
            
    # 如果 pad_idx 不为空
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        # 当前位为pad_idx的位需要被mask掉
        non_pad_mask = padded_input.equal(pad_idx).less(1).float()
            
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)


def get_subsequent_mask(seq):
    '''
    计算自注意力的mask, 用于屏蔽未来时刻的信息
    :param seq: 输入序列
    '''

    sz_b, len_s = seq.size()    
    
    # mask阵列为上三角矩阵
    subsequent_mask = mindspore.numpy.triu(
        mindspore.ops.ones((len_s, len_s), dtype=mindspore.uint8), 
        k=1
    )
    # 将mask扩展为 batch_size x len_s x len_s
    subsequent_mask = subsequent_mask.unsqueeze(0).expand((sz_b, -1, -1))  

    return subsequent_mask


def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    ''' 
    计算自注意力的mask, 用于屏蔽padding位
    :param seq_k:   key序列
    :param seq_q:   query序列
    :param pad_idx: padding的值
    '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    
    # 当前位为 pad_idx 的位置, mask为1
    padding_mask = seq_k.equal(pad_idx)
    
    # 将mask扩展为 batch_size x len_q x len_k
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)

    return padding_mask


def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    '''
    计算Encoder-Decoder注意力的mask
    :param padded_input:    填充后的输入序列
    :param input_lengths:   输入序列的长度
    :param expand_length:   扩展后的序列长度
    '''
    # 计算由于padding产生的mask, N x T_input x 1
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    
    # 删除最后一维, 并通过lt(1)将mask取反, N x T_input
    pad_mask = non_pad_mask.squeeze(-1).less(1)
    
    # 计算自注意力的mask, N x T_input x T_expand
    attn_mask = pad_mask.unsqueeze(1).expand((-1, expand_length, -1))
    
    return attn_mask
