import torch
import torch.nn.functional as F

from config import IGNORE_ID


def cal_performance(pred, gold, smoothing=0.0):
    """
    计算交叉熵损失函数, 如果需要则应用标签平滑
    :param pred: N x T x C, score before softmax
    :param gold: N x T
    """

    pred = pred.view(-1, pred.size(2))
    gold = gold.contiguous().view(-1)

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(IGNORE_ID)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing=0.0):
    """
    计算交叉熵损失函数, 如果需要则应用标签平滑
    """

    if smoothing > 0.0:
        eps = smoothing
        n_class = pred.size(1)

        # 生成 N x C 的one-hot 矩阵: 只有标签位置是1, 所有其他位置都是0
        # gold包含-1值（IGNORE_ID）, 这将导致 assert error
        gold_for_scatter = gold.ne(IGNORE_ID).long() * gold
        one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(IGNORE_ID)
        n_word = non_pad_mask.sum().item()
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum() / n_word
    else:
        loss = F.cross_entropy(
            pred, gold,
            ignore_index=IGNORE_ID,
            reduction='elementwise_mean'
        )

    return loss
