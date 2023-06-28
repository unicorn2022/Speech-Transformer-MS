class TransformerOptimizer(object):
    '''
    A simple wrapper class for learning rate scheduling
    一个简单的学习率调度包装类
    :param optimizer:       优化器
    :param warmup_steps:    warmup步数
    :param k:               学习率的放缩因子
    '''

    def __init__(self, optimizer, warmup_steps=4000, k=0.2):
        self.optimizer = optimizer
        self.k = k
        self.warmup_steps = warmup_steps
        d_model = 512                       # embedding的维数
        self.init_lr = d_model ** (-0.5)    # 学习率的初始值
        self.lr = self.init_lr              # 学习率的当前值
        self.warmup_steps = warmup_steps    # warmup步数
        self.k = k                          # 学习率的放缩因子
        self.step_num = 0                   # 当前步数

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        '''
        更新学习率
        '''
        self.step_num += 1
        self.lr = self.k * self.init_lr * min(self.step_num ** (-0.5),
                                              self.step_num * (self.warmup_steps ** (-1.5)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
