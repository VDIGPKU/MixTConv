# Code for MixTConv: Mixed Temporal Convolutional Kernels for Efficient Action Recognition
# arXiv: https://arxiv.org/abs/2001.06769
# Kaiyu Shan
# shankyle@pku.edu.cn

import numpy as np


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class EarlyAjust:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta
        self.earlyadj = 0

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            print('EarlyAdjust counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.earlyadj += 1
                self.counter = 0
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0


class EarlyAjustAc:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta
        self.earlyadj = 0

    def __call__(self, accuracy):
        score = accuracy

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            print('EarlyAdjustAC counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.earlyadj += 1
                self.counter = 0
        else:
            self.best_score = score
            self.counter = 0


class EarlyAjustPre:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.pre_score = None
        self.val_loss_min = np.Inf
        self.delta = delta
        self.earlyadj = 0

    def __call__(self, val_loss):
        score = -val_loss

        if self.pre_score is None:
            self.pre_score = score
        elif score < self.pre_score - self.delta:
            self.counter += 1
            self.pre_score = score
            print('EarlyAdjust counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.earlyadj += 1
                self.counter = 0
        else:
            self.pre_score = score
            self.val_loss_min = val_loss
            self.counter = 0
