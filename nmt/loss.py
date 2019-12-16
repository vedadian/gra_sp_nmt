# coding: utf-8
"""
Loss functions for neural machine translation
"""

# pylint: disable=no-member
import torch
from torch import nn
from torch.nn import functional as F

from nmt.common import configured

@configured('train')
class SmoothedCrossEntropyLoss(nn.Module):

    def __init__(self, smoothing_amount: float = 0.2):
        nn.Module.__init__(self)
        self.smoothing_amount = smoothing_amount

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        n = log_probs.size(-1)
        loss = F.nll_loss(
            log_probs.view(-1, n),
            targets.view(-1)
        )
        if self.smoothing_amount > 0:
            alpha = self.smoothing_amount / (n - 1)
            beta = self.smoothing_amount * n / (n - 1) - 1
            loss = -alpha * log_probs.sum() - beta * loss
        return loss

@configured('train')
def get_loss_function(type: str = 'smoothed_cross_entropy'):

    LOSS_EVALUATORS = {
        'smoothed_cross_entropy': SmoothedCrossEntropyLoss
    }

    if type not in LOSS_EVALUATORS:
        raise Exception('`{}` loss function is not registered.'.format(type))

    return LOSS_EVALUATORS[type]()