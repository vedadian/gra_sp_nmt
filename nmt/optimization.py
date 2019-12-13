# coding: utf-8
"""
Optimizers
"""

from typing import Any, Tuple

from torch.optim import Adam

from nmt.common import configuration, configured

configuration.ensure_submodule('train').ensure_submodule('optimizer').ensure_param('type', 'adam')

def get_optimizer(params: Any):

    @configured('train.optimizer')
    def get_adam_optimizer(
        lr: float = 1e-4,
        betas: Tuple[float] = (0.9, 0.999),
        eps: float = 1e-7,
        weight_decay: float = 0.8
    ):
        return Adam(params, lr, betas, eps, weight_decay)
    
    OPTIMIZERS = {
        'adam': get_adam_optimizer
    }

    type_ = configuration.ensure_submodule('train').ensure_submodule('optimizer').type
    if type_ not in OPTIMIZERS:
        raise Exception('Optimizer `{}` is not registered.'.format(type_))
    
    return OPTIMIZERS[type_]()