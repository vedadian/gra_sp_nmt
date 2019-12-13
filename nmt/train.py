# coding: utf-8
"""
Routines related to training a neural machine translation model
"""

from nmt.common import configured, get_logger
from nmt.dataset import get_train_dataset, get_validation_dataset
from nmt.model import build_model
from nmt.loss import get_loss_function
from nmt.optimization import get_optimizer

@configured('train')
def train(
    max_steps: int = 100,
    batch_size_limit: int = 400,
    batch_limit_by_tokens: bool = True,
    report_interval_steps: int = 10,
    validation_interval_steps: int = 100,
    use_gpu: bool = False
):
    
    logger = get_logger()

    model = build_model()
    loss_function = get_loss_function()
    optimizer = get_optimizer(model.parameters())

    train_dataset = get_train_dataset()
    validation_dataset = get_validation_dataset()

    step = 0
    while step < max_steps:
        for batch in train_dataset.iterate(
            'cuda' if use_gpu else 'cpu',
            batch_size_limit,
            batch_limit_by_tokens
        ):
            step += 1
            if step > 0 and step % report_interval_steps == 0:
                logger.info('step# {}'.format(step))
            if step > 0 and step % validation_interval_steps == 0:
                logger.info('validating# {}'.format(step))
            if step >= max_steps:
                break