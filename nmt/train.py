# coding: utf-8
"""
Routines related to training a neural machine translation model
"""

# pylint: disable=no-member
import torch

from nmt.common import configured, get_logger
from nmt.dataset import get_train_dataset, get_validation_dataset
from nmt.model import build_model
from nmt.loss import get_loss_function
from nmt.optimization import get_optimizer

# pylint: disable=no-value-for-parameter

@configured('train')
def train(
    max_steps: int = 100,
    batch_size_limit: int = 400,
    batch_limit_by_tokens: bool = True,
    report_interval_steps: int = 10,
    validation_interval_steps: int = 100,
    teacher_forcing: bool = True,
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
            if step >= max_steps:
                break

            def closure():
                optimizer.zero_grad()
                x_e = model.encode(batch[0])
                log_probs = model.decode(batch[1][:, :-1], x_e, teacher_forcing=teacher_forcing)
                loss = loss_function(log_probs, batch[1][:, 1:])
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)

            if step > 0 and step % report_interval_steps == 0:
                logger.info('#{}: loss={}'.format(step, loss.cpu() / batch.size(0) .item()))
            if step > 0 and step % validation_interval_steps == 0:
                total_item_count = 0
                total_validation_loss = 0
                with torch.no_grad():
                    for validation_batch in validation_dataset.iterate(
                        'cuda' if use_gpu else 'cpu',
                        batch_size_limit,
                        batch_limit_by_tokens
                    ):
                        x_e = model.encode(validation_batch[0])
                        log_probs = model.decode(validation_batch[1][:, :-1], x_e, teacher_forcing=teacher_forcing)
                        loss = loss_function(log_probs, validation_batch[1][:, 1:])
                        total_item_count += validation_batch.size(0)
                        total_validation_loss += loss.cpu().item()
                logger.info('#{}: validation_loss={}'.format(step, total_validation_loss / total_item_count))
