# coding: utf-8
"""
Translation models
"""

import os
import math
import sys
from importlib import import_module

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nmt.dataset import Vocabulary
from nmt.common import configured
from nmt.encoderdecoder import EncoderDecoder

@configured('model')
def build_model(
    src_vocab: Vocabulary, tgt_vocab: Vocabulary, type: str = 'transformer'
):

    try:
        model_module = import_module(f'nmt.models.{type}')
    except:
        raise Exception(
            f'No `{type}` exists in defined seq2seq models.'
        )

    model = model_module.Model(src_vocab, tgt_vocab)
    model.source_code_path = f'{os.path.dirname(__file__)}/models/{type}.py'

    return model

@configured('model')
def get_model_short_description(type: str = 'transformer'):

    try:
        model_module = import_module(f'nmt.models.{type}')
    except:
        raise Exception(
            f'No `{type}` exists in defined seq2seq models.'
        )

    if hasattr(model_module.Model, 'short_description'):
        return model_module.Model.short_description()
    
    return None

@configured('model')
def get_model_source_code_path(type: str = 'transformer'):
    return f'{os.path.dirname(__file__)}/models/{type}.py'