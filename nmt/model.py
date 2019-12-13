# coding: utf-8
"""
Translation models
"""

from nmt.common import configured
from nmt.transformer import Transformer

MODELS = {
    'transformer': Transformer
}

@configured('model')
def build_model(type: str = 'transformer'):
    
    if type not in MODELS:
        raise Exception('No `{}` exists in defined seq2seq models.'.format(type))

    return MODELS[type]()