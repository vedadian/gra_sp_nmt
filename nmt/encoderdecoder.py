# coding: utf-8
"""
Base encoder decoder model
"""

# pylint: disable=no-member
from torch import nn

class EncoderDecoder(nn.Module):

    def __init__(self):
        super(EncoderDecoder, self).__init__()
    
    def encode(self, x):
        raise NotImplementedError('Abstract EncoderDecoder.encode function called')

    def decode_one_step(self, y, x_e, state):
        raise NotImplementedError('Abstract EncoderDecoder.decode_one_step function called')

    def decode(self, y, x_e, teacher_forcing):
        raise NotImplementedError('Abstract EncoderDecoder.decode function called')

    def forward(self, *args, **kwargs):
        raise RuntimeError('Direct calling of translation models is prohibited.')
