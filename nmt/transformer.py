# coding: utf-8
"""
Transformer model for translation
"""

# pylint: disable=no-member
import torch
from torch import nn
from torch.nn import functional as F

from nmt.encoderdecoder import EncoderDecoder

class Transformer(EncoderDecoder):

    def __init__(self):
        super(Transformer, self).__init__()
        self.dummy = nn.Parameter(torch.FloatTensor([1, 2]))
    
    def encode(self, x):
        pass
    
    def decode_one_step(self, y_i, state):
        pass

    def decode(self, y, x_e):
        pass
