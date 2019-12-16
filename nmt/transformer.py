# coding: utf-8
"""
Transformer model for translation
"""

import math

# pylint: disable=no-member,arguments-differ
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from nmt.dataset import Vocabulary
from nmt.common import configured, epsilon
from nmt.encoderdecoder import EncoderDecoder


class MultiheadAttention(nn.Module):

    def __init__(self, num_heads: int, embedding_size: int, dropout: float = 0.1, causual: bool = False):
        nn.Module.__init__(self)
        
        assert embedding_size % num_heads == 0
        
        self.head_size = embedding_size // num_heads
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.causual = causual

        self.k_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.v_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.q_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.o_linear = nn.Linear(embedding_size, embedding_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, k: Tensor, v: Tensor, q:Tensor):

        b = v.size(0)
        v_len = v.size(-2)
        q_len = q.size(-2)


        k = self.k_linear(k)
        v = self.v_linear(v)
        q = self.q_linear(q)

        k = k.view(-1, v_len, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(-1, v_len, self.num_heads, self.head_size).transpose(1, 2)
        q = q.view(-1, v_len, self.num_heads, self.head_size).transpose(1, 2)

        q = q / math.sqrt(self.head_size)

        scores = torch.matmul(q, k.transpose(2, 3))

        if self.causual:
            mask = torch.tril(torch.ones(q_len, v_len))
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand_as(b, self.num_heads, q_len, v_len)
            scores = scores.masked_fill(mask != 1, -math.inf)
        
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        result = torch.matmul(attention, v) \
            .transpose(1, 2).contiguous() \
            .view(-1, q_len, self.embedding_size)

        result = self.o_linear(result)

        return result

def create_positional_encoding_vector(embedding_size: int, max_length: int):
    pe = torch.zeros(max_length, embedding_size)
    p = torch.arange(0, max_length, dtype=torch.float)
    s = torch.exp(-torch.arange(0, embedding_size, 2, dtype=torch.float()) * (math.log(1e4) / embedding_size))
    pe[:, 0::2] = torch.sin(p * s)
    pe[:, 1::2] = torch.cos(p * s)
    return pe

class TransformerEncoderLayer(nn.Module):

    def __init__(self, num_heads: int, embedding_size: int, ffn_size: int, dropout: float):
        nn.Module.__init__(self)
        self.mha = MultiheadAttention(num_heads, embedding_size, dropout=dropout, causual=False)
        self.ln0 = nn.LayerNorm(embedding_size, eps=epsilon)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, ffn_size),
            nn.ReLU(),
            nn.Linear(ffn_size, embedding_size),
        )
        self.ln1 = nn.LayerNorm(embedding_size, eps=epsilon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.mha(x, x, x)
        h = self.dropout(self.ln0(h)) + x
        h = self.ffn(h)
        h = self.dropout(self.ln1(h)) + x
        return h

class TransformerDecoderLayer(nn.Module):

    def __init__(self, num_heads: int, embedding_size: int, ffn_size: int, dropout: float):
        nn.Module.__init__(self)
        self.mha0 = MultiheadAttention(num_heads, embedding_size, dropout=dropout, causual=True)
        self.ln0 = nn.LayerNorm(embedding_size, eps=epsilon)
        self.mha1 = MultiheadAttention(num_heads, embedding_size, dropout=dropout, causual=False)
        self.ln1 = nn.LayerNorm(embedding_size, eps=epsilon)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, ffn_size),
            nn.ReLU(),
            nn.Linear(ffn_size, embedding_size),
        )
        self.ln2 = nn.LayerNorm(embedding_size, eps=epsilon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, x_e):
        h = self.mha0(y, y, y)
        h = self.dropout(self.ln0(h)) + y
        h = self.mha1(x_e, x_e, h) + y
        h = self.ffn(h)
        h = self.dropout(self.ln1(h)) + y
        return h

class Transformer(EncoderDecoder):

    def __init__(
        self,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        n_encoders: int,
        n_decoders: int,
        num_heads: int,
        embedding_size: int,
        ffn_size: int,
        dropout: float,
        emb_dropout: float,
        ):

        EncoderDecoder.__init__(self)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.pe = create_positional_encoding_vector(embedding_size, 512)
        self.encoder_layers = [
            TransformerEncoderLayer(num_heads, embedding_size, ffn_size, dropout)
            for _ in range(n_encoders)
        ]
        self.src_embed = nn.Embedding(len(src_vocab.itos), embedding_size)
        if src_vocab == tgt_vocab:
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = nn.Embedding(len(tgt_vocab.itos), embedding_size)
        self.decoder_layers = [
            TransformerEncoderLayer(num_heads, embedding_size, ffn_size, dropout)
            for _ in range(n_encoders - 1)
        ]
        self.emb_dropout = nn.Dropout(emb_dropout)

    def encode(self, x: Tensor):
        x = self.src_embed(x)
        x = x + self.pe[:x.size(1), :].unsqueeze(0).expand_as(x)
        x = self.emb_dropout(x)
        for encode in self.encoder_layers:
            x = encode(x)
        return x

    def decode_one_step(self, y: Tensor, x_e: Tensor, state: Tensor):
        y = self.decode(y, x_e, True)
        return y[:, -1, :], None

    def decode(self, y: Tensor, x_e: Tensor, teacher_forcing: bool):
        y = y + self.pe[:y.size(1), :].unsqueeze(0).expand_as(y)
        y = self.emb_dropout(y)
        for decode in self.decoder_layers:
            y = decode(y, x_e)
        return y
