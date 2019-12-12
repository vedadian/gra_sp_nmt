# coding: utf-8
"""
Implementation of a simple yet fast dataset for neural machine translation
"""

import pickle
from typing import List, Callable, IO, Union, Any

from constants import UNK_TOKEN, EOS_TOKEN, SOS_TOKEN, PAD_TOKEN

class Vocabulary(object):

    def __init__(self):
        self.stoi = {}
        self.itos = []
    
        for token in [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
            self.append_and_get_index(token)

    def append_and_get_index(self, token: str):
        if not token in self.stoi:
            index = len(self.itos)
            self.stoi[token] = index
            self.itos.append(token)
            return index
        return self.stoi[token]
    
    def get_index(self, token: str):
        if not token in self.stoi:
            return 0 # self.stoi[UNK_TOKEN]
        return self.stoi[token]

def default_normalizer(sentence: str):
    pass

def default_tokenizer(sentence: str):
    pass

NORMALIZERS = {
    'default': default_normalizer
}

TOKENIZERS = {
    'default': default_tokenizer
}

class Field(object):

    def __init__(
        self,
        lowercase: bool = True,
        vocabulary: Vocabulary = None,
        normalize: str = None,
        tokenize: str = None
    ):

        self.lowercase = lowercase
        self.has_external_vocabulary = vocabulary is not None
        self.vocabulary = vocabulary if self.has_external_vocabulary else Vocabulary()
        self.normalizer_name = 'default' if normalize is None else normalize
        self.normalize = NORMALIZERS[self.normalizer_name]
        self.tokenizer_name = 'default' if tokenize is None else tokenize
        self.tokenize = TOKENIZERS[self.tokenizer_name]

    def process(self, sentence: str):
        
        if self.lowercase:
            result = sentence.lower()
        else:
            result = sentence
        result = self.tokenize(self.normalize(result))
        if self.has_external_vocabulary:
            result = [self.vocabulary.get_index(e) for e in result]
        else:
            result = [self.vocabulary.append_and_get_index(e) for e in result]
        return result
    
    def __getstate__(self):
        if self.has_external_vocabulary:
            return [self.lowercase, True, self.normalizer_name, self.tokenizer_name]
        else:
            return [self.lowercase, False, self.vocabulary, self.normalizer_name, self.tokenizer_name]

    def __setstate__(self, state):
        if state[1]:
            self.lowercase = state[0]
            self.has_external_vocabulary = True
            self.vocabulary = None
            self.normalizer_name = state[2]
            self.normalize = NORMALIZERS[state[2]]
            self.tokenizer_name = state[3]
            self.tokenize = TOKENIZERS[state[3]]
        else:
            self.lowercase = state[0]
            self.has_external_vocabulary = False
            self.vocabulary = state[2]
            self.normalizer_name = state[3]
            self.normalize = NORMALIZERS[state[3]]
            self.tokenizer_name = state[4]
            self.tokenize = TOKENIZERS[state[4]]

class Corpora(object):

    def __init__(self, fields: List[Field]):
        self.fields = fields
        self.samples = []
    
    def append(self, sentences: List[str]):
        assert len(self.fields) == len(sentences), \
               'Number of sentences and fields do not match. ({} != {})'.format(
                   len(sentences), len(self.fields))
        
        sample = [f.process(s) for f, s in zip(self.fields, sentences)]

        self.samples.append(sample)
    
    def __getstate__(self):
        external_vocabularies = list(set(f.vocabulary for f in self.fields if f.has_external_vocab))
        field_vocabulary_index = [
            (external_vocabularies.index(f.vocabulary) if f.has_external_vocab else -1)
            for f in self.fields
        ]
        return [self.samples, self.fields, external_vocabularies, field_vocabulary_index]

    def __setstate__(self, state):
        self.samples = state[0]
        self.fields = state[1]
        for field, vocabulary_index in zip(state[1], state[3]):
            if vocabulary_index < 0:
                continue
            field.vocabulary = state[2][vocabulary_index]

    def save(self, output: Union[str, IO[Any]]):
        if isinstance(output, str):
            stream = open(output, 'wb')
        else:
            stream = output
        pickle.dump(self, stream)
    
    @staticmethod
    def load(input: Union[str, IO[Any]]):
        if isinstance(input, str):
            stream = open(input, 'rb')
        else:
            stream = input
        corpora = pickle.load(stream)
        if not isinstance(corpora, Corpora):
            raise Exception('Corpora::load: Input to load function is not an instance of `Corpora`.')
        return corpora

