# coding: utf-8
"""
Tools for visualizing model artifcats and output
"""

import os
import math
gv = None
try:
    import graphviz
    gv = graphviz
except:
    pass
from torch import Tensor

from nmt.common import configured

class MultiheadAttentionType(object):
    SrcAtt = 0
    TgtAtt = 1
    CrsAtt = 2

class __Visualization(object):

    def __init__(self) -> None:
        super().__init__()
        self.enabled = False
        self.suspended = True

        self.source = None
        self.target = None

        self.registered_multihead_attention_modules = {}

    def wrap(self, f):
        def wrapped(*args, **kwargs):
            if gv is None or not self.enabled or self.suspended:
                return
            f(*args, **kwargs, source=self.source, target=self.target)
        return wrapped

    def __add_nodes(self, g, labels, r, j, n, add_separator_node):
        for i, label in enumerate(labels):
            a = 2 * math.pi * (i + j) / n
            x = -math.cos(a) * r
            y = math.sin(a) * r
            g.node(str(i + j), pos=f'{x},{y}!', label=label)
        if add_separator_node:
            a = 2 * math.pi * (len(labels) + j) / n
            x = -math.cos(a) * r
            y = math.sin(a) * r
            g.node(str(len(labels) + j), pos=f'{x},{y}!', label=label, style='invis')

    def __add_edges(self, g, s, e, A):
        if not self.enabled or self.suspended:
            return
        A = A.cpu().numpy()
        def get_color(weight):
            r = g = b = 16
            if weight < 0:
                b = int(min(128, -128 * weight))
            elif weight > 0:
                r = int(min(128, 128 * weight))
            return f'#{r:02x}{g:02x}{b:02x}'

        for i, e_n in enumerate(e):
            for j, s_n in enumerate(s):
                if A[i, j] < 0.1:
                    continue
                c = get_color(A[i, j])
                w = str(4 * abs(A[i, j]))
                g.edge(s_n, e_n, penwidth=w, color=c)

    def make_sentence_graph(
        self, 
        sentence: list,
        A: Tensor,
        output_path: str
    ):
        if gv is None or not self.enabled or self.suspended:
            return

        n = len(sentence)
        assert A.size() == (n, n)

        g = gv.Digraph(comment=' '.join(sentence), engine='neato')

        n = len(sentence)
        self.__add_nodes(g, sentence, n / 5.0, 0, n, add_separator_node=False)
        self.__add_edges(g,
            list(str(e) for e in range(n)),
            list(str(e) for e in range(n)),
            A / A.abs().max().item()
        )
        
        g.render(output_path)

    def make_sentence_pair_graph(
        self, 
        sentences: tuple,
        A: tuple,
        output_path: str
    ):
        if gv is None or not self.enabled or self.suspended:
            return

        Amax = max(Aij.abs().max().item() for Aij in A if Aij is not None)

        g = gv.Digraph(comment='(' + ','.join(' '.join(s) for s in sentences) + ')', engine='neato')

        n = sum(len(s) for s in sentences) + len(sentences)

        start_indexes = []

        r = n / 5.0
        j = 0
        for sentence in sentences:
            start_indexes.append(j)
            self.__add_nodes(g, sentence, r, j, n, add_separator_node=True)
            j += len(sentence) + 1

        m = len(sentences)
        for i in range(m):
            for j in range(m):
                if A[m * i + j] is None:
                    continue
                self.__add_edges(
                    g,
                    list(str(start_indexes[i] + e) for e in range(len(sentences[i]))),
                    list(str(start_indexes[j] + e) for e in range(len(sentences[j]))),
                    A[m * i + j] / Amax
                )
        
        g.render(output_path)

    def __produce_multihead_attention_artifacts(
        self,
        src_vocab,
        tgt_vocab,
        module_type,
        layer_index,
        attention,
        source,
        target
    ):
        
        @configured('model')
        def get_output_path(output_path: str):
            return output_path

        artifact_output_path = f'{get_output_path()}/artifacts'
        if not os.path.exists(artifact_output_path):
            os.makedirs(artifact_output_path, exist_ok=True)

        if module_type == MultiheadAttentionType.SrcAtt:
            for s, Ah in zip(source, attention):
                sentence = list(src_vocab.itos[e] for e in s)
                for h, A in enumerate(Ah):
                    visualization.make_sentence_graph(
                        sentence, A,
                        f'{artifact_output_path}/{" ".join(sentence)}_src_l{layer_index}_h{h}'
                    )
        elif module_type == MultiheadAttentionType.TgtAtt:
            for t, Ah in zip(target, attention):
                sentence = list(tgt_vocab.itos[e] for e in t)
                for h, A in enumerate(Ah):
                    self.make_sentence_graph(
                        sentence, A,
                        f'{artifact_output_path}/{" ".join(sentence)}_tgt_l{layer_index}_h{h}'
                    )
        elif module_type == MultiheadAttentionType.CrsAtt:
            for s, t, Ah in zip(source, target, attention):
                src_sentence = list(tgt_vocab.itos[e] for e in s)
                tgt_sentence = list(tgt_vocab.itos[e] for e in t)
                for h, A in enumerate(Ah):
                    self.make_sentence_pair_graph(
                        (src_sentence, tgt_sentence),
                        (None, None, A, None),
                        f'{artifact_output_path}/{" ".join(tgt_sentence)}_crs_l{layer_index}_h{h}'
                    )
        else:
            raise Exception('Unknown attention type.')

    def multihead_attention(self, module, attention):
        if gv is None or not self.enabled or self.suspended:
            return
        if module not in self.registered_multihead_attention_modules:
            return
        layer_index, attention_type, root = self.registered_multihead_attention_modules[module]
        self.__produce_multihead_attention_artifacts(
            root.src_vocab,
            root.tgt_vocab,
            attention_type,
            layer_index,
            attention,
            self.source,
            self.target
        )

visualization = __Visualization()