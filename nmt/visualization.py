# coding: utf-8
"""
Tools for visualizing model artifcats and output
"""

import math
gv = None
try:
    import graphviz
    gv = graphviz
except:
    pass
from torch import Tensor

def __add_nodes(g, labels, r, j, n, add_separator_node):
    for i, label in enumerate(labels):
        a = 2 * math.pi * (i + j) / n
        x = -math.cos(a) * r
        y = math.sin(a) * r
        g.node(str(i + j), pos=f'{x},{y}!', label=label)
    if add_separator_node:
        a = 2 * math.pi * (len(labels) + j) / n
        x = -math.cos(a) * r
        y = math.sin(a) * r
        g.node(str(len(labels) + j), pos=f'{x},{y}!', label=label)
        
def __add_edges(g, s, e, A):

    A = A.cpu()

    def get_color(weight):
        r = g = b = 16
        if weight < 0:
            b = int(min(128, -128 * weight))
        elif weight > 0:
            r = int(min(128, 128 * weight))
        return f'#{r:02x}{g:02x}{b:02x}'

    for i, s_n in enumerate(s):
        for j, e_n in enumerate(e):
            if A[i, j] < 0.1:
                continue
            c = get_color(A[i, j])
            w = str(4 * abs(A[i, j]))
            g.edge(s_n, e_n, penwidth=w, color=c)

def make_sentence_graph(
    sentence: list[str],
    A: Tensor,
    output_path: str
):
    if gv is None:
        return

    n = len(sentence)
    assert A.size() == (n, n)

    g = gv.Digraph(comment=' '.join(sentence))

    n = len(sentence)
    __add_nodes(g, sentence, n / 5.0, 0, n)
    __add_edges(g, str(e) for e in range(n), str(e) for e in range(n), A / A.abs().max().item())
    
    g.render(output_path)

def make_sentence_pair_graph(
    sentences: tuple[list[str]],
    A: tuple(Tensor),
    output_path: str
):
    if gv is None:
        return

    def get_color(weight):
        r = g = b = 16
        if weight < 0:
            b = int(min(128, -128 * weight))
        elif weight > 0:
            r = int(min(128, 128 * weight))
        return f'#{r:02x}{g:02x}{b:02x}'

    Amax = max(Aij.abs().max().item() for Aij in A)

    g = gv.Digraph(comment='(' + ','.join(' '.join(s) for s in sentences) + ')')

    n = sum(len(s) for s in sentences) + len(sentences)

    start_indexes = []

    r = n / 5.0
    j = 0
    for sentence in sentences:
        start_indexes.append(j)
        __add_nodes(g, sentence, r, j, n, add_separator_node=True)
        j += len(sentence) + 1

    m = len(sentences)
    for i in range(m):
        for j in range(m):
            if A[m * i + j] is None:
                continue
            __add_edges(
                g,
                str(start_indexes[i] * e) for e in range(len(sentences[i])),
                str(start_indexes[j] * e) for e in range(len(sentences[j])),
                A[m * i + j] / Amax
            )
    
    g.render(output_path)