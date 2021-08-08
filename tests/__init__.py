import unittest
import tempfile
import shutil
import torch

from nmt.visualization import make_sentence_graph, make_sentence_pair_graph

class TestMakingSentenceGraphs(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        return super().setUp()

    # def test_SentenceGraph(self):
    #     s = 'This is an absurd situation !'.split()
    #     n = len(s)
    #     A = 2 * torch.rand(n, n) - 1
    #     make_sentence_graph(s, A, f'{self.tmp}/sentence')
    #     shutil.copy(f'{self.tmp}/sentence.pdf', '/data/PhD/draw_graphs/sentence.pdf')
    #     self.assertTrue(True, 'WTF!')

    def test_SentencePairGraph(self):
        s0 = 'This is an absurd situation !'.split()
        s1 = 'In Yek Mogheiate Absurd Ast !'.split()
        n0 = len(s0)
        n1 = len(s1)
        A00 = 2 * torch.rand(n0, n0) - 1
        A01 = 2 * torch.rand(n0, n1) - 1
        A10 = 2 * torch.rand(n1, n0) - 1
        A11 = 2 * torch.rand(n1, n1) - 1
        make_sentence_pair_graph((s0, s1), (None, A01, None, A11), f'{self.tmp}/sentence_pair')
        shutil.copy(f'{self.tmp}/sentence_pair.pdf', '/data/PhD/draw_graphs/sentence_pair.pdf')
        self.assertTrue(True, 'WTF!')

    def tearDown(self):
        shutil.rmtree(self.tmp)
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()