import numpy as np
import torch
import unittest

from highway import HighwayNetwork
from cnn import CharConv

class Tests(unittest.TestCase):
    def test_highway(self):
        e_word = 3
        ones = np.ones((8,e_word))
        x = torch.Tensor(ones)
        model = HighwayNetwork(e_word)
        out = model.forward(x)
        self.assertEqual(out.shape, (8,3))

        bad_input = torch.Tensor(np.ones((8,e_word+1)))
        with self.assertRaises(RuntimeError):
            model.forward(bad_input)


    def test_cnn(self):
        e_word = 30
        m_word = 4
        e_char = 20
        ones = np.ones((8, e_char, m_word))
        x = torch.Tensor(ones)
        cnn_model = CharConv(e_char, e_word)
        out = cnn_model.forward(x)
        self.assertEqual(out.shape, (8, e_word))



        pass
