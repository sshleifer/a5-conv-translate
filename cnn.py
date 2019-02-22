#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
from torch import nn
import torch.functional as F

class CNN(nn.Module):

    def __init__(self, echar, eword, max_word_len=21, k=5):
        super(CNN, self).__init__()
        self.k = k
        self.e_word = eword
        self.conv = nn.Conv1d(
            in_channels=echar, out_channels=eword, kernel_size=k,
            stride=1,
        )
        assert self.conv.weight.shape == (eword, echar, k)
        self.max_pool = nn.MaxPool1d(max_word_len - self.k + 1)

    def forward(self, x):
        """Inputs are (batch_size, e_char, m_word) outputs are (batch_size, e_word)"""
        (batch_size, e_char, m_word) = x.shape
        xconv = self.conv(x)
        xconv= nn.ReLU()(xconv)
        assert xconv.shape == (x.shape[0], self.e_word, m_word - self.k + 1)
        out = self.max_pool(xconv)
        squoze = out.squeeze(-1)  # (batch_size, e_word)
        return squoze # we want fx m_word -k + 1
