#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
from torch import nn
import torch.functional as F

class Highway(nn.Module):

    def __init__(self, e_word):
        super(Highway, self).__init__()
        self.proj = nn.Linear(e_word, e_word, bias=True)
        self.gate = nn.Linear(e_word, e_word, bias=True)
        self.dropout = nn.Dropout(.5)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, xconv_out):
        """Excpects inputs like (batch_size, e_word), outputs of same shape"""
        xproj = nn.ReLU()(self.proj(xconv_out))
        xgate = self.sigmoid(self.gate(xconv_out))
        xhighway = (xgate * xproj) + ((1 - xgate) * xconv_out)
        xword_emb = self.dropout(xhighway)
        return xword_emb
