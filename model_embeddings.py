#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based char_embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size  # e_char
        self.echar = 50
        k = 5
        self.m_word = 21
        pad_token_idx = vocab.char2id['<pad>']
        self.char_embeddings = nn.Embedding(
            len(vocab.char2id), self.echar, padding_idx=pad_token_idx)
        self.dropout = nn.Dropout(.3)
        self.cnn = CNN(self.echar, embed_size) # inchannels, out_channels
        self.highway = Highway(embed_size)

    def forward(self, input):
        """
        Looks up character-based CNN char_embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, MAX_WORD_LENGTH) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based char_embeddings for each word of the sentences in the batch
        """
        sentence_length_1, batch_size_1, mword = input.shape
        char_emb = self.char_embeddings(input)
        (sentence_length, batch_size, max_word_length, e_char) = char_emb.shape
        assert max_word_length == mword == self.m_word
        assert max_word_length == 21
        assert sentence_length == sentence_length_1
        assert batch_size == batch_size_1
        assert e_char == self.echar
        # nervous about this reshape.
        # We want to make sure we are convolving over sentences not the ith word of every sentence
        # so maybe we should: Transpose to get the batch dim in front
        # Transpose(3,2) equivalent?
        x_reshaped = char_emb.permute(0, 1, 3, 2)
        # now combine axes 0 and 1 into 1 axis
        x_reshaped = x_reshaped.reshape(
            sentence_length * batch_size, e_char, max_word_length)

        # need inputs like (batch_size, e_char, m_word) for conv
        xconv_out = self.cnn.forward(x_reshaped)
        output = self.highway.forward(xconv_out)
        e_word = xconv_out.shape[-1]
        assert e_word == self.embed_size
        return self.dropout(output).reshape(sentence_length, batch_size, self.embed_size)
        # right place for dropout?


