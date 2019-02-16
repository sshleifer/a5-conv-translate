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
        pad_token_idx = vocab['<pad>']
        self.char_embeddings = nn.Embedding(len(vocab), embed_size, padding_idx=pad_token_idx)
        self.dropout = nn.Dropout(.3)
        self.cnn = CNN(embed_size, embed_size)
        self.highway = Highway(embed_size)

    def forward(self, input):
        """
        Looks up character-based CNN char_embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, MAX_WORD_LENGTH) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based char_embeddings for each word of the sentences in the batch
        """
        sentence_length, batch_size, mword = input.shape
        print(f'input: {input.shape}')
        char_emb = self.char_embeddings(input)
        #char_emb_tp = char_emb.transpose(0, 1)
        # import ipdb; ipdb.set_trace()

        (sentence_length, batch_size, max_word_length, e_char) = char_emb.shape
        char_emb_reshape = char_emb.reshape(sentence_length * batch_size, max_word_length,
                                            e_char)
        print(f'char_emb: {char_emb.shape}')#REMOVE ME
        print(f'char_emb reshape: {char_emb_reshape.shape}')  # REMOVE ME

        # have (
        # need inputs like (batch_size, e_char, m_word) for conv

        xconv_out = self.cnn.forward(char_emb_reshape.transpose(1,2))
        #xconv_reshaped = xconv_out.reshape()
        print(f'xconv_out: {xconv_out.shape}')
        output = self.highway.forward(xconv_out)
        e_word = xconv_out.shape[-1]
        print(f'inferring e_word={e_word}')
        output = output.reshape(sentence_length, batch_size, e_word) # ideally
        print(f'ModelEmb  output: {output.shape} desired {(sentence_length, batch_size, e_word)}')

        assert output.shape == (sentence_length, batch_size, e_word)
        return self.dropout(output)
        # right place for dropout?

