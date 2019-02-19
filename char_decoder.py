#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import numpy as np
import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character char_embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """

        super(CharDecoder, self).__init__()
        char_vocab_size = len(target_vocab.char2id)
        self.n_chars = char_vocab_size

        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size,
                                   bidirectional=False,
                                   bias=True)
        self.char_output_projection = nn.Linear(hidden_size, char_vocab_size, bias=True)
        self.decoderCharEmb = nn.Embedding(char_vocab_size, char_embedding_size,
                                           padding_idx=0)
        self.target_vocab = target_vocab

        self.class_weights = torch.Tensor(np.ones(char_vocab_size))
        self.class_weights[0] = 0  # ignore pad chars
        self.ce_loss_fn = nn.CrossEntropyLoss(#weight=self.class_weights,
                                              reduction='sum',
                                              ignore_index=0)
        self.softmax = nn.Softmax(dim=-1)


        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character char_embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse char_embeddings created in Part 1 of this assignment.


    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        embedded = self.decoderCharEmb(input)
        ht, ct = self.charDecoder(embedded, dec_hidden)
        st = self.char_output_projection(ht)
        return st, ct

    def old_train_forward(self,char_sequence, dec_hidden=None):
        st, ct = self.forward(char_sequence, dec_hidden)
        pt = self.softmax(st)  # do we need to do softmax?
        batch_size = char_sequence.shape[1]
        loss_char_dec = 0

        for b in range(batch_size):  # for every word in the batch
            true_chars = char_sequence[:, b]
            preds = pt[:, b]
            loss_char_dec += self.ce_loss_fn(preds, true_chars)

        return loss_char_dec

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch).
            Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder.
            A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy
        losses of all the words in the batch.
        """
        # what if different than length expected by forward?

        target = char_sequence[1:]
        xinput = char_sequence[:-1]
        st, ct = self.forward(xinput, dec_hidden)  # st -> (length, batch, self.vocab_size)
        st_perm = st.contiguous().view(-1, self.n_chars)

        reshaped_targ = target.contiguous().view(-1)


        return self.ce_loss_fn(st_perm, reshaped_targ)

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###    - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout
        #  (e.g., <START>,m,u,s,i,c,<END>).

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """
        batch_size = initialStates[0].shape[1]
        start_seed = np.array([self.target_vocab.start_of_word] * batch_size).reshape(1, batch_size)
        current_char = torch.tensor(start_seed, dtype=torch.long).to(device)
        output_word = []
        c = initialStates
        for t in range(max_length):
            st, c = self.forward(current_char, c)
            pt = self.softmax(st)
            # set pad idx to negative inf

            current_char = torch.argmax(pt, dim=-1)
            output_word.append([self.target_vocab.id2char[x]
                                for x in current_char.detach().numpy()[0]])
        words = self.clip_from_end_char(output_word)
        return [''.join(w) for w in words]

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.


        ### END YOUR CODE

    def clip_from_end_char(self, output_word):
        out = np.array(output_word).T
        words = []
        for i in out:
            cur_word = []
            for char in i:
                if self.target_vocab.char2id[char] == self.target_vocab.end_of_word:
                    break
                cur_word.append(char)
            words.append(cur_word)
        return words

