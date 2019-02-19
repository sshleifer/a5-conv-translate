#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
sanity_check.py: sanity checks for assignment 5
Usage:
    sanity_check.py 1e
    sanity_check.py 1f
    sanity_check.py 1j
    sanity_check.py 2a
    sanity_check.py 2b
    sanity_check.py 2c
    sanity_check.py 2d
    sanity_check.py all
"""
import json
import pickle
import sys

import numpy as np
import torch
import torch.nn.utils
from docopt import docopt

from char_decoder import CharDecoder
from cnn import CNN
from highway import Highway
from nmt_model import NMT
from utils import pad_sents_char
from vocab import Vocab, VocabEntry

import unittest

#----------
# CONSTANTS
#----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0
# Seed the Random Number Generators
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed * 13 // 7)

vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

# Create NMT Model
model = NMT(
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    dropout_rate=DROPOUT_RATE,
    vocab=vocab
)




class DummyVocab():
    def __init__(self):
        self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]

char_vocab = DummyVocab()

# Initialize CharDecoder
decoder = CharDecoder(
    hidden_size=HIDDEN_SIZE,
    char_embedding_size=EMBED_SIZE,
    target_vocab=char_vocab)

class TestEverything(unittest.TestCase):

    def setUp(cls):

        # Initialize CharDecoder
        cls.decoder = CharDecoder(
            hidden_size=HIDDEN_SIZE,
            char_embedding_size=EMBED_SIZE,
            target_vocab=char_vocab)
        cls.vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

        # cl NMT Model
        cls.model = NMT(
            embed_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            dropout_rate=DROPOUT_RATE,
            vocab=vocab
        )
        cls.char_vocab = DummyVocab()

    def test_question_1e_sanity_check(self):
        """ Sanity check for words2charindices function.
        """
        vocab = VocabEntry()


        sentences = [["a", "b", "c?"], ["~d~", "c", "b", "a"]]
        small_ind = vocab.words2charindices(sentences)
        small_ind_gold = [[[1, 30, 2], [1, 31, 2], [1, 32, 70, 2]], [[1, 85, 33, 85, 2], [1, 32, 2], [1, 31, 2], [1, 30, 2]]]
        assert(small_ind == small_ind_gold), \
            "small test resulted in indices list {:}, expected {:}".format(small_ind, small_ind_gold)

        # print('Running test on single sentence')
        # sentence = ["right", "arcs", "only"]
        # single_ind = vocab.words2charindices(sentence)
        # single_ind_gold = [[[1, 47, 2], [1, 38, 2], [1, 36, 2], [1, 37, 2], [1, 49, 2]], [[1, 30, 2], [1, 47, 2], [1, 32, 2], [1, 48, 2]], [[1, 44, 2], [1, 43, 2], [1, 41, 2], [1, 54, 2]]]
        # assert(single_ind == single_ind_gold), \
        #     "single sentence test resulted in indices list {:}, expected {:}".format(single_ind, single_ind_gold)

        print('Running test on large list of sentences')
        tgt_sents = [['<s>', "Let's", 'start', 'by', 'thinking', 'about', 'the', 'member', 'countries', 'of', 'the', 'OECD,', 'or', 'the', 'Organization', 'of', 'Economic', 'Cooperation', 'and', 'Development.', '</s>'], ['<s>', 'In', 'the', 'case', 'of', 'gun', 'control,', 'we', 'really', 'underestimated', 'our', 'opponents.', '</s>'], ['<s>', 'Let', 'me', 'share', 'with', 'those', 'of', 'you', 'here', 'in', 'the', 'first', 'row.', '</s>'], ['<s>', 'It', 'suggests', 'that', 'we', 'care', 'about', 'the', 'fight,', 'about', 'the', 'challenge.', '</s>'], ['<s>', 'A', 'lot', 'of', 'numbers', 'there.', 'A', 'lot', 'of', 'numbers.', '</s>']]
        tgt_ind = vocab.words2charindices(tgt_sents)
        tgt_ind_gold = pickle.load(open('./sanity_check_en_es_data/1e_tgt.pkl', 'rb'))
        assert(tgt_ind == tgt_ind_gold), "target vocab test resulted in indices list {:}, expected {:}".format(tgt_ind, tgt_ind_gold)


    def test_question_1f_sanity_check(self):
        """ Sanity check for pad_sents_char() function.
        """
        vocab = VocabEntry()

        print("Running test on a list of sentences")
        sentences = [['Human:', 'What', 'do', 'we', 'want?'], ['Computer:', 'Natural', 'language', 'processing!'], ['Human:', 'When', 'do', 'we', 'want', 'it?'], ['Computer:', 'When', 'do', 'we', 'want', 'what?']]
        word_ids = vocab.words2charindices(sentences)

        padded_sentences = pad_sents_char(word_ids, 0)
        gold_padded_sentences = torch.load('./sanity_check_en_es_data/gold_padded_sentences.pkl')
        assert len(gold_padded_sentences) == len(padded_sentences)
        for expected, got in zip(gold_padded_sentences, padded_sentences):
            if got != expected:
                raise AssertionError(f'got {got}: expected: {expected}')
        assert padded_sentences == gold_padded_sentences, "Sentence padding is incorrect: it should be:\n {} but is:\n{}".format(gold_padded_sentences, padded_sentences)



    def test_question_1j_sanity_check(self):
        """ Sanity check for model_embeddings.py
            basic shape check
        """

        sentence_length = 10
        max_word_length = 21
        inpt = torch.zeros(sentence_length, BATCH_SIZE, max_word_length, dtype=torch.long)
        ME_source = self.model.model_embeddings_source
        output = ME_source.forward(inpt)
        output_expected_size = [sentence_length, BATCH_SIZE, EMBED_SIZE]
        assert(list(output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))

    def test_question_2a_sanity_check(self):
        """ Sanity check for CharDecoder.__init__()
            basic shape check
        """
        decoder = self.decoder
        char_vocab = self.char_vocab
        assert(decoder.charDecoder.input_size == EMBED_SIZE), "Input dimension is incorrect:\n it should be {} but is: {}".format(EMBED_SIZE, decoder.charDecoder.input_size)
        assert(decoder.charDecoder.hidden_size == HIDDEN_SIZE), "Hidden dimension is incorrect:\n it should be {} but is: {}".format(HIDDEN_SIZE, decoder.charDecoder.hidden_size)
        assert(decoder.char_output_projection.in_features == HIDDEN_SIZE), "Input dimension is incorrect:\n it should be {} but is: {}".format(HIDDEN_SIZE, decoder.char_output_projection.in_features)
        assert(decoder.char_output_projection.out_features == len(char_vocab.char2id)), "Output dimension is incorrect:\n it should be {} but is: {}".format(len(char_vocab.char2id), decoder.char_output_projection.out_features)
        assert(decoder.decoderCharEmb.num_embeddings == len(char_vocab.char2id)), "Number of char_embeddings is incorrect:\n it should be {} but is: {}".format(len(char_vocab.char2id), decoder.decoderCharEmb.num_embeddings)
        assert(decoder.decoderCharEmb.embedding_dim == EMBED_SIZE), "Embedding dimension is incorrect:\n it should be {} but is: {}".format(EMBED_SIZE, decoder.decoderCharEmb.embedding_dim)

    def test_question_2b_sanity_check(self):
        """ Sanity check for CharDecoder.forward()
            basic shape check
        """
        decoder = self.decoder
        char_vocab = self.char_vocab
        sequence_length = 4
        inpt = torch.zeros(sequence_length, BATCH_SIZE, dtype=torch.long)
        logits, (dec_hidden1, dec_hidden2) = decoder.forward(inpt)
        logits_expected_size = [sequence_length, BATCH_SIZE, len(char_vocab.char2id)]
        dec_hidden_expected_size = [1, BATCH_SIZE, HIDDEN_SIZE]
        assert(list(logits.size()) == logits_expected_size), "Logits shape is incorrect:\n it should be {} but is:\n{}".format(logits_expected_size, list(logits.size()))
        assert(list(dec_hidden1.size()) == dec_hidden_expected_size), "Decoder hidden state shape is incorrect:\n it should be {} but is: {}".format(dec_hidden_expected_size, list(dec_hidden1.size()))
        assert(list(dec_hidden2.size()) == dec_hidden_expected_size), "Decoder hidden state shape is incorrect:\n it should be {} but is: {}".format(dec_hidden_expected_size, list(dec_hidden2.size()))

    def test_train_forward(self):
        sequence_length = 4
        inpt = torch.zeros(sequence_length, BATCH_SIZE, dtype=torch.long)
        loss = decoder.train_forward(inpt)
        self.assertGreaterEqual(loss, 0)

    def test_question_2c_sanity_check_train_fwd(self):
        """ Sanity check for CharDecoder.train_forward()
            basic shape check
        """
        decoder = self.decoder
        sequence_length = 6
        inpt = np.ones((sequence_length, BATCH_SIZE,))
        first_element = np.array([1, 6, 12, 12, 2, 0])
        inpt[:,1] = first_element
        inpt = torch.tensor(inpt, dtype=torch.long)
        #inpt = torch.zeros(sequence_length, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.long)
        #inpt = torch.normal(mean=inpt, std=inpt+1)


        loss = decoder.train_forward(inpt)
        # "Loss should be a scalar but its shape is: {}".format(list(loss.size()))
        self.assertEqual(list(loss.size()), [])
        preds = decoder.forward(inpt)
        #import ipdb; ipdb.set_trace()
        self.assertGreater(loss, 0)
        #self.assertGreaterEqual(10, loss)


    def test_decoder_loss_fn(self):

        preds = torch.tensor([[.25, .25, .25, .25], [.25, .25, .25, .25]])
        targets = torch.tensor([1, 1])
        loss1 = self.decoder.ce_loss_fn(preds, targets)
        print(f'loss1:{loss1:.4f}')

    def test_ignore_index_in_loss(self):

        preds = torch.tensor([[0., 1., 0., 0.], [.25, .25, .25, .25]])
        targets = torch.tensor([1, 0])
        loss2 = self.decoder.ce_loss_fn(preds, targets)
        print(f'loss2:{loss2:.4f}')
        loss3 = self.decoder.ce_loss_fn(preds[:1], targets[:1]) # doesnt consider 0 example
        self.assertEqual(loss3, loss2)
        #self.assertEqual(loss2, 0)


    def test_reduction_sum_works(self):
        preds = torch.tensor([[0., 1., 0., 0.], [.25, .25, .25, .25]])
        targets = torch.tensor([1, 1])
        loss_total = self.decoder.ce_loss_fn(preds, targets)
        loss_a = self.decoder.ce_loss_fn(preds[1:], targets[1:])
        loss_b = self.decoder.ce_loss_fn(preds[:1], targets[:1])
        self.assertEqual(loss_total, loss_a + loss_b)

    def test_question_2d_sanity_check(self):
        """ Sanity check for CharDecoder.decode_greedy()
            basic shape check
        """

        decoder = self.decoder
        sequence_length = 4
        inpt = torch.zeros(1, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float)
        inpt = torch.normal(mean=inpt, std=inpt+1)
        initialStates = (inpt, inpt)
        device = decoder.char_output_projection.weight.device
        decodedWords = decoder.decode_greedy(initialStates, device)
        assert(len(decodedWords) == BATCH_SIZE), "Length of decodedWords should be {} but is: {}".format(BATCH_SIZE, len(decodedWords))


MAX_WORD_LEN = 21


class Tests(unittest.TestCase):
    def test_highway(self):
        e_word = 3
        ones = np.ones((8,e_word))
        x = torch.Tensor(ones)
        model = Highway(e_word)
        out = model.forward(x)
        self.assertEqual(out.shape, (8,3))

        bad_input = torch.Tensor(np.ones((8,e_word+1)))
        with self.assertRaises(RuntimeError):
            model.forward(bad_input)

    def test_cnn(self):
        e_word = 30
        m_word = MAX_WORD_LEN
        e_char = 50
        ones = np.ones((8, e_char, m_word))
        x = torch.Tensor(ones)
        cnn_model = CNN(e_char, e_word, MAX_WORD_LEN)
        self.assertEqual(cnn_model.conv.weight.shape, (e_word, e_char, 5))
        out = cnn_model.forward(x)
        self.assertEqual(out.shape, (8, e_word))

    def test_cnn_bigger_eword(self):
        e_word = 300
        m_word = MAX_WORD_LEN
        e_char = 50
        ones = np.ones((8, e_char, m_word))
        x = torch.Tensor(ones)
        cnn_model = CNN(e_char, e_word, MAX_WORD_LEN)
        cnn_model.conv.weight.shape
        out = cnn_model.forward(x)
        self.assertEqual(out.shape, (8, e_word))

    def test_max_pool_on_last_dim(self):
        mock_arr = np.array(range(25)).reshape(1, 5, 5)
        tensor = torch.Tensor(mock_arr)
        mpool = torch.squeeze(torch.nn.MaxPool1d(kernel_size=5)(tensor))
        np.testing.assert_array_equal(mpool.numpy(),
                                      np.array([4, 9, 14, 19, 24]))

    def test_cnn_k_too_big(self):
        e_word = 30
        m_word = 4
        e_char = 20
        ones = np.ones((8, e_char, m_word))
        x = torch.Tensor(ones)
        cnn_model = CNN(e_char, e_word, k=m_word + 1)
        with self.assertRaises(RuntimeError):
            cnn_model.forward(x)

    def test_1j(self):
        pass
