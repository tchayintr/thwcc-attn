from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.nn.util import get_mask_from_sequence_lengths
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import sys

import constants
import models.util
from models.util import ModelUsage
from models.common import BiaffineCombination, MLP


class RNNTagger(nn.Module):
    def __init__(self,
                 n_vocab,
                 unigram_embed_size,
                 n_bigrams,
                 bigram_embed_size,
                 rnn_unit_type,
                 rnn_bidirection,
                 rnn_batch_first,
                 rnn_n_layers,
                 rnn_hidden_size,
                 mlp_n_layers,
                 mlp_hidden_size,
                 n_labels,
                 use_crf=True,
                 feat_size=0,
                 mlp_additional_hidden_size=0,
                 embed_dropout=0.0,
                 rnn_dropout=0.0,
                 mlp_dropout=0.0,
                 pretrained_unigram_embed_size=0,
                 pretrained_bigram_embed_size=0,
                 pretrained_embed_usage=ModelUsage.NONE):
        super(RNNTagger, self).__init__()
        self.n_vocab = n_vocab
        self.unigram_embed_size = unigram_embed_size
        self.n_bigrams = n_bigrams
        self.bigram_embed_size = bigram_embed_size

        self.rnn_unit_type = rnn_unit_type
        self.rnn_bidirection = rnn_bidirection
        self.rnn_batch_first = rnn_batch_first
        self.rnn_n_layers = rnn_n_layers
        self.rnn_hidden_size = rnn_hidden_size

        self.mlp_n_layers = mlp_n_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.n_labels = n_labels
        self.use_crf = use_crf
        self.mlp_additional_hidden_size = mlp_additional_hidden_size

        self.embed_dropout = embed_dropout
        self.rnn_dropout = rnn_dropout
        self.mlp_dropout = mlp_dropout

        self.pretrained_unigram_embed_size = pretrained_unigram_embed_size
        self.pretrained_bigram_embed_size = pretrained_bigram_embed_size
        self.pretrained_embed_usage = pretrained_embed_usage

        self.unigram_embed = None
        self.bigram_embed = None
        self.pretrained_unigram_embed = None
        self.pretrained_bigram_embed = None
        self.rnn = None
        self.mlp = None
        self.crf = None
        self.cross_entropy_loss = None

        print('### Parameters', file=sys.stderr)

        # embeddings layer(s)

        print('# Embedding dropout ratio={}'.format(self.embed_dropout),
              file=sys.stderr)
        self.unigram_embed, self.pretrained_unigram_embed = models.util.construct_embeddings(
            n_vocab, unigram_embed_size, pretrained_unigram_embed_size,
            pretrained_embed_usage)
        if self.pretrained_embed_usage != ModelUsage.NONE:
            print('# Pretrained embedding usage: {}'.format(
                self.pretrained_embed_usage),
                  file=sys.stderr)
        print('# Unigram embedding matrix: W={}'.format(
            self.unigram_embed.weight.shape),
              file=sys.stderr)
        embed_size = self.unigram_embed.weight.shape[1]
        if self.pretrained_unigram_embed is not None:
            if self.pretrained_embed_usage == ModelUsage.CONCAT:
                embed_size += self.pretrained_unigram_embed_size
                print('# Pretrained unigram embedding matrix: W={}'.format(
                    self.pretrained_unigram_embed.weight.shape),
                      file=sys.stderr)

        if n_bigrams > 0 and bigram_embed_size > 0:
            self.bigram_embed, self.pretrained_bigram_embed = models.util.construct_embeddings(
                n_bigrams, bigram_embed_size, pretrained_bigram_embed_size,
                pretrained_embed_usage)
            if self.pretrained_embed_usage != ModelUsage.NONE:
                print('# Pretrained embedding usage: {}'.format(
                    self.pretrained_embed_usage),
                      file=sys.stderr)
            print('# Bigram embedding matrix: W={}'.format(
                self.bigram_embed.weight.shape),
                  file=sys.stderr)
            embed_size += self.bigram_embed.weight.shape[1]
            if self.pretrained_bigram_embed is not None:
                if self.pretrained_embed_usage == ModelUsage.CONCAT:
                    embed_size += self.pretrained_bigram_embed.weight.shape[1]
                print('# Pretrained bigram embedding matrix: W={}'.format(
                    self.pretrained_bigram_embed.weight.shape),
                      file=sys.stderr)

        # recurrent layers

        self.rnn_unit_type = rnn_unit_type
        self.rnn = models.util.construct_RNN(unit_type=rnn_unit_type,
                                             embed_size=embed_size,
                                             hidden_size=rnn_hidden_size,
                                             n_layers=rnn_n_layers,
                                             batch_first=rnn_batch_first,
                                             dropout=rnn_dropout,
                                             bidirectional=rnn_bidirection)
        rnn_output_size = rnn_hidden_size * (2 if rnn_bidirection else 1)

        # MLP

        print('# MLP', file=sys.stderr)
        mlp_in = rnn_output_size
        self.mlp = MLP(input_size=mlp_in,
                       hidden_size=mlp_hidden_size,
                       n_layers=mlp_n_layers,
                       output_size=n_labels,
                       dropout=mlp_dropout,
                       activation=nn.Identity)

        # Inference layer (CRF/softmax)

        if self.use_crf:
            self.crf = ConditionalRandomField(n_labels)
            print('# CRF cost: {}'.format(self.crf.transitions.shape),
                  file=sys.stderr)
        else:
            self.softmax_cross_entropy = nn.CrossEntropyLoss()

    """
    us: batch of unigram sequences
    bs: batch of bigram sequences
    es: batch of attr sequences
    fs: batch of additonal feature sequences
    ls: batch of label sequences
    """

    # unigram, bigram, attr, feature, label
    def forward(self,
                us,
                bs=None,
                es=None,
                fs=None,
                ls=None,
                calculate_loss=True):
        lengths = self.extract_lengths(us)
        us, bs, es, fs, ls = self.pad_features(us, bs, es, fs, ls)
        xs = self.extract_features(us, bs, es, fs)
        rs = self.rnn_output(xs, lengths)
        ys = self.mlp(rs)
        loss, ps = self.predict(ys,
                                ls=ls,
                                lengths=lengths,
                                calculate_loss=calculate_loss)

        return loss, ps

    def extract_lengths(self, ts):
        device = ts[0].device
        return torch.tensor([t.shape[0] for t in ts], device=device)

    def pad_features(self, us, bs, es, fs, ls):
        batch_first = self.rnn_batch_first
        us = pad_sequence(us, batch_first=batch_first)
        bs = pad_sequence(bs, batch_first=batch_first) if bs else None
        es = pad_sequence(es, batch_first=batch_first) if es else None
        fs = pad_sequence(fs, batch_first=batch_first) if fs else None
        ls = pad_sequence(ls, batch_first=batch_first) if ls else None

        return us, bs, es, fs, ls

    def extract_features(self, us, bs, es, fs):
        xs = []
        if bs is None:
            bs = [None] * len(us)
        if es is None:
            es = [None] * len(us)
        if fs is None:
            fs = [None] * len(us)

        for u, b, e, f in zip(us, bs, es, fs):
            ue = self.unigram_embed(u)
            if self.pretrained_unigram_embed is not None:
                if self.pretrained_embed_usage == ModelUsage.ADD:
                    pe = self.pretrained_unigram_embed(u)
                    ue = ue + pe
                elif self.pretrained_embed_usage == ModelUsage.CONCAT:
                    pe = self.pretrained_unigram_embed(u)
                    ue = torch.cat((ue, pe), 1)
            ue = F.dropout(ue, p=self.embed_dropout)
            xe = ue

            if b is not None:
                be = self.bigram_embed(b)
                if self.pretrained_unigram_embed is not None:
                    if self.pretrained_embed_usage == ModelUsage.ADD:
                        pe = self.pretrained_bigram_embed(b)
                        be = be + pe
                    elif self.pretrained_embed_usage == ModelUsage.CONCAT:
                        pe = self.pretrained_bigram_embed(b)
                        be = torch.cat((be, pe), 1)
                be = F.dropout(be, p=self.embed_dropout)
                xe = torch.cat((xe, be), 1)

            if e is not None:
                ee = self.attr_embed(e)
                ee = F.dropout(ee, p=self.embed_dropout)
                xe = torch.cat((xe, ee), 1)

            if f is not None:
                xe = torch.cat((xe, f), 1)

            xs.append(xe)

        if self.rnn_batch_first:
            xs = torch.stack(xs, dim=0)
        else:
            xs = torch.stack(xs, dim=1)

        return xs

    def rnn_output(self, xs, lengths=None):
        if self.rnn_unit_type == 'lstm':
            hs, (hy, cy) = self.rnn(xs, lengths)
        else:
            hs, hy = self.rnn(xs)
        return hs

    def predict(self, rs, ls=None, lengths=None, calculate_loss=True):
        if self.crf:
            return self.predict_crf(rs, ls, lengths, calculate_loss)
        else:
            return self.predict_softmax(rs, ls, calculate_loss)

    def predict_softmax(self, ys, ls=None, calculate_loss=True):
        ps = []
        loss = torch.tensor(0, dtype=torch.float, device=ys.device)
        if ls is None:
            ls = [None] * len(ys)
        for y, l in zip(ys, ls):
            if calculate_loss:
                loss += self.softmax_cross_entropy(y, l)
            ps.append([torch.argmax(yi.data) for yi in y])

        return loss, ps

    def predict_crf(self, hs, ls=None, lengths=None, calculate_loss=True):
        device = hs.device
        if lengths is None:
            lengths = torch.tensor([h.shape[0] for h in hs], device=device)
        mask = get_mask_from_sequence_lengths(lengths, max_length=max(lengths))
        ps = self.crf.viterbi_tags(hs, mask)
        ps, score = zip(*ps)

        if calculate_loss:
            log_likelihood = self.crf(hs, ls, mask)
            loss = -1 * log_likelihood / len(lengths)
        else:
            loss = torch.tensor(np.array(0), dtype=torch.float, device=device)

        return loss, ps

    def decode(self, us, bs=None, es=None, fs=None):
        with torch.no_grad():
            _, ps = self.forward(us, bs, es, fs, calculate_loss=False)
        return ps
