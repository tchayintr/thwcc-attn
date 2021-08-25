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
from models.common import MLP

class TFMTagger(nn.Module):
    def __init__(self,
                 n_vocab,
                 unigram_embed_size,
                 tfm_n_layers,
                 tfm_ff_hidden_size,
                 tfm_hidden_size,
                 tfm_n_heads,
                 mlp_n_layers,
                 mlp_hidden_size,
                 n_labels,
                 use_crf=True,
                 embed_dropout=0.0,
                 tfm_dropout=0.0,
                 mlp_dropout=0.0,
                 max_seq_len=512,
                 pretrained_unigram_embed_size=0,
                 pretrained_embed_usage=ModelUsage.NONE):

        super(TFMTagger, self).__init__()
        self.n_vocab = n_vocab
        self.unigram_embed_size = unigram_embed_size
        self.tfm_n_layers = tfm_n_layers
        self.tfm_ff_hidden_size = tfm_ff_hidden_size
        self.tfm_hidden_size = tfm_hidden_size
        self.tfm_n_heads = tfm_n_heads

        self.mlp_n_layers = mlp_n_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.n_labels = n_labels
        self.use_crf = use_crf
        self.embed_dropout = embed_dropout
        self.tfm_dropout = tfm_dropout
        self.mlp_dropout = mlp_dropout
        self.max_seq_len = max_seq_len

        self.pretrained_unigram_embed_size = pretrained_unigram_embed_size
        self.pretrained_embed_usage = pretrained_embed_usage

        self.unigram_embed = None
        self.pretrained_unigram_embed = None
        self.transformer = None
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

        # transformer layers

        self.transformer = models.util.construct_TFM(
            embed_size=embed_size,
            ff_hidden_size=tfm_ff_hidden_size,
            hidden_size=tfm_hidden_size,
            n_layers=tfm_n_layers,
            n_heads=tfm_n_heads,
            dropout=tfm_dropout,
            max_seq_len=max_seq_len)
        tfm_output_size = tfm_hidden_size

        # MLP

        print('# MLP', file=sys.stderr)
        mlp_in = tfm_output_size
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

    # unigram, label
    def forward(self, us, ls=None, calculate_loss=True):
        lengths = self.extract_lengths(us)
        us, ls = self.pad_features(us, ls)
        mask = models.util.generate_src_key_padding_mask(us)
        xs = self.extract_features(us)
        rs = self.tfm_output(xs, ~mask)
        ys = self.mlp(rs)
        loss, ps = self.predict(ys,
                                ls=ls,
                                lengths=lengths,
                                calculate_loss=calculate_loss)

        return loss, ps

    def extract_lengths(self, ts):
        device = ts[0].device
        return torch.tensor([t.shape[0] for t in ts], device=device)

    def pad_features(self, us, ls):
        us = pad_sequence(us, batch_first=True)
        ls = pad_sequence(ls, batch_first=True) if ls else None
        return us, ls

    def extract_features(self, us):
        xs = []

        for u in us:
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

            xs.append(xe)

        # batch first
        xs = torch.stack(xs, dim=0)
        return xs

    def tfm_output(self, xs, mask):
        hs = self.transformer(xs, mask)
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

    def decode(self, *us):
        with torch.no_grad():
            _, ps = self.forward(us, calculate_loss=False)
        return ps

    # def generate_3d_mask_from_sequence_lengths(self, xs, lengths):
    #     # add new axis at dim 1 (unsqueeze)
    #     # then repeat the batch size for MultiHeadAttn times n_heads
    #     # the maximum length in the new axis (dim 1)
    #     mask = (xs != 0).unsqueeze(1).repeat(1 * self.tfm_n_heads,
    #                                          max(lengths), 1)
    #     return mask
