from allennlp.modules.conditional_random_field import ConditionalRandomField
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
from models.tagger import RNNTagger


class RNNTaggerWithChunk(RNNTagger):
    def __init__(self,
                 n_vocab,
                 unigram_embed_size,
                 n_bigrams,
                 bigram_embed_size,
                 n_chunks,
                 chunk_embed_size,
                 rnn_unit_type,
                 rnn_bidirection,
                 rnn_batch_first,
                 rnn_n_layers1,
                 rnn_hidden_size1,
                 rnn_n_layers2,
                 rnn_hidden_size2,
                 mlp_n_layers,
                 mlp_hidden_size,
                 n_labels,
                 use_crf=True,
                 feat_size=0,
                 rnn_dropout=0.0,
                 embed_dropout=0.0,
                 biaffine_dropout=0.0,
                 mlp_dropout=0.0,
                 chunk_vector_dropout=0,
                 pretrained_unigram_embed_size=0,
                 pretrained_bigram_embed_size=0,
                 pretrained_chunk_embed_size=0,
                 pretrained_embed_usage=ModelUsage.NONE,
                 chunk_pooling_type=constants.AVG,
                 min_chunk_len=1,
                 max_chunk_len=0,
                 chunk_loss_ratio=0,
                 biaffine_type='',
                 file=sys.stderr):
        nn.Module.__init__(self)

        self.n_vocab = n_vocab
        self.unigram_embed_size = unigram_embed_size
        self.n_bigrams = n_bigrams
        self.bigram_embed_size = bigram_embed_size
        self.n_chunks = n_chunks
        self.chunk_embed_size = chunk_embed_size

        self.rnn_unit_type = rnn_unit_type
        self.rnn_bidirection = rnn_bidirection
        self.rnn_batch_first = rnn_batch_first
        self.rnn_n_layers1 = rnn_n_layers1
        self.rnn_hidden_size1 = rnn_hidden_size1
        self.rnn_n_layers2 = rnn_n_layers2
        self.rnn_hidden_size2 = rnn_hidden_size2

        self.mlp_n_layers = mlp_n_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.n_labels = n_labels
        self.use_crf = use_crf
        self.feat_size = feat_size

        self.rnn_dropout = rnn_dropout
        self.embed_dropout = embed_dropout
        self.biaffine_dropout = biaffine_dropout
        self.mlp_dropout = mlp_dropout
        self.chunk_vector_dropout = chunk_vector_dropout

        self.pretrained_unigram_embed_size = pretrained_unigram_embed_size
        self.pretrained_bigram_embed_size = pretrained_bigram_embed_size
        self.pretrained_chunk_embed_size = pretrained_chunk_embed_size
        self.pretrained_embed_usage = pretrained_embed_usage

        self.chunk_pooling_type = chunk_pooling_type
        self.min_chunk_len = min_chunk_len
        self.max_chunk_len = max_chunk_len
        self.chunk_loss_ratio = chunk_loss_ratio

        self.biaffine_type = biaffine_type

        self.use_attention = (chunk_pooling_type == constants.WAVG
                              or chunk_pooling_type == constants.WCON)
        self.use_concat = (chunk_pooling_type == constants.CON
                           or chunk_pooling_type == constants.WCON)
        self.use_average = not self.use_concat
        self.use_rnn2 = rnn_n_layers2 > 0 and rnn_hidden_size2 > 0

        self.chunk_embed_size_merged = (
            chunk_embed_size +
            (pretrained_chunk_embed_size
             if pretrained_embed_usage == ModelUsage.CONCAT else 0))

        if self.use_concat:
            self.chunk_concat_num = sum(
                [i for i in range(min_chunk_len, max_chunk_len + 1)])
            self.chunk_embed_out_size = self.chunk_embed_size_merged * self.chunk_concat_num
        else:
            self.chunk_embed_out_size = self.chunk_embed_size_merged

        self.unigram_embed = None
        self.bigram_embed = None
        self.chunk_embed = None
        self.pretrained_unigram_embed = None
        self.pretrained_bigram_embed = None
        self.pretrained_chunk_embed = None
        self.rnn = None
        self.biaffine = None
        self.rnn2 = None
        self.mlp = None
        self.crf = None
        self.cross_entropy_loss = None

        print('### Parameters', file=sys.stderr)
        print('# Chunk pooling type: {}'.format(self.chunk_pooling_type),
              file=sys.stderr)
        print('# Chunk loss ratio: {}'.format(self.chunk_loss_ratio),
              file=sys.stderr)

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

        self.additional_feat_size = feat_size
        if feat_size > 0:
            embed_size += feat_size
            print(
                '# Additional features size (dimension): {}'.format(feat_size),
                file=sys.stderr)

        self.chunk_embed, self.pretrained_chunk_embed = models.util.construct_embeddings(
            n_chunks, chunk_embed_size, pretrained_chunk_embed_size,
            pretrained_embed_usage)
        print('# Chunk embedding matrix: W={}'.format(
            self.chunk_embed.weight.shape),
              file=sys.stderr)
        if self.pretrained_chunk_embed is not None:
            print('# Pretrained chunk embedding matrix: W={}'.format(
                self.pretrained_chunk_embed.weight.shape),
                  file=sys.stderr)

        # recurrent layers 1

        self.rnn_unit_type = rnn_unit_type
        self.rnn = models.util.construct_RNN(unit_type=rnn_unit_type,
                                             embed_size=embed_size,
                                             hidden_size=rnn_hidden_size1,
                                             n_layers=rnn_n_layers1,
                                             batch_first=rnn_batch_first,
                                             dropout=rnn_dropout,
                                             bidirectional=rnn_bidirection)
        rnn_output_size1 = rnn_hidden_size1 * (2 if rnn_bidirection else 1)

        # biaffine b/w token and chunk

        if self.use_attention:
            use_U = 'u' in biaffine_type or 'U' in biaffine_type
            use_V = 'v' in biaffine_type or 'V' in biaffine_type
            use_b = 'b' in biaffine_type or 'B' in biaffine_type

            biaffine_left_size = rnn_output_size1
            self.biaffine = BiaffineCombination(
                biaffine_left_size,
                self.chunk_embed_size_merged,
                use_U=use_U,
                use_V=use_V,
                use_b=use_b,
                dropout=biaffine_dropout,
            )

            print(
                '# Biaffine layer for attention: W={}, U={}, V={}, b={}, dropout={}'
                .format(
                    self.biaffine.W.weight.shape, self.biaffine.U.weight.shape
                    if self.biaffine.U is not None else None,
                    self.biaffine.V.weight.shape
                    if self.biaffine.V is not None else None,
                    self.biaffine.b if self.biaffine.b is not None else None,
                    self.biaffine_dropout),
                file=sys.stderr)

        # chunk vector dropouta

        print('# Chunk vector dropout={}'.format(self.chunk_vector_dropout),
              file=sys.stderr)

        # recurrent layers 2

        if self.use_rnn2:
            rnn_input_size2 = rnn_output_size1 + self.chunk_embed_out_size

            self.rnn2 = models.util.construct_RNN(
                unit_type=rnn_unit_type,
                embed_size=rnn_input_size2,
                hidden_size=rnn_hidden_size2,
                n_layers=rnn_n_layers2,
                batch_first=rnn_batch_first,
                dropout=rnn_dropout,
                bidirectional=rnn_bidirection)
            rnn_output_size2 = rnn_hidden_size2 * (2 if rnn_bidirection else 1)
            mlp_input_size = rnn_output_size2
        else:
            mlp_input_size = rnn_output_size1 + self.chunk_embed_out_size

        # MLP

        print('# MLP', file=sys.stderr)
        self.mlp = MLP(input_size=mlp_input_size,
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
    us: mini-batch of token (char) sequences
    cs: mini-batch of chunk (word) sequences
    ds: mini-batch of chunk (word) sequences for concat models
    ms: mini-batch of masking matrix (tuples)
    bs: mini-batch of bigram sequences
    fs: mini-batch of additional features
    gls: mini-batch of segmentation label sequences
    gcs: mini-batch of attention label sequences
    """

    def forward(self,
                us,
                cs,
                ds,
                ms,
                bs=None,
                fs=None,
                gls=None,
                gcs=None,
                calculate_loss=True):
        lengths = super().extract_lengths(us)
        us, cs, bs, fs, gls, gcs = self.pad_features(us, cs, bs, fs, gls, gcs)

        closs = None
        pcs = None

        xs = self.extract_token_features(us, bs, None,
                                         fs)  # token unigram etc. -[Embed]-> x
        rs = self.rnn_output(xs, lengths)  # x -[RNN]-> r

        if cs is not None:
            ws = self.extract_chunk_features(
                cs)  # chunk -[Embed]-> w (chunk sequence)
        else:
            ws = [None] * len(us)

        if ds is not None:
            vs = self.extract_chunk_features(
                ds)  # chunk -[Embed]-> w (concatenated chunk matrix)
        else:
            vs = [None] * len(us)

        # r @ r$w -> h
        closs, pcs, hs = self.act_and_merge_features(rs,
                                                     ws,
                                                     vs,
                                                     ms,
                                                     gcs,
                                                     lengths,
                                                     get_att_score=False)

        if self.use_rnn2:
            hs = self.rnn2_output(hs, lengths)  # h -[RNN]-> h'
        ys = self.mlp(hs)
        sloss, pls = self.predict(ys,
                                  ls=gls,
                                  lengths=lengths,
                                  calculate_loss=calculate_loss)

        if closs is not None:
            loss = (1 - self.chunk_loss_ratio
                    ) * sloss + self.chunk_loss_ratio * closs
        else:
            loss = sloss

        return loss, pls, pcs

    def pad_features(self, us, cs, bs, fs, gls, gcs):
        batch_first = self.rnn_batch_first
        us = pad_sequence(us, batch_first=batch_first)
        cs = pad_sequence(cs, batch_first=batch_first) if cs else None
        bs = pad_sequence(bs, batch_first=batch_first) if bs else None
        fs = pad_sequence(fs, batch_first=batch_first) if fs else None
        gls = pad_sequence(gls, batch_first=batch_first) if gls else None
        gcs = pad_sequence(gcs, batch_first=batch_first) if gcs else None

        return us, cs, bs, fs, gls, gcs

    def trim_features_by_length(self, x, v, gc, length):
        x = x[:length, :] if x is not None else None
        v = v[:, :length, :] if v is not None else None
        gc = gc[:length] if gc is not None else None
        return x, v, gc

    def decode(self, us, cs, ds, ms, bs=None, fs=None):
        with torch.no_grad():
            _, ps, _ = self.forward(us,
                                    cs,
                                    ds,
                                    ms,
                                    bs,
                                    fs,
                                    calculate_loss=False)
        return ps

    def rnn2_output(self, xs, lengths=None):
        if self.rnn_unit_type == 'lstm':
            hs, (hy, cy) = self.rnn2(xs, lengths)
        else:
            hs, hy = self.rnn2(xs)
        return hs

    def extract_token_features(self, us, bs, es, fs):
        return super().extract_features(us, bs, es, fs)

    def extract_chunk_features(self, cs):
        xs = []
        for c in cs:
            xe = self.chunk_embed(c) if c.byte().any() else None

            if c is not None and self.pretrained_chunk_embed is not None:
                if self.pretrained_embed_usage == ModelUsage.ADD:
                    pce = self.pretrained_chunk_embed(c)
                    xe = xe + pce
                elif self.pretrained_embed_usage == ModelUsage.CONCAT:
                    pce = self.pretrained_chunk_embed(c)
                    xe = F.concat((xe, pce), 1)
            xs.append(xe)
        return xs

    def act_and_merge_features(self,
                               xs,
                               ws,
                               vs,
                               ms,
                               gcs=None,
                               lengths=None,
                               get_att_score=False):
        hs = []
        pcs = []
        ass = []  # attention scores

        device = xs.device
        batch_first = self.rnn_batch_first
        closs = torch.tensor(0, dtype=torch.float, device=device)

        if gcs is None:
            gcs = [None] * len(xs)
        for x, w, v, gc, mask, l in zip(xs, ws, vs, gcs, ms, lengths):
            # print('x', x.shape)
            x, v, gc = self.trim_features_by_length(x, v, gc, l)

            if w is None and v is None:  # no words were found for validation/test data
                a = torch.zeros((len(x), self.chunk_embed_out_size),
                                dtype=torch.float,
                                device=device)
                pc = torch.zeros(len(x), dtype=int, device=device)
                pcs.append(pc)
                h = torch.cat((x, a), dim=1)  # (n, dt) @ (n, dc) => (n, dt+dc)
                hs.append(h)
                continue

            if w is not None:
                w = F.dropout(w, p=self.embed_dropout)
                mask_ij = mask[0]
                cl, wl = mask_ij.size()
                w = w[:wl, :]

            # calculate weight for w

            mask_ij = mask[0]
            if self.use_attention:  # wavg or wcon
                mask_i = mask[1]

                w_scores = self.biaffine(
                    F.dropout(x, p=self.biaffine_dropout),
                    F.dropout(w, p=self.biaffine_dropout))  # (n, m)
                w_scores = w_scores + mask_ij  # a masked element becomes 0 after softmax operation

                w_weight = F.softmax(w_scores,
                                     dim=1)  # sum(rows[i], cols) == 1
                w_weight = w_weight * mask_i  # raw of char w/o no candidate words become a 0 vector
                # print('ww', w_weight.shape, '\n', w_weight)

            elif self.chunk_pooling_type == constants.AVG:
                w_weight = self.normalize(mask_ij)

            if not self.use_concat and self.chunk_vector_dropout > 0:
                mask_drop = torch.ones(w_weight.shape,
                                       dtype=torch.float,
                                       device=device)
                for i in range(w_weight.shape[0]):
                    if self.chunk_vector_dropout > np.random.rand():
                        mask_drop[i] = torch.zeros(w_weight.shape[1],
                                                   dtype=torch.float,
                                                   device=device)
                w_weight *= mask_drop

            # calculate weight for v

            if self.use_concat:
                mask_ik = mask[2]
                n = x.shape[0]
                wd = self.chunk_embed_size_merged  # w.shape[1]
                if self.chunk_pooling_type == constants.WCON:
                    ikj_table = mask[3]
                    v_weight0 = torch.cat(
                        [
                            torch.unsqueeze(  # (n, m) -> (n, k)
                                w_weight[i][ikj_table[i]],
                                dim=0) for i in range(n)
                        ],
                        dim=0)
                    v_weight0 *= mask_ik

                else:
                    v_weight0 = mask_ik

                v_weight = torch.transpose(v_weight0, 1, 0)  # (n,k) -> (k,n)
                v_weight = torch.unsqueeze(v_weight, 2)  # (k,n) -> (k,n,1)
                v_weight = v_weight.repeat((1, 1, wd))  # (k,n,1) -> (k,n,wd)
                v_weight = torch.transpose(v_weight, 1, 0).reshape(
                    (v_weight.size(1), -1))  # (k,n,wd) -> (n,k*wd)

                if self.chunk_vector_dropout > 0:
                    mask_drop = torch.ones(v_weight.shape,
                                           dtype=torch.float,
                                           device=device)
                    for i in range(v_weight.shape[0]):
                        if self.chunk_vector_dropout > np.random.rand():
                            mask_drop[i] = torch.zeros(v_weight.shape[1],
                                                       dtype=torch.float,
                                                       device=device)
                    v_weight *= mask_drop

            # calculate summary vector a

            if self.use_average:  # avg or wavg
                a = torch.matmul(w_weight, w)  # (n, m) * (m, dc)  => (n, dc)

            else:  # con or wcon
                v = torch.transpose(v, 1, 0).reshape((v.size(1), -1))
                a = v * v_weight
                # print('a', a.shape, a)

            # get predicted (attended) chunks
            if self.use_attention:  # wavg or wcon
                if self.chunk_pooling_type == constants.WAVG:
                    weight = w_weight
                else:
                    weight = v_weight0

                pc = torch.argmax(weight, dim=1).data.cpu().numpy()
                pcs.append(pc)

                if get_att_score:
                    ascore = torch.argmax(weight, dim=1).data.cpu().numpy()
                    ass.append(ascore)

            h = torch.cat((x, a), dim=1)  # (n, dt) @ (n, dc) => (n, dt+dc)

            hs.append(h)

        hs = pad_sequence(hs, batch_first=batch_first)

        if closs.data == 0:
            closs = None
        else:
            closs /= len(xs)

        if get_att_score:
            return closs, pcs, hs, ass
        else:
            return closs, pcs, hs

    def normalize(self, array):
        device = array.device
        denom = torch.sum(array, dim=1, keepdims=True)
        adjust = torch.tensor([[
            torch.tensor(1, dtype=torch.float, device=device)
            if denom.data[i][0] == 0 else torch.tensor(
                0, dtype=torch.float, device=device)
        ] for i in range(denom.shape[0])],
                              dtype=torch.float,
                              device=device)
        denom = denom + adjust  # avoid zero division
        return torch.div(array, denom)
