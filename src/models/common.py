import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.util


class Embedding(nn.Module):
    def __init__(self, input_size, embed_size):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(input_size, embed_size)

    def forward(self, inputs):
        x = inputs[0]
        embeddings = self.embed(x)
        return embeddings


class MLP(nn.Sequential):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers,
                 output_size,
                 dropout,
                 activation=nn.ReLU):
        self.dropout = dropout
        self.layers = None

        layers = [None] * n_layers
        self.acts = [None] * n_layers
        hidden_size = hidden_size if hidden_size > 0 else output_size

        for i in range(n_layers):
            if i == 0:
                prev_size = input_size
                next_size = output_size if n_layers == 1 else hidden_size
                act = activation() if n_layers == 1 else nn.ReLU()

            elif i == n_layers - 1:
                prev_size = hidden_size
                next_size = output_size
                act = activation()

            else:
                prev_size = next_size = hidden_size
                act = nn.ReLU()

            layers[i] = nn.Linear(prev_size, next_size)
            self.acts[i] = act

        self.layers = layers
        super(MLP, self).__init__(*layers)

        for i in range(n_layers):
            print(
                '#    Affine {}-th layer: W={}, b={}, dropout={}, activation={}'
                .format(i, self.layers[i].weight.shape,
                        self.layers[i].bias.shape, self.dropout, self.acts[i]),
                file=sys.stderr)

    def forward(self, xs):
        hs_prev = xs
        hs = None

        for i in range(len(self.layers)):
            hs = self.acts[i](self.layers[i](F.dropout(hs_prev,
                                                       p=self.dropout)))
            hs_prev = hs

        return hs


class RNNTanh(nn.Module):
    def __init__(self,
                 embed_size,
                 hidden_size,
                 n_layers,
                 batch_first,
                 dropout,
                 bidirectional,
                 nonlinearity='tanh'):
        super(RNNTanh, self).__init__()
        self.dropout = dropout
        self.rnn = nn.RNN(input_size=embed_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          nonlinearity=nonlinearity,
                          batch_first=batch_first,
                          dropout=dropout,
                          bidirectional=bidirectional)

    def forward(self, xs):
        hs, hy = self.rnn(xs)
        return F.dropout(hs, p=self.dropout), hy


class GRU(nn.Module):
    def __init__(
        self,
        embed_size,
        hidden_size,
        n_layers,
        batch_first,
        dropout,
        bidirectional,
    ):
        super(GRU, self).__init__()
        self.dropout = dropout
        self.gru = nn.GRU(input_size=embed_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=batch_first,
                          dropout=dropout,
                          bidirectional=bidirectional)

    def forward(self, xs):
        hs, hy = self.gru(xs)
        return F.dropout(hs, p=self.dropout), hy


class LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, n_layers, batch_first, dropout,
                 bidirectional):
        super(LSTM, self).__init__()
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=batch_first,
                            dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, xs, lengths):
        self.lstm.flatten_parameters()
        device = xs.device
        batch_first = self.lstm.batch_first

        # batch tensor sorting for sort packing
        lengths, perm_index = torch.sort(lengths, dim=0, descending=True)
        xs = xs[perm_index]
        # pack input and lengths
        xs = nn.utils.rnn.pack_padded_sequence(xs,
                                               lengths.cpu(),
                                               batch_first=batch_first)
        hs, (hy, cy) = self.lstm(xs)
        # unpack input and lengths for unsorting
        hs, lengths = nn.utils.rnn.pad_packed_sequence(hs,
                                                       batch_first=batch_first)
        perm_index_rev = torch.tensor(models.util.inverse_indices(perm_index),
                                      device=device)
        # unsort
        # hs = hs[perm_index_rev, :, :]
        hs = hs[perm_index_rev, :]

        return F.dropout(hs, p=self.dropout), (hy, cy)


class SimpleLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, n_layers, batch_first, dropout,
                 bidirectional):
        super(SimpleLSTM, self).__init__()
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=batch_first,
                            dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, xs, lengths=None):
        # self.lstm.flatten_parameters()
        hs, (hy, cy) = self.lstm(xs)
        return F.dropout(hs, p=self.dropout), (hy, cy)


class BiaffineCombination(nn.Module):
    def __init__(self,
                 left_size,
                 right_size,
                 use_U=False,
                 use_V=False,
                 use_b=False,
                 dropout=0.0):
        # z = yWx + Ux + Vy + b ; b is scalar equals to zero
        super(BiaffineCombination, self).__init__()
        self.dropout = dropout
        self.W = nn.Linear(left_size, right_size)

        if use_U:
            self.U = nn.Linear(left_size, 1, bias=False)
        else:
            self.U = None

        if use_V:
            # self.V = nn.Linear(1, right_size, bias=False)
            self.V = nn.Linear(right_size, 1, bias=False)
        else:
            self.V = None

        if use_b:
            initialb = torch.tensor(0, dtype=torch.float)
            self.b = nn.Parameter(initialb)
        else:
            self.b = None

    def forward(self, x1, x2):
        # inputs: x1 = [x1_1 ... x1_i ... x1_n1]; dim(x1_i)=d1=left_size
        #         x2 = [x2_1 ... x2_j ... x2_n2]; dim(x2_j)=d2=right_size
        # output: o_ij = x1_i * W * x2_j + x2_j * U + b

        n1 = x1.shape[0]
        n2 = x2.shape[0]
        x2T = torch.transpose(x2, 1, 0)
        x1_W = self.W(x1)  # (n1, d1) -[linear(d1, d2)]-> (n1, d2)
        res = torch.matmul(x1_W, x2T)  # (n1, d2) * (d2, n2) => (n1, n2)

        # # batch case
        # n1 = x1.shape[0]
        # n2 = x2.shape[0]
        # # add batch axix as 1
        # x1 = x1.unsqueeze(0)
        # x2 = x2.unsqueeze(0)
        # x2T = x2.transpose(1, 2)
        # x1_W = self.W(x1)
        # res = torch.bmm(x1_W, x2T)
        # # remove batch axis
        # res = res.squeeze(0)

        if self.U is not None:
            x1_U = self.U(x1)  # (n1, d1) -[linear(d1, 1)]-> (n1, 1)
            res = torch.add(res, x1_U)  # (n1, 1) -> (n1, n2)

        if self.V is not None:
            V_x2 = self.V(x2)  # (d2, n2) -[linear(d2, 1)]-> (n1, 1)
            V_x2T = V_x2.transpose(1, 0)  # (n1, 1) -> (1, n1)
            res = res + V_x2T  # (1, n1) => (n1, n2)

        if self.b is not None:
            b = self.b
            res = torch.add(res, b)

        return res


class Transformer(nn.Module):
    def __init__(self,
                 embed_size,
                 ff_hidden_size,
                 hidden_size,
                 n_layers,
                 n_heads,
                 dropout,
                 max_seq_len=5000,
                 activation='relu'):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.ff_hidden_size = ff_hidden_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.pos_encoder = PositionalEncoding(embed_size=embed_size,
                                              dropout=dropout,
                                              max_seq_len=max_seq_len)
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=n_heads,
            dim_feedforward=ff_hidden_size,
            dropout=dropout,
            activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layers, num_layers=n_layers)
        self.decoder = nn.Linear(embed_size, hidden_size)

    def forward(self, xs, src_key_padding_mask=None):
        xs = xs * math.sqrt(self.hidden_size)
        xs = xs.transpose(0, 1)
        xs = self.pos_encoder(xs)

        hs = self.transformer_encoder(
            xs, src_key_padding_mask=src_key_padding_mask)
        hs = self.decoder(hs)
        hs = hs.transpose(0, 1)
        return hs


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = dropout

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() *
            (-math.log(10000.0) / embed_size))
        pe = torch.zeros(max_seq_len, 1, embed_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, xs):
        xs = xs + self.pe[:xs.size(0), :]
        return F.dropout(xs, p=self.dropout)
