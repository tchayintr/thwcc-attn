from enum import Enum, auto
import numpy as np
import sys
import torch
import torch.nn as nn

from models.common import Embedding, GRU, RNNTanh, LSTM, Transformer


class ModelUsage(Enum):
    NONE = auto()
    ADD = auto()
    CONCAT = auto()
    INIT = auto()

    def get_instance(key):
        if key.lower() == 'concat':
            return ModelUsage.CONCAT
        elif key.lower() == 'add':
            return ModelUsage.ADD
        elif key.lower() == 'init':
            return ModelUsage.INIT
        else:
            return ModelUsage.NONE


def construct_embeddings(n_vocab,
                         rand_size,
                         pretrained_size=0,
                         usage=ModelUsage.INIT,
                         ignore_label=None):
    if pretrained_size <= 0 or usage == ModelUsage.NONE:
        rand_embedding = nn.Embedding(n_vocab,
                                      rand_size,
                                      padding_idx=ignore_label)
        pretrained_embedding = None

    elif usage == ModelUsage.CONCAT or usage == ModelUsage.ADD:
        rand_embedding = nn.Embedding(n_vocab,
                                      rand_size,
                                      padding_idx=ignore_label)
        pretrained_embedding = nn.Embedding(n_vocab,
                                            pretrained_size,
                                            padding_idx=ignore_label)

    elif usage == ModelUsage.INIT:
        rand_embedding = nn.Embedding(n_vocab,
                                      pretrained_size,
                                      padding_idx=ignore_label)
        pretrained_embedding = None

    return rand_embedding, pretrained_embedding


def construct_RNN(
    unit_type,
    embed_size,
    hidden_size,
    n_layers,
    batch_first,
    dropout,
    bidirectional,
):
    rnn = None

    if unit_type == 'lstm':
        rnn = LSTM(embed_size, hidden_size, n_layers, batch_first, dropout,
                   bidirectional)

    elif unit_type == 'gru':
        rnn = GRU(embed_size, hidden_size, n_layers, batch_first, dropout,
                  bidirectional)

    else:
        rnn = RNNTanh(
            embed_size,
            hidden_size,
            n_layers,
            batch_first,
            dropout,
            bidirectional,
            nonlinearity='tanh',
        )

    print('# RNN unit: {}'.format(rnn), file=sys.stderr)

    return rnn


def construct_TFM(embed_size,
                  ff_hidden_size,
                  hidden_size,
                  n_layers,
                  n_heads,
                  dropout,
                  max_seq_len=5000,
                  activation='relu'):

    tfm = Transformer(embed_size=embed_size,
                      ff_hidden_size=ff_hidden_size,
                      hidden_size=hidden_size,
                      n_layers=n_layers,
                      n_heads=n_heads,
                      dropout=dropout,
                      max_seq_len=max_seq_len,
                      activation=activation)
    print('# Transformer unit: {}'.format(tfm), file=sys.stderr)

    return tfm


def load_pretrained_embedding_layer(id2unigram,
                                    embed,
                                    external_model,
                                    finetuning=False):
    xp = torch
    n_vocab = len(id2unigram)
    size = external_model.wv.syn0[0].shape[0]

    weight = []
    count = 0

    for i in range(n_vocab):
        key = id2unigram[i]
        if key in external_model.wv.vocab:
            vec = external_model.wv[key]
            count += 1
        else:
            if finetuning:
                vec = nn.init.normal_(torch.zeros((size, ), dtype=torch.float))
            else:
                vec = xp.zeros(size, dtype=torch.float)
        weight.append(vec)

    weight = xp.reshape(weight, (n_vocab, size))
    embed.weight = torch.nn.Parameter(weight)

    if count >= 1:
        print('Use {} pretrained embedding vectors\n'.format(count),
              file=sys.stderr)


def grow_embedding_layers(n_vocab_org,
                          n_vocab_grown,
                          rand_embed,
                          pretrained_embed=None,
                          external_model=None,
                          id2unigram_grown=None,
                          pretrained_model_usage=ModelUsage.NONE,
                          train=False,
                          fasttext=False):
    if n_vocab_org == n_vocab_grown:
        return

    if external_model and pretrained_model_usage != ModelUsage.NONE:
        if pretrained_model_usage == ModelUsage.INIT:
            grow_embedding_layer_with_pretrained_model(n_vocab_org,
                                                       n_vocab_grown,
                                                       rand_embed,
                                                       external_model,
                                                       id2unigram_grown,
                                                       train=train,
                                                       fasttext=fasttext)

        else:
            grow_embedding_layers_with_pretrained_model(n_vocab_org,
                                                        n_vocab_grown,
                                                        rand_embed,
                                                        pretrained_embed,
                                                        external_model,
                                                        id2unigram_grown,
                                                        train=train)

    else:
        grow_embedding_layer_without_pretrained_model(n_vocab_org,
                                                      n_vocab_grown,
                                                      rand_embed,
                                                      train=train)


def grow_embedding_layer_without_pretrained_model(n_vocab_org,
                                                  n_vocab_grown,
                                                  rand_embed,
                                                  train=False):
    xp = torch
    device = rand_embed.weight.device
    diff = n_vocab_grown - n_vocab_org
    d_rand = rand_embed.weight.shape[1]

    if train:
        w2_rand = nn.init.normal_(
            xp.zeros((diff, d_rand), dtype=torch.float, device=device))
    else:
        w2_rand = xp.zeros((diff, d_rand), dtype=torch.float, device=device)

    w_rand = torch.cat((rand_embed.weight, w2_rand), axis=0)
    rand_embed.weight = torch.nn.Parameter(w_rand)
    print('Grow embedding matrix: {} -> {}'.format(n_vocab_org,
                                                   rand_embed.weight.shape[0]),
          file=sys.stderr)


# rand model -> grow using external model
def grow_embedding_layer_with_pretrained_model(n_vocab_org,
                                               n_vocab_grown,
                                               rand_embed,
                                               external_model,
                                               id2unigram_grown,
                                               train=False,
                                               fasttext=False):
    diff = n_vocab_grown - n_vocab_org
    d_rand = rand_embed.weight.shape[1]

    count = 0
    w2_rand = []

    # [MEMO] the following error happened if fasttext=True:
    #   cupy.cuda.cudnn.CuDNNError: CUDNN_STATUS_INTERNAL_ERROR: b'CUDNN_STATUS_INTERNAL_ERROR'
    wv_vocab = external_model.wv if fasttext else external_model.wv.vocab
    for i in range(n_vocab_org, n_vocab_grown):
        key = id2unigram_grown[i]
        if key in wv_vocab:
            vec_rand = torch.tensor(external_model.wv[key], dtype=torch.float)
            count += 1
        elif train:
            vec_rand = nn.init.normal_(
                torch.zeros((d_rand, ), dtype=torch.float))
        else:
            vec_rand = rand_embed.weight[
                0].data  # use pretrained vector of unknown token
        w2_rand.append(vec_rand)

    # w2_rand = np.reshape(w2_rand, (diff, d_rand))
    # if cuda.get_array_module(rand_embed.W) == cuda.cupy:
    #     w2_rand = chainer.Variable(w2_rand)
    #     w2_rand.to_gpu()
    w2_rand = torch.stack(w2_rand)
    assert w2_rand.size() == (diff, d_rand)

    w_rand = torch.cat((rand_embed.weight, w2_rand), axis=0)
    rand_embed.weight = torch.nn.Parameter(w_rand)

    print('Grow embedding matrix: {} -> {}'.format(n_vocab_org,
                                                   rand_embed.weight.shape[0]),
          file=sys.stderr)
    if count >= 1:
        print('Add {} pretrained embedding vectors'.format(count),
              file=sys.stderr)


# rand model       -> grow
# pretrained model -> grow using external model
def grow_embedding_layers_with_pretrained_model(n_vocab_org,
                                                n_vocab_grown,
                                                rand_embed,
                                                pretrained_embed,
                                                external_model,
                                                id2unigram_grown,
                                                train=False):
    diff = n_vocab_grown - n_vocab_org
    d_rand = rand_embed.weight.shape[1]
    d_pretrained = pretrained_embed.weight.shape[
        1]  # external_model.wv.syn0[0].shape[0]

    count = 0
    w2_rand = []
    w2_pretrained = []

    for i in range(n_vocab_org, n_vocab_grown):
        if train:  # resume training
            vec_rand = nn.init.normal_(
                torch.zeros((d_rand, ), dtype=torch.float))
        else:  # test
            vec_rand = rand_embed.weight[
                0].data  # use pretrained vector of unknown token
        w2_rand.append(vec_rand)

        key = id2unigram_grown[i]
        if key in external_model.wv.vocab:
            vec_pretrained = torch.tensor(external_model.wv[key],
                                          dtype=torch.float)
            count += 1
        else:
            vec_pretrained = torch.zeros(d_pretrained, dtype=torch.float)
        w2_pretrained.append(vec_pretrained)

    # w2_rand = np.reshape(w2_rand, (diff, d_rand))
    # if cuda.get_array_module(rand_embed.W) == cuda.cupy:
    #     w2_rand = chainer.Variable(w2_rand)
    #     w2_rand.to_gpu()
    w2_rand = torch.stack(w2_rand)
    assert w2_rand.size() == (diff, d_rand)
    w_rand = torch.cat((rand_embed.weight, w2_rand), axis=0)
    rand_embed.weight = torch.nn.Parameter(w_rand)

    # w2_pretrained = np.reshape(w2_pretrained, (diff, d_pretrained))
    # if cuda.get_array_module(rand_embed.W) == cuda.cupy:
    #     w2_pretrained = chainer.Variable(w2_pretrained)
    #     w2_pretrained.to_gpu()
    w2_pretrained = torch.stack(w2_pretrained)
    assert w2_pretrained.size() == (diff, d_pretrained)
    assert w2_rand.size() == (diff, d_rand)
    w_pretrained = torch.cat((pretrained_embed.weight, w2_pretrained), 0)
    pretrained_embed.weight = torch.nn.Parameter(w_pretrained)

    print('Grow embedding matrix: {} -> {}'.format(n_vocab_org,
                                                   rand_embed.weight.shape[0]),
          file=sys.stderr)
    print('Grow pretrained embedding matrix: {} -> {}'.format(
        n_vocab_org, pretrained_embed.weight.shape[0]),
          file=sys.stderr)
    if count >= 1:
        print('Add {} pretrained embedding vectors'.format(count),
              file=sys.stderr)


def grow_crf_layer(n_labels_org, n_labels_grown, crf, file=sys.stderr):
    diff = n_labels_grown - n_labels_org
    if diff <= 0:
        return

    c_org = crf.cost
    c_diff1 = torch.tensor(np.zeros((n_labels_org, diff), dtype=np.float32))
    c_diff2 = torch.tensor(np.zeros((diff, n_labels_grown), dtype=np.float32))
    c_tmp = torch.cat((c_org, c_diff1), 1)
    c_new = torch.cat((c_tmp, c_diff2), 0)
    crf.cost = torch.nn.Parameter(c_new)

    print('Grow CRF layer: {} -> {}'.format(c_org.shape,
                                            crf.cost.shape,
                                            file=sys.stderr))


def grow_MLP(n_labels_org, n_labels_grown, out_layer, file=sys.stderr):
    diff = n_labels_grown - n_labels_org
    if diff <= 0:
        return

    w_org = out_layer.weight
    w_org_shape = w_org.shape

    size = w_org.shape[1]
    w_diff_array = np.zeros((diff, size), dtype=np.float32)
    w_diff_array[:] = np.random.normal(scale=1.0, size=(diff, size))
    w_diff = torch.nn.Parameter(w_diff_array)
    w_new = torch.cat((w_org, w_diff), 0)
    out_layer.weight = torch.nn.Parameter(w_new)
    w_shape = out_layer.weight.shape

    if 'b' in out_layer.__dict__:
        b_org = out_layer.b
        b_org_shape = b_org.shape
        b_diff_array = np.zeros((diff, ), dtype=np.float32)
        b_diff_array[:] = np.random.normal(scale=1.0, size=(diff, ))
        b_diff = torch.nn.Parameter(b_diff_array)
        b_new = torch.cat((b_org, b_diff), 0)
        out_layer.b = torch.nn.Parameter(b_new)
        b_shape = out_layer.b.shape
    else:
        b_org_shape = b_shape = None

    print('Grow MLP output layer: {}, {} -> {}, {}'.format(
        w_org_shape, b_org_shape, w_shape, b_shape),
          file=sys.stderr)


def grow_biaffine_layer(n_labels_org,
                        n_labels_grown,
                        biaffine,
                        file=sys.stderr):
    diff = n_labels_grown - n_labels_org
    if diff <= 0:
        return

    w_org = biaffine.weight
    w_org_shape = w_org.shape

    size = w_org.shape[1]

    w_diff_array = np.zeros((diff, size), dtype=np.float32)
    w_diff_array[:] = np.random.normal(scale=1.0, size=(diff, size))
    w_diff = torch.nn.Parameter(w_diff_array)
    w_new = torch.cat((w_org, w_diff), 0)
    biaffine.weight = torch.nn.Parameter(w_new)
    w_shape = biaffine.weight.shape

    if 'b' in biaffine.__dict__:
        b_org = biaffine.b
        b_org_shape = b_org.shape
        b_diff_array = np.zeros((diff, ), dtype=np.float32)
        b_diff_array[:] = np.random.normal(scale=1.0, size=(diff, ))
        b_diff = torch.nn.Parameter(b_diff_array)
        b_new = torch.cat((b_org, b_diff), 0)
        biaffine.b = torch.nn.Parameter(b_new)
        b_shape = biaffine.b.shape
    else:
        b_org_shape = b_shape = None

    print('Grow biaffine layer: {}, {} -> {}, {}'.format(
        w_org_shape, b_org_shape, w_shape, b_shape),
          file=sys.stderr)


def inverse_indices(indices):
    indices = indices.cpu().numpy()
    r = np.empty_like(indices)
    r[indices] = np.arange(len(indices))
    return r


def generate_src_key_padding_mask(xs, padding_idx=0):
    mask = (xs != padding_idx)
    return mask
