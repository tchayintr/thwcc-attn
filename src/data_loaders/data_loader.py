import pickle
import re

import constants


class Data(object):
    def __init__(self, inputs=None, outputs=None):
        self.inputs = inputs  # list of input sequences e.g. chars, words
        self.outputs = outputs  # list of output sequences (label sequences)


class RestorableData(Data):
    def __init__(self, inputs=None, outputs=None, orgdata=None):
        super().__init__(inputs, outputs)
        self.orgdata = orgdata


class DataLoader(object):
    def __init__(self):
        self.lowercasing = False
        self.normalize_digits = False

    def load_gold_data(self, data_format, path, train=True):
        # to be implemented in sub-class
        pass

    def load_decode_data(self, data_format, path, dic=None):
        # to be implemented in sub-class
        pass

    def normalize_input_line(self, line):
        line = re.sub(' +', ' ', line).strip(' \t\n')
        return line

    def preprocess_token(self, token):
        if self.lowercasing:
            token = token.lower()
        if self.normalize_digits:
            token = re.sub(r'[0-9๐-๙]+', constants.NUM_SYMBOL, token)
        return token

    def to_be_registered(self,
                         token,
                         train,
                         freq_tokens=set(),
                         refer_vocab=set()):
        if train:
            if not freq_tokens or token in freq_tokens:
                return True
        else:
            if token in refer_vocab:
                return True

        return False


def load_pickled_data(filename_wo_ext):
    dump_path = filename_wo_ext + '.pickle'

    with open(dump_path, 'rb') as f:
        obj = pickle.load(f)

    return Data(obj)


def dump_pickled_data(filename_wo_ext, data):
    dump_path = filename_wo_ext + '.pickle'

    with open(dump_path, 'wb') as f:
        obj = (data)
        obj = pickle.dump(obj, f)


def create_all_char_ngrams(chars, n):
    seq_len = len(chars)
    if n > seq_len or n <= 0 or seq_len == 0:
        return []

    ngrams = []
    for i in range(seq_len - n + 1):
        ngrams.append(chars[i:i + n])
    return ngrams


def create_all_char_ngram_indexes(chars, n):
    seq_len = len(chars)
    if n > seq_len or n <= 0 or seq_len == 0:
        return []

    index_pairs = []
    for i in range(seq_len - n + 1):
        index_pairs.append((i, i + n))
    return index_pairs


def get_label_BI(index, cate=None):
    prefix = 'B' if index == 0 else 'I'
    suffix = '' if cate is None else '-' + cate
    return prefix + suffix


def get_label_BIES(index, last, cate=None):
    if last == 0:
        prefix = 'S'
    else:
        if index == 0:
            prefix = 'B'
        elif index < last:
            prefix = 'I'
        else:
            prefix = 'E'
    suffix = '' if not cate else '-' + cate
    return '{}{}'.format(prefix, suffix)
