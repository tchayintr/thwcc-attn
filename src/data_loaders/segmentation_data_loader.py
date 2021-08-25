import torch
import sys

import ccs
import constants
from data_loaders import data_loader
from data_loaders.data_loader import Data, DataLoader, RestorableData
import dictionary


class SegmentationDataLoader(DataLoader):
    def __init__(
            self,
            use_bigram=False,
            use_chunk_trie=False,  # for hybrid model
            use_subword_trie=False,  # for sub-combinative model
            use_cc_trie=False,  # for mutant/combinative model
            bigram_max_vocab_size=-1,
            bigram_freq_threshold=1,
            chunk_max_vocab_size=-1,
            chunk_freq_threshold=1,
            subword_max_vocab_size=-1,
            subword_freq_threshold=1,
            cc_max_vocab_size=-1,
            cc_freq_threshold=1,
            min_chunk_len=1,
            max_chunk_len=4,
            min_subword_len=1,
            max_subword_len=4,
            min_cc_len=1,
            max_cc_len=4,
            add_gold_chunk=True,
            add_nongold_chunk=True,
            add_unknown_pretrained_chunk=True,
            add_gold_subword=True,
            add_nongold_subword=True,
            add_unknown_pretrained_subword=True,
            add_gold_cc=True,
            add_nongold_cc=True,
            add_unknown_pretrained_cc=True,
            unigram_vocab=set(),
            bigram_vocab=set(),
            chunk_vocab=set(),
            subword_vocab=set(),
            cc_vocab=set(),
            generate_ngram_chunks=False,
            generate_subwords=False,
            generate_ccs=False,
            trie_ext=None,
            trie_subword=None):
        self.use_bigram = use_bigram
        self.use_chunk_trie = use_chunk_trie
        self.use_subword_trie = use_subword_trie
        self.use_cc_trie = use_cc_trie
        self.bigram_max_vocab_size = bigram_max_vocab_size
        self.bigram_freq_threshold = bigram_freq_threshold
        self.chunk_max_vocab_size = chunk_max_vocab_size
        self.chunk_freq_threshold = chunk_freq_threshold
        self.subword_max_vocab_size = subword_max_vocab_size
        self.subword_freq_threshold = subword_freq_threshold
        self.cc_max_vocab_size = cc_max_vocab_size
        self.cc_freq_threshold = cc_freq_threshold
        self.min_chunk_len = min_chunk_len
        self.max_chunk_len = max_chunk_len
        self.min_subword_len = min_subword_len
        self.max_subword_len = max_subword_len
        self.min_cc_len = min_cc_len
        self.max_cc_len = max_cc_len
        self.add_gold_chunk = add_gold_chunk
        self.add_nongold_chunk = add_nongold_chunk
        self.add_unknown_pretrained_chunk = add_unknown_pretrained_chunk
        self.add_gold_subword = add_gold_subword
        self.add_nongold_subword = add_nongold_subword
        self.add_unknown_pretrained_subword = add_unknown_pretrained_subword
        self.add_gold_cc = add_gold_cc
        self.add_nongold_cc = add_nongold_cc
        self.add_unknown_pretrained_cc = add_unknown_pretrained_cc
        self.unigram_vocab = unigram_vocab
        self.bigram_vocab = bigram_vocab
        self.chunk_vocab = chunk_vocab
        self.subword_vocab = subword_vocab
        self.cc_vocab = cc_vocab
        self.freq_bigrams = set()
        self.freq_subwords = set()
        self.freq_chunks = set()
        self.freq_ccs = set()
        self.generate_ngram_chunks = generate_ngram_chunks
        self.generate_subwords = generate_subwords
        self.generate_ccs = generate_ccs
        self.trie_ext = trie_ext
        self.trie_subword = trie_subword

    def register_chunks(self,
                        sen,
                        unigram_seq,
                        get_chunk_id,
                        label_seq=None,
                        train=True):
        if train:
            spans_gold = get_segmentation_spans(label_seq)
            is_external_chunk = (not self.chunk_vocab
                                 and not self.generate_ngram_chunks
                                 and self.trie_ext)
            if is_external_chunk:
                for chunk in self.trie_ext.__chunks__():
                    ci = get_chunk_id(
                        chunk, chunk,
                        True) if len(chunk) <= self.max_chunk_len else None

        for n in range(self.min_chunk_len, self.max_chunk_len + 1):
            span_ngrams = data_loader.create_all_char_ngram_indexes(
                unigram_seq, n)
            cid_ngrams = [unigram_seq[span[0]:span[1]] for span in span_ngrams]
            str_ngrams = [sen[span[0]:span[1]] for span in span_ngrams]
            for span, cn, sn in zip(span_ngrams, cid_ngrams, str_ngrams):
                is_pretrained_chunk = self.chunk_vocab and sn in self.chunk_vocab.wv.vocab
                is_generable_chunk = (
                    self.chunk_vocab and self.generate_ngram_chunks and
                    (not self.trie_ext or self.trie_ext.get_chunk_id(sn) > 0)
                    and sn in self.chunk_vocab.wv  # for fasttext
                )

                if train:
                    is_gold_chunk = self.add_gold_chunk and span in spans_gold
                    is_pretrained_chunk = is_pretrained_chunk and self.add_nongold_chunk
                    pass_freq_filter = not self.freq_chunks or sn in self.freq_chunks
                    if pass_freq_filter and (is_gold_chunk
                                             or is_pretrained_chunk):
                        ci = get_chunk_id(cn, sn, True)
                else:
                    if (self.add_unknown_pretrained_chunk
                            and is_pretrained_chunk or is_generable_chunk):
                        ci = get_chunk_id(cn, sn, True)

    def register_subwords(self,
                          sen,
                          unigram_seq,
                          get_subword_id,
                          label_seq=None,
                          train=True):
        if train:
            spans_gold = get_segmentation_spans(label_seq)

        for n in range(self.min_chunk_len, self.max_chunk_len + 1):
            span_subwords = data_loader.create_all_char_ngram_indexes(
                unigram_seq, n)
            cid_subwords = [
                unigram_seq[span[0]:span[1]] for span in span_subwords
            ]
            str_subwords = [sen[span[0]:span[1]] for span in span_subwords]

        for span, cn, sn in zip(span_subwords, cid_subwords, str_subwords):
            is_pretrained_subword = self.subword_vocab and sn in self.subword_vocab.wv.vocab
            is_generable_subword = (self.subword_vocab
                                    and self.generate_subwords and
                                    (not self.trie_subword or
                                     self.trie_subword.get_subword_id(sn) > 0)
                                    and sn in self.subword_vocab.wv)

            if train:
                is_gold_subword = self.add_gold_subword
                is_pretrained_subword = is_pretrained_subword and self.add_nongold_subword
                pass_freq_filter = not self.freq_subwords or sn in self.freq_subwords
                if pass_freq_filter and (
                        is_gold_subword or is_pretrained_subword
                ) and self.trie_subword.get_subword_id(sn) > 0:
                    ci = get_subword_id(cn, sn, True)
            else:
                if (self.add_unknown_pretrained_subword
                        and is_pretrained_subword or is_generable_subword
                        and self.trie_subword.get_subword_id(sn) > 0):
                    ci = get_subword_id(cn, sn, True)

    def register_ccs(self,
                     sen,
                     unigram_seq,
                     get_cc_id,
                     label_seq=None,
                     cc_extractor=None,
                     train=True):
        if not cc_extractor:
            cc_extractor = init_cc_extractor(constants.CHAR_CLUSTERS)

        span_ccs = cc_extractor.create_all_char_cluster_indexes(
            sen, self.max_cc_len)
        cid_ccs = [unigram_seq[span[0]:span[1]] for span in span_ccs]
        str_ccs = [sen[span[0]:span[1]] for span in span_ccs]

        for span, cn, sn in zip(span_ccs, cid_ccs, str_ccs):
            is_pretrained_cc = self.cc_vocab and sn in self.cc_vocab.wv.vocab
            is_generable_cc = (self.cc_vocab and self.generate_ccs
                               and (not self.trie_ext
                                    or self.trie_ext.get_cc_id(sn) > 0)
                               and sn in self.cc_vocab.wv)

            if train:
                is_gold_cc = self.add_gold_cc
                is_pretrained_cc = is_pretrained_cc and self.add_nongold_cc
                pass_freq_filter = not self.freq_ccs or sn in self.freq_ccs
                if pass_freq_filter and (is_gold_cc or is_pretrained_cc):
                    ci = get_cc_id(cn, sn, True)
            else:
                if (self.add_unknown_pretrained_cc and is_pretrained_cc
                        or is_generable_cc):
                    ci = get_cc_id(cn, sn, True)

    def load_gold_data(self, path, data_format, dic=None, train=True):
        if data_format == constants.SL_FORMAT:
            if self.bigram_freq_threshold > 1 or self.bigram_max_vocab_size > 0:
                self.freq_bigrams = self.get_frequent_bigrams_SL(
                    path, self.bigram_freq_threshold,
                    self.bigram_max_vocab_size)

            if self.chunk_freq_threshold > 1 or self.chunk_max_vocab_size > 0:
                self.freq_chunks = self.get_frequent_ngrams_SL(
                    path, self.chunk_freq_threshold, self.chunk_max_vocab_size,
                    self.min_chunk_len, self.max_chunk_len)

            if self.cc_freq_threshold > 1 or self.cc_max_vocab_size > 0:
                self.freq_ccs = self.get_frequent_ccs_SL(
                    path, self.cc_freq_threshold, self.cc_max_vocab_size,
                    self.min_cc_len, self.max_cc_len)

            data, dic = self.load_gold_data_SL(path, dic, train)

        return data, dic

    def load_decode_data(self, path, data_format, dic=None):
        return self.load_decode_data_SL(path, dic)

    """
    Read data with SL (one Sentence in one Line) format.
    The following format is expected:
      word1 word2 ... wordn
    """

    def load_gold_data_SL(self, path, dic=None, train=True):
        if not dic:
            dic = init_dictionary(use_bigram=self.use_bigram,
                                  use_chunk_trie=self.use_chunk_trie,
                                  use_subword_trie=self.use_subword_trie,
                                  use_cc_trie=self.use_cc_trie)

        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_bigram_id = dic.tables[
            constants.BIGRAM].get_id if self.use_bigram else None
        get_chunk_id = dic.tries[
            constants.CHUNK].get_chunk_id if self.use_chunk_trie else None
        get_subword_id = dic.tries[
            constants.
            SUBWORD].get_subword_id if self.use_subword_trie else None
        get_cc_id = dic.tries[
            constants.CC].get_cc_id if self.use_cc_trie else None
        get_seg_id = dic.tables[constants.SEG_LABEL].get_id

        token_seqs = []
        bigram_seqs = []
        seg_seqs = []  # list of segmentation sequences

        ins_cnt = 0

        with open(path) as f:
            for line in f:
                line = self.normalize_input_line(line)
                if len(line) <= 1:
                    continue

                # elif line[0] == constants.COMMENT_SYM:
                #     continue

                entries = line.split(constants.SL_TOKEN_DELIM)
                uni_seq = []
                bi_seq = []
                seg_seq = []
                raw_sen = ''

                for entry in entries:
                    token = entry
                    tlen = len(token)
                    raw_sen += token

                    uni_seq.extend(
                        [get_unigram_id(token[i], True) for i in range(tlen)])
                    seg_seq.extend([
                        get_seg_id(data_loader.get_label_BIES(i, tlen - 1),
                                   update=train) for i in range(tlen)
                    ])

                if self.use_bigram:
                    str_bigrams = data_loader.create_all_char_ngrams(
                        raw_sen, 2)
                    str_bigrams.append('{}{}'.format(raw_sen[-1],
                                                     constants.EOS))
                    bi_seq = [
                        get_bigram_id(sb,
                                      update=self.to_be_registered(
                                          sb, train, self.freq_bigrams,
                                          self.bigram_vocab))
                        for sb in str_bigrams
                    ]

                if self.use_chunk_trie:
                    self.register_chunks(raw_sen,
                                         uni_seq,
                                         get_chunk_id,
                                         seg_seq,
                                         train=train)

                if self.use_subword_trie:
                    self.register_subwords(raw_sen,
                                           uni_seq,
                                           get_subword_id,
                                           seg_seq,
                                           train=train)

                if self.use_cc_trie:
                    cc_extractor = init_cc_extractor(constants.CHAR_CLUSTERS)
                    self.register_ccs(raw_sen,
                                      uni_seq,
                                      get_cc_id,
                                      seg_seq,
                                      cc_extractor,
                                      train=train)

                token_seqs.append(uni_seq)
                if bi_seq:
                    bigram_seqs.append(bi_seq)
                seg_seqs.append(seg_seq)

                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)

        inputs = [token_seqs]
        inputs.append(bigram_seqs if bigram_seqs else None)
        inputs.append(None)
        outputs = [seg_seqs]

        return Data(inputs, outputs), dic

    def load_decode_data_SL(self, path, dic):
        get_unigram_id = dic.tables[constants.UNIGRAM].get_id
        get_bigram_id = dic.tables[
            constants.BIGRAM].get_id if self.use_bigram else None
        get_chunk_id = dic.tries[
            constants.CHUNK].get_chunk_id if self.use_chunk_trie else None
        get_subword_id = dic.tries[
            constants.
            SUBWORD].get_subword_id if self.use_subword_trie else None
        get_cc_id = dic.tries[
            constants.CC].get_cc_id if self.use_cc_trie else None
        get_seg_id = dic.tables[constants.SEG_LABEL].get_id

        org_token_seqs = []
        token_seqs = []
        bigram_seqs = []

        ins_cnt = 0
        with open(path) as f:
            for line in f:
                line = self.normalize_input_line(line)
                if len(line) == 0:
                    continue

                # elif line[0] == constants.COMMENT_SYM:
                #     continue

                org_token_seqs.append([char for char in line])
                uni_seq = [get_unigram_id(char) for char in line]

                if self.use_bigram:
                    str_bigrams = data_loader.create_all_char_ngrams(line, 2)
                    str_bigrams.append('{}{}'.format(line[-1], constants.EOS))
                    bi_seq = [
                        get_bigram_id(sb,
                                      update=self.to_be_registered(
                                          sb, False, self.freq_bigrams,
                                          self.bigram_vocab))
                        for sb in str_bigrams
                    ]

                if self.use_chunk_trie:
                    self.register_chunks(line,
                                         uni_seq,
                                         get_chunk_id,
                                         train=False)

                if self.use_subword_trie:
                    self.register_subwords(line,
                                           uni_seq,
                                           get_subword_id,
                                           train=False)

                if self.use_cc_trie:
                    cc_extractor = init_cc_extractor(constants.CHAR_CLUSTERS)
                    self.register_ccs(line,
                                      uni_seq,
                                      get_cc_id,
                                      cc_extractor,
                                      train=False)

                token_seqs.append(uni_seq)
                if self.use_bigram:
                    bigram_seqs.append(bi_seq)
                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)

        inputs = [token_seqs]
        inputs.append(bigram_seqs if bigram_seqs else None)
        inputs.append(None)
        outputs = []
        orgdata = [org_token_seqs]

        return RestorableData(inputs, outputs, orgdata=orgdata)


def get_char_type(char):
    if len(char) != 1:
        return

    if char == '\u30fc':
        return constants.TYPE_LONG
    elif '\u0e01' <= char <= '\u0e5b':
        return constants.TYPE_THAI
    elif '\u3041' <= char <= '\u3093':
        return constants.TYPE_HIRA
    elif '\u30A1' <= char <= '\u30F4':
        return constants.TYPE_KATA
    elif '\u4e8c' <= char <= '\u9fa5':
        return constants.TYPE_KANJI
    elif '\uff10' <= char <= '\uff19' or '0' <= char <= '9':
        return constants.TYPE_DIGIT
    elif '\uff21' <= char <= '\uff5a' or 'A' <= char <= 'z':
        return constants.TYPE_ALPHA
    elif char == '\u3000' or char == ' ':
        return constants.TYPE_SPACE
    elif '!' <= char <= '~':
        return constants.TYPE_ASCII_OTHER
    else:
        return constants.TYPE_SYMBOL


def get_segmentation_spans(label_seq):
    spans = []
    first = -1
    for i, label in enumerate(label_seq):
        if label == 3:  # 'S'
            spans.append((i, i + 1))
        elif label == 0:  # 'B'
            first = i
        elif label == 2:  # 'E'
            spans.append((first, i + 1))
    return spans


def load_external_dictionary(path, dic=None):
    if not dic:
        dic = init_dictionary(use_chunk_trie=True, use_cc_trie=True)

    get_unigram_id = dic.tables[constants.UNIGRAM].get_id
    get_chunk_id = dic.tries[constants.CHUNK].get_chunk_id

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(constants.COMMENT_SYM):
                continue

            if len(line) == 0:
                continue

            word = line
            char_ids = [get_unigram_id(char, update=True) for char in word]
            word_id = get_chunk_id(char_ids, word, update=True)
    dic.create_id2strs()

    return dic


def init_dictionary(use_bigram=False,
                    use_chunk_trie=False,
                    use_subword_trie=False,
                    use_cc_trie=False):

    dic = dictionary.Dictionary()

    # unigram
    dic.create_table(constants.UNIGRAM)
    dic.tables[constants.UNIGRAM].set_unk(constants.UNK_SYMBOL)

    # segmentation label
    dic.create_table(constants.SEG_LABEL)
    for lab in constants.SEG_LABELS:
        dic.tables[constants.SEG_LABEL].get_id(lab, update=True)

    # bigram
    if use_bigram:
        dic.create_table(constants.BIGRAM)
        dic.tables[constants.BIGRAM].set_unk(constants.UNK_SYMBOL)

    # chunk
    if use_chunk_trie:
        dic.init_trie(constants.CHUNK)

    # subword
    if use_subword_trie:
        dic.init_subword_trie(constants.SUBWORD)

    # character-cluster
    if use_cc_trie:
        dic.init_cc_trie(constants.CC)

    return dic


"""
  avg / wavg
    chunk_seq:  [w_0, ..., w_{m-1}]
    gchunk_seq: [gold_index(chunk_seq, c_0), ..., gold_index(chunk_seq, c_{n-1})]
    mask_ij:      [[exist(c_0, w_0), ..., exist(c_0, w_{m-1})],
                 ...
                 [exist(c_{n_1}, w_0), ..., exist(c_{n-1}, w_{m-1})]]
  con / wcon:
    feat_seq:   [[word_id(c_0, 0), ..., word_id(c_{n-1}, 0)],
                 ...
                 [word_id(c_0, k-1), ..., word_id(c_{n-1}, k-1)]]
    gchunk_seq: [gold_index([0,...,k-1], c_0), ..., gold_index([0,...,k-1], c_{n-1})]
    mask_ik:      zero vectors for characters w/o no candidate words
"""

def add_chunk_sequences(data,
                        dic,
                        min_len=1,
                        max_len=4,
                        evaluate=True,
                        model_type=constants.AVG,
                        chunk_type=constants.CHUNK):
    is_con = model_type == constants.CON
    is_wcon = (model_type == constants.WCON or model_type == constants.SWCON
               or model_type == constants.CCCON)
    is_att_based = ((model_type == constants.WAVG)
                    or (model_type == constants.WCON or model_type
                        == constants.SWCON or model_type == constants.CCCON))
    is_con_based = ((model_type == constants.CON)
                    or (model_type == constants.WCON or model_type
                        == constants.SWCON or model_type == constants.CCCON))

    trie = dic.tries[chunk_type]
    if chunk_type == constants.CC:
        get_chunk_id_from_trie = trie.get_cc_id
    elif chunk_type == constants.SUBWORD:
        get_chunk_id_from_trie = trie.get_subword_id
    else:
        get_chunk_id_from_trie = trie.get_chunk_id

    token_seqs = data.inputs[0]
    gold_label_seqs = data.outputs[0] if evaluate else None
    gold_chunk_seqs = []
    chunk_seqs = [] if not is_con else None
    feat_seqs = [] if is_con_based else None
    feat_size = sum([h for h in range(min_len, max_len +
                                      1)]) if is_con_based else None
    masks = []

    ins_cnt = 0
    for sen_id, tseq in enumerate(token_seqs):
        if ins_cnt > 0 and ins_cnt % 100000 == 0:
            print('Processed', ins_cnt, 'sentences', file=sys.stderr)
        ins_cnt += 1

        n = len(tseq)
        gchunk_seq = [-1] * n
        feats = [[0] * n for k in range(feat_size)] if is_con_based else None
        mask_ij = [] if not is_con else None  # for biaffine
        mask_ik = [] if is_con_based else None  # for concat
        table_ikj = [[-1 for k in range(feat_size)]
                     for i in range(n)] if is_wcon else None  # for wcon

        if evaluate:
            lseq = gold_label_seqs[sen_id]
            spans_gold = get_segmentation_spans(lseq)
        else:
            lseq = spans_gold = None

        spans_found = []
        for i in range(n):
            res = trie.common_prefix_search(tseq, i, i + max_len)
            for span in res:
                if min_len == 1 or span[1] - span[0] >= min_len:
                    spans_found.append(span)

        if not is_con:
            m = len(spans_found)
            chunk_seq = [None] * m

        for j, span in enumerate(spans_found):
            is_gold_span = evaluate and span in spans_gold
            cid = get_chunk_id_from_trie(tseq[span[0]:span[1]])
            if not is_con:
                chunk_seq[j] = cid

            for i in range(span[0], span[1]):
                if not is_con:
                    mask_ij.append(
                        (i,
                         j))  # (char i, word j) has value; used for biaffine

                if is_con_based:
                    k = token_index2feat_index(i, span, min_token_len=min_len)
                    feats[k][i] = cid
                    mask_ik.append(
                        (i, k)
                    )  # (char i, feat k) has value; used for concatenation

                if is_wcon:
                    table_ikj[i][k] = j

                if is_gold_span:
                    gchunk_seq[i] = k if is_con_based else j

        if not is_con:
            chunk_seqs.append(chunk_seq)
        if is_con_based:
            feat_seqs.append(feats)
        masks.append((mask_ij, mask_ik, table_ikj))
        if evaluate:
            gold_chunk_seqs.append(gchunk_seq)

    data.inputs.append(chunk_seqs)
    data.inputs.append(feat_seqs)
    data.inputs.append(masks)
    if evaluate:
        data.outputs.append(gold_chunk_seqs)


def token_index2feat_index(ti, span, min_token_len=1):
    token_len = span[1] - span[0]
    fi = sum([i for i in range(min_token_len, token_len)]) + (span[1] - ti - 1)
    return fi


FLOAT_MIN = -100000.0  # input for exp


def convert_mask_matrix(mask,
                        n_tokens,
                        n_chunks,
                        feat_size,
                        emb_size,
                        use_attention=True,
                        device=constants.CPU_DEVICE):
    if use_attention:
        mask_val = FLOAT_MIN
        non_mask_val = 0.0
    else:
        mask_val = 0.0
        non_mask_val = 1.0

    pairs_ij = mask[0]
    pairs_ik = mask[1]
    table_ikj = mask[2]

    if pairs_ik:  # con or wcon
        mask_ik = torch.zeros((n_tokens, feat_size),
                              dtype=torch.float,
                              device=device)
        for i, k in pairs_ik:
            mask_ik[i][k] = torch.tensor(1, dtype=torch.float, device=device)
    else:
        mask_ik = None

    if pairs_ij is not None:  # ave, wave, or wcon
        mask_ij = torch.full((n_tokens, n_chunks),
                             fill_value=mask_val,
                             dtype=torch.float,
                             device=device)
        for i, j in pairs_ij:
            mask_ij[i, j] = non_mask_val
    else:
        mask_ij = None

    mask_i = None
    if use_attention:  # for softmax
        mask_i = torch.ones((n_tokens, n_chunks),
                            dtype=torch.float,
                            device=device)
        for i in range(n_tokens):
            tmp = [k1 == i for k1, k2 in pairs_ij]
            if len(tmp) == 0 or max(tmp) is False:  # not (i, *) in index_pair
                mask_i[i] = torch.zeros((n_chunks, ),
                                        dtype=torch.float,
                                        device=device)

    return (mask_ij, mask_i, mask_ik, table_ikj)


def init_cc_extractor(clusters=constants.CHAR_CLUSTERS):
    cc_extractor = ccs.CCExtractor(clusters)
    return cc_extractor
