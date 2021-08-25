import sys
import torch

import classifiers.sequence_tagger
import common
import constants
from data_loaders import data_loader, segmentation_data_loader
from evaluators.common import (AccuracyEvaluator, FMeasureEvaluator,
                               DoubleFMeasureEvaluator)
import models.tagger
from trainers import trainer
from trainers.trainer import Trainer
import util


class TaggerTrainerBase(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)

    def show_data_info(self, data_type):
        dic = self.dic_val if data_type == 'valid' else self.dic
        self.log('### {} dic'.format(data_type))
        self.log('Number of tokens: {}'.format(
            len(dic.tables[constants.UNIGRAM])))
        if dic.has_table(constants.BIGRAM):
            self.log('Number of bigrams: {}'.format(
                len(dic.tables[constants.BIGRAM])))
        self.log()

    def show_training_data(self):
        train = self.train
        valid = self.valid

        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train.inputs[0][0],
                                               train.inputs[0][-1]))
        self.log('# train_gold: {} ... {}\n'.format(train.outputs[0][0],
                                                    train.outputs[0][-1]))
        t2i_tmp = list(self.dic.tables[constants.UNIGRAM].str2id.items())
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10],
                                                  t2i_tmp[len(t2i_tmp) - 10:]))

        if self.dic.has_table(constants.BIGRAM):
            b2i_tmp = list(self.dic.tables[constants.BIGRAM].str2id.items())
            self.log('# bigram2id: {} ... {}\n'.format(
                b2i_tmp[:10], b2i_tmp[len(b2i_tmp) - 10:]))
        if self.dic.has_trie(constants.CHUNK):
            id2chunk = self.dic.tries[constants.CHUNK].id2chunk
            n_chunks = len(self.dic.tries[constants.CHUNK])
            c2i_head = [(id2chunk[i], i) for i in range(0, min(10, n_chunks))]
            c2i_tail = [(id2chunk[i], i)
                        for i in range(max(0, n_chunks - 10), n_chunks)]
            self.log('# chunk2id: {} ... {}\n'.format(c2i_head, c2i_tail))
        if self.dic.has_trie(constants.SUBWORD):
            id2subword = self.dic.tries[constants.SUBWORD].id2subword
            n_subwords = len(self.dic.tries[constants.SUBWORD])
            c2i_head = [(id2subword[i], i)
                        for i in range(0, min(10, n_subwords))]
            c2i_tail = [(id2subword[i], i)
                        for i in range(max(0, n_subwords - 10), n_subwords)]
            self.log('# subword2id: {} ... {}\n'.format(c2i_head, c2i_tail))
        if self.dic.has_trie(constants.CC):
            id2cc = self.dic.tries[constants.CC].id2cc
            n_ccs = len(self.dic.tries[constants.CC])
            c2i_head = [(id2cc[i], i) for i in range(0, min(10, n_ccs))]
            c2i_tail = [(id2cc[i], i)
                        for i in range(max(0, n_ccs - 10), n_ccs)]
            self.log('# cc2id: {} ... {}\n'.format(c2i_head, c2i_tail))
        if self.dic.has_table(constants.SEG_LABEL):
            id2seg = {
                v: k
                for k, v in self.dic.tables[
                    constants.SEG_LABEL].str2id.items()
            }
            self.log('# label_set: {}\n'.format(id2seg))

        # attr_indexes = common.get_attribute_values(self.args.attr_indexes)
        # for i in range(len(attr_indexes)):
        #     if self.dic.has_table(constants.ATTR_LABEL(i)):
        #         id2attr = {
        #             v: k
        #             for k, v in self.dic.tables[constants.ATTR_LABEL(
        #                 i)].str2id.items()
        #         }
        #         self.log('# {}-th attribute labels: {}\n'.format(i, id2attr))

        self.report('[INFO] vocab: {}'.format(
            len(self.dic.tables[constants.UNIGRAM])))
        self.report('[INFO] data length: train={} valid={}'.format(
            len(train.inputs[0]),
            len(valid.inputs[0]) if valid else 0))

    def gen_inputs(self, data, ids, evaluate=True):
        # create tensor on cpu
        device = torch.device(
            constants.CUDA_DEVICE) if self.args.gpu >= 0 else torch.device(
                constants.CPU_DEVICE)

        us = [
            torch.tensor(data.inputs[0][j], dtype=int, device=device)
            for j in ids
        ]
        bs = [
            torch.tensor(data.inputs[1][j], dtype=int, device=device)
            for j in ids
        ] if data.inputs[1] else None
        es = [
            torch.tensor(data.inputs[2][j], dtype=int, device=device)
            for j in ids
        ] if data.inputs[2] else None

        # if self.hparams['feature_template']:
        #     if self.args.batch_feature_extraction:
        #         fs = [data.featvecs[j] for j in ids]
        #     else:
        #         us_int = [data.inputs[0][j] for j in ids]
        #         fs = self.feat_extractor.extract_features(
        #             us_int, self.dic.tries[constants.CHUNK])
        # else:
        #     fs = None
        fs = None

        ls = [
            torch.tensor(data.outputs[0][j], dtype=int, device=device)
            for j in ids
        ] if evaluate else None

        if evaluate:
            return us, bs, es, fs, ls
        else:
            return us, bs, es, fs

    def decode(self, rdata, file=sys.stdout):
        n_ins = len(rdata.inputs[0])
        org_tokens = rdata.orgdata[0]
        org_attrs = rdata.orgdata[1] if len(rdata.orgdata) > 1 else None

        timer = util.Timer()
        timer.start()
        for ids in trainer.batch_generator(n_ins,
                                           batch_size=self.args.batch_size,
                                           shuffle=False):
            inputs = self.gen_inputs(rdata, ids, evaluate=False)
            ot = [org_tokens[j] for j in ids]
            oa = [org_attrs[j] for j in ids] if org_attrs else None
            self.decode_batch(*inputs, org_tokens=ot, org_attrs=oa, file=file)
        timer.stop()

        print(
            'Parsed %d sentences. Elapsed time: %.4f sec (total) / %.4f sec (per sentence)'
            % (n_ins, timer.elapsed, timer.elapsed / n_ins),
            file=sys.stderr)

    def run_interactive_mode(self):
        print('Please input text or type \'q\' to quit this mode:')
        while True:
            line = sys.stdin.readline().rstrip(' \t\n')
            if len(line) == 0:
                continue
            elif line == 'q':
                break

            rdata = self.data_loader.parse_commandline_input(line, self.dic)
            # if self.hparams['feature_template']:
            #     rdata.featvecs = self.feat_extractor.extract_features(ws[0], self.dic.tries[constants.CHUNK])
            inputs = self.gen_inputs(rdata, [0], evaluate=False)
            ot = rdata.orgdata[0]
            oa = rdata.orgdata[1] if len(rdata.orgdata) > 1 else None

            self.decode_batch(*inputs, org_tokens=ot, org_attrs=oa)

    def run_eval_mode(self):
        if self.args.evaluation_method == 'stat_test':
            self.run_eval_mode_for_stat_test()
        else:
            super().run_eval_mode()

    def run_eval_mode_for_stat_test(self):
        classifier = self.classifier.copy()
        self.update_model(classifier=classifier, dic=self.dic)
        classifier.change_dropout_ratio(0)

        data = self.test
        n_ins = len(data.inputs[0])
        n_sen = 0
        total_counts = None

        for ids in trainer.batch_generator(n_ins,
                                           batch_size=self.args.batch_size,
                                           shuffle=False):
            inputs = self.gen_inputs(data, ids)
            xs = inputs[0]
            n_sen += len(xs)
            gls = inputs[self.label_begin_index]

            with torch.no_grad():
                ret = classifier.predictor(*inputs)
            pls = ret[1]

            for gl, pl in zip(gls, pls):
                for gli, pli in zip(gl, pl):
                    print('{}'.format(1 if int(gli) == int(pli) else 0))
                print()

        print('Finished', n_sen, file=sys.stderr)

    def convert_to_valid_BIES_seq(self, y_str):
        y_str2 = y_str.copy()

        for j in range(len(y_str)):
            prv = y_str2[j - 1] if j >= 1 else None
            crt = y_str[j]
            nxt = y_str[j + 1] if j <= len(y_str) - 2 else None

            # invalid I or E assigned for a first token
            if ((crt[0] == 'I' or crt[0] == 'E') and
                (prv is None or prv[0] == 'S' or prv[0] == 'E' or
                 (self.task == constants.TASK_SEGTAG and crt[2:] != prv[2:]))):
                if nxt == 'I' + crt[1:] or nxt == 'E' + crt[1:]:
                    y_str2[j] = 'B' + crt[1:]
                else:
                    y_str2[j] = 'S' + crt[1:]

            # invalid B or I assignied for a last token
            elif (
                (crt[0] == 'B' or crt[0] == 'I') and
                (nxt is None or nxt[0] == 'B' or nxt[0] == 'S' or
                 (self.task == constants.TASK_SEGTAG and crt[2:] != nxt[2:]))):
                if (prv == 'B' + crt[1:] or prv == 'I' + crt[1:]):
                    y_str2[j] = 'E' + crt[1:]
                else:
                    y_str2[j] = 'S' + crt[1:]

            # if crt != y_str2[j]:
            #     print('{} {} [{}] {} -> {}'.format(j, prv, crt, nxt, y_str2[j]))

        return y_str2

    def decode_batch(self,
                     *inputs,
                     org_tokens=None,
                     org_attrs=None,
                     file=sys.stdout):
        ys = self.classifier.decode(*inputs)
        id2label = (
            self.dic.tables[constants.SEG_LABEL if common.
                            is_segmentation_task(self.task) else constants.
                            ATTR_LABEL(0)].id2str)

        if not org_attrs:
            org_attrs = [None] * len(org_tokens)

        for x_str, a_str, y in zip(org_tokens, org_attrs, ys):
            y_str = [id2label[int(yi)] for yi in y]
            y_str = self.convert_to_valid_BIES_seq(y_str)

            if self.task == constants.TASK_TAG:
                if a_str:
                    res = [
                        '{}{}{}{}{}'.format(xi_str,
                                            self.args.output_attr_delim,
                                            ai_str,
                                            self.args.output_attr_delim,
                                            yi_str)
                        for xi_str, ai_str, yi_str in zip(x_str, a_str, y_str)
                    ]
                else:
                    res = [
                        '{}{}{}'.format(xi_str, self.args.output_attr_delim,
                                        yi_str)
                        for xi_str, yi_str in zip(x_str, y_str)
                    ]

                if self.args.output_data_format == 'wl':
                    res.append('')
                res = self.args.output_token_delim.join(res)

            elif self.task == constants.TASK_SEG:
                res = [
                    '{}{}'.format(
                        xi_str, self.args.output_token_delim if
                        (yi_str.startswith('E')
                         or yi_str.startswith('S')) else '')
                    for xi_str, yi_str in zip(x_str, y_str)
                ]
                res = ''.join(res).rstrip(' ')

            elif self.task == constants.TASK_SEGTAG:
                res = [
                    '{}{}'.format(xi_str,
                                  (self.args.output_attr_delim + yi_str[2:] +
                                   self.args.output_token_delim) if
                                  (yi_str.startswith('E-')
                                   or yi_str.startswith('S-')) else '')
                    for xi_str, yi_str in zip(x_str, y_str)
                ]
                res = ''.join(res).rstrip(' ')

            else:
                print('Error: Invalid decode type', file=self.logger)
                sys.exit()

            print(res, file=file)

    def load_external_dictionary(self):
        if self.args.external_dic_path:
            edic_path = self.args.external_dic_path
            self.dic = segmentation_data_loader.load_external_dictionary(
                edic_path, dic=self.dic)
            self.log('Load external dictionary: {}'.format(edic_path))
            self.log('Num of unigrams: {}'.format(
                len(self.dic.tables[constants.UNIGRAM])))
            self.log('Num of chunks: {}'.format(
                len(self.dic.tries[constants.CHUNK])))
            self.log('')


class TaggerTrainer(TaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.unigram_embed_model = None
        self.bigram_embed_model = None
        self.label_begin_index = 4

    def load_external_embedding_models(self):
        if self.args.unigram_embed_model_path:
            self.unigram_embed_model = trainer.load_embedding_model(
                self.args.unigram_embed_model_path)
        if self.args.bigram_embed_model_path:
            self.bigram_embed_model = trainer.load_embedding_model(
                self.args.bigram_embed_model_path)

    def init_model(self):
        super().init_model()
        if self.unigram_embed_model or self.bigram_embed_model:
            self.classifier.load_pretrained_embedding_layer(
                self.dic,
                self.unigram_embed_model,
                self.bigram_embed_model,
                finetuning=True)

    def load_model(self):
        super().load_model()
        if self.args.execute_mode == 'train':
            if 'embed_dropout' in self.hparams:
                self.classifier.change_embed_dropout_ratio(
                    self.hparams['embed_dropout'])
            if 'rnn_dropout' in self.hparams:
                self.classifier.change_rnn_dropout_ratio(
                    self.hparams['rnn_dropout'])
            if 'mlp_dropout' in self.hparams:
                self.classifier.change_mlp_dropout_ratio(
                    self.hparams['mlp_dropout'])
        else:
            self.classifier.change_dropout_ratio(0)

    def update_model(self, classifier=None, dic=None, train=False):
        if not classifier:
            classifier = self.classifier
        if not dic:
            dic = self.dic

        if (self.args.execute_mode == 'train'
                or self.args.execute_mode == 'eval'
                or self.args.execute_mode == 'decode'):
            classifier.grow_embedding_layers(dic,
                                             self.unigram_embed_model,
                                             self.bigram_embed_model,
                                             train=train)
            classifier.grow_inference_layers(dic)

    def init_hyperparameters(self):
        if self.unigram_embed_model:
            pretrained_unigram_embed_size = self.unigram_embed_model.wv.syn0[
                0].shape[0]
        else:
            pretrained_unigram_embed_size = 0

        if self.bigram_embed_model:
            pretrained_bigram_embed_size = self.bigram_embed_model.wv.syn0[
                0].shape[0]
        else:
            pretrained_bigram_embed_size = 0

        self.hparams = {
            'pretrained_unigram_embed_size': pretrained_unigram_embed_size,
            'pretrained_bigram_embed_size': pretrained_bigram_embed_size,
            'pretrained_embed_usage': self.args.pretrained_embed_usage,
            'unigram_embed_size': self.args.unigram_embed_size,
            'bigram_embed_size': self.args.bigram_embed_size,
            # 'attr1_embed_size': self.args.attr1_embed_size,
            'rnn_unit_type': self.args.rnn_unit_type,
            'rnn_bidirection': self.args.rnn_bidirection,
            'rnn_batch_first': self.args.rnn_batch_first,
            'rnn_n_layers': self.args.rnn_n_layers,
            'rnn_hidden_size': self.args.rnn_hidden_size,
            'mlp_n_layers': self.args.mlp_n_layers,
            'mlp_hidden_size': self.args.mlp_hidden_size,
            'inference_layer': self.args.inference_layer,
            'embed_dropout': self.args.embed_dropout,
            'rnn_dropout': self.args.rnn_dropout,
            'mlp_dropout': self.args.mlp_dropout,
            # 'feature_template': self.args.feature_template,
            'task': self.args.task,
            'lowercasing': self.args.lowercasing,
            'normalize_digits': self.args.normalize_digits,
            'token_freq_threshold': self.args.token_freq_threshold,
            'token_max_vocab_size': self.args.token_max_vocab_size,
            'bigram_freq_threshold': self.args.bigram_freq_threshold,
            'bigram_max_vocab_size': self.args.bigram_max_vocab_size
        }

        self.log('Init hyperparameters')
        self.log('# arguments')
        for k, v in self.args.__dict__.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')

    def load_hyperparameters(self, hparams_path):
        hparams = {}
        with open(hparams_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue

                kv = line.split('=')
                key = kv[0]
                val = kv[1]

                if (key == 'pretrained_unigram_embed_size'
                        or key == 'pretrained_bigram_embed_size'
                        or key == 'unigram_embed_size'
                        or key == 'bigram_embed_size'
                        # or key == 'attr1_embed_size'
                        # or key == 'additional_feat_size'
                        or key == 'rnn_n_layers' or key == 'rnn_hidden_size' or
                        key == 'mlp_n_layers' or key == 'mlp_hidden_size' or
                        key == 'token_freq_threshold' or
                        key == 'token_max_vocab_size' or
                        key == 'bigram_freq_threshold' or
                        key == 'bigram_max_vocab_size' or
                        key == 'chunk_freq_threshold' or key
                        == 'chunk_max_vocab_size'):
                    val = int(val)

                elif (key == 'embed_dropout' or key == 'rnn_dropout'
                      or key == 'mlp_dropout'):
                    val = float(val)

                elif (key == 'rnn_batch_first' or key == 'rnn_bidirection'
                      or key == 'lowercasing' or key == 'normalize_digits'):
                    val = (val.lower() == 'true')

                hparams[key] = val

        self.hparams = hparams
        self.task = self.hparams['task']

        if (self.args.execute_mode != 'interactive'
                and self.hparams['pretrained_unigram_embed_size'] > 0
                and not self.unigram_embed_model):
            self.log('Error: unigram embedding model is necessary.')
            sys.exit()

        if self.unigram_embed_model:
            pretrained_unigram_embed_size = self.unigram_embed_model.wv.syn0[
                0].shape[0]
            if hparams[
                    'pretrained_unigram_embed_size'] != pretrained_unigram_embed_size:
                self.log(
                    'Error: pretrained_unigram_embed_size and size (dimension) of loaded embedding model are conflicted.'
                    .format(hparams['pretrained_unigram_embed_size'],
                            pretrained_unigram_embed_size))
                sys.exit()

        if self.bigram_embed_model:
            pretrained_bigram_embed_size = self.bigram_embed_model.wv.syn0[
                0].shape[0]
            if hparams[
                    'pretrained_bigram_embed_size'] != pretrained_bigram_embed_size:
                self.log(
                    'Error: pretrained_bigram_embed_size and size (dimension) of loaded embedding model are conflicted.'
                    .format(hparams['pretrained_bigram_embed_size'],
                            pretrained_bigram_embed_size))
                sys.exit()

    def setup_data_loader(self):
        # attr_indexes = common.get_attribute_values(self.args.attr_indexes)

        if self.task == constants.TASK_SEG:
            self.data_loader = segmentation_data_loader.SegmentationDataLoader(
                # token_index=self.args.token_index,
                # attr_indexes=attr_indexes,
                # attr_depths=common.get_attribute_values(
                #     self.args.attr_depths, len(attr_indexes)),
                # attr_target_labelsets=common.get_attribute_labelsets(
                #     self.args.attr_target_labelsets, len(attr_indexes)),
                # attr_delim=self.args.attr_delim,
                use_bigram=(self.hparams['bigram_embed_size'] > 0),
                # use_chunk_trie=(True if self.hparams['feature_template'] else
                #                 False),
                use_chunk_trie=False,
                bigram_max_vocab_size=self.hparams['bigram_max_vocab_size'],
                bigram_freq_threshold=self.hparams['bigram_freq_threshold'],
                unigram_vocab=(self.unigram_embed_model.wv
                               if self.unigram_embed_model else set()),
                bigram_vocab=(self.bigram_embed_model.wv
                              if self.bigram_embed_model else set()),
            )

    def setup_classifier(self):
        dic = self.dic
        hparams = self.hparams

        n_vocab = len(dic.tables['unigram'])
        unigram_embed_size = hparams['unigram_embed_size']

        if 'bigram_embed_size' in hparams and hparams['bigram_embed_size'] > 0:
            bigram_embed_size = hparams['bigram_embed_size']
            n_bigrams = len(dic.tables[constants.BIGRAM])
        else:
            bigram_embed_size = n_bigrams = 0

        if 'pretrained_unigram_embed_size' in hparams and hparams[
                'pretrained_unigram_embed_size'] > 0:
            pretrained_unigram_embed_size = hparams[
                'pretrained_unigram_embed_size']
        else:
            pretrained_unigram_embed_size = 0

        if 'pretrained_bigram_embed_size' in hparams and hparams[
                'pretrained_bigram_embed_size'] > 0:
            pretrained_bigram_embed_size = hparams[
                'pretrained_bigram_embed_size']
        else:
            pretrained_bigram_embed_size = 0

        if 'pretrained_embed_usage' in hparams:
            pretrained_embed_usage = models.util.ModelUsage.get_instance(
                hparams['pretrained_embed_usage'])
        else:
            pretrained_embed_usage = models.util.ModelUsage.NONE

        if common.is_segmentation_task(self.task):
            n_label = len(dic.tables[constants.SEG_LABEL])
            n_labels = [n_label]
            # attr1_embed_size = n_attr1 = 0

        # else:
        #     n_labels = []
        #     for i in range(3):  # tmp
        #         if constants.ATTR_LABEL(i) in dic.tables:
        #             n_label = len(dic.tables[constants.ATTR_LABEL(i)])
        #             n_labels.append(n_label)

        #     if 'attr1_embed_size' in hparams and hparams[
        #             'attr1_embed_size'] > 0:
        #         attr1_embed_size = hparams['attr1_embed_size']
        #         n_attr1 = n_labels[1] if len(n_labels) > 1 else 0
        #     else:
        #         attr1_embed_size = n_attr1 = 0

        if (pretrained_embed_usage == models.util.ModelUsage.ADD
                or pretrained_embed_usage == models.util.ModelUsage.INIT):
            if pretrained_unigram_embed_size > 0 and pretrained_unigram_embed_size != unigram_embed_size:
                print(
                    'Error: pre-trained and random initialized unigram embedding vectors must be the same size (dimension) for {} operation: d1={}, d2={}'
                    .format(hparams['pretrained_embed_usage'],
                            pretrained_unigram_embed_size, unigram_embed_size),
                    file=sys.stderr)
                sys.exit()

            if pretrained_bigram_embed_size > 0 and pretrained_bigram_embed_size != bigram_embed_size:
                print(
                    'Error: pre-trained and random initialized bigram embedding vectors must be the same size (dimension) for {} operation: d1={}, d2={}'
                    .format(hparams['pretrained_embed_usage'],
                            pretrained_bigram_embed_size, bigram_embed_size),
                    file=sys.stderr)
                sys.exit()

        predictor = models.tagger.construct_RNNTagger(
            n_vocab=n_vocab,
            unigram_embed_size=unigram_embed_size,
            n_bigrams=n_bigrams,
            bigram_embed_size=bigram_embed_size,
            # n_attrs=n_attr1,
            # attr_embed_size=attr1_embed_size,
            n_chunks=0,
            chunk_embed_size=0,
            n_subwords=0,
            subword_embed_size=0,
            n_ccs=0,
            cc_embed_size=0,
            rnn_unit_type=hparams['rnn_unit_type'],
            rnn_bidirection=hparams['rnn_bidirection'],
            rnn_batch_first=hparams['rnn_batch_first'],
            rnn_n_layers=hparams['rnn_n_layers'],
            rnn_hidden_size=hparams['rnn_hidden_size'],
            rnn_n_layers2=hparams['rnn_n_layers2']
            if 'rnn_n_layers2' in hparams else 0,
            rnn_hidden_size2=hparams['rnn_hidden_size2']
            if 'rnn_hidden_size2' in hparams else 0,
            rnn_n_layers3=hparams['rnn_n_layers3']
            if 'rnn_n_layers3' in hparams else 0,
            rnn_hidden_size3=hparams['rnn_hidden_size3']
            if 'rnn_hidden_size3' in hparams else 0,
            mlp_n_layers=hparams['mlp_n_layers'],
            mlp_hidden_size=hparams['mlp_hidden_size'],
            n_labels=n_labels[0],
            use_crf=hparams['inference_layer'] == 'crf',
            # feat_size=hparams['additional_feat_size'],
            mlp_additional_hidden_size=0,
            rnn_dropout=hparams['rnn_dropout'],
            embed_dropout=hparams['embed_dropout']
            if 'embed_dropout' in hparams else 0.0,
            mlp_dropout=hparams['mlp_dropout'],
            pretrained_unigram_embed_size=pretrained_unigram_embed_size,
            pretrained_bigram_embed_size=pretrained_bigram_embed_size,
            pretrained_embed_usage=pretrained_embed_usage,
        )

        self.classifier = classifiers.sequence_tagger.SequenceTagger(
            predictor, task=self.task)

    def gen_vocabs(self):
        if not self.task == constants.TASK_SEG:
            return

        train_path = self.args.path_prefix + self.args.train_data
        char_table = self.dic.tables[constants.UNIGRAM]
        vocabs = self.data_loader.gen_vocabs(
            train_path, char_table, data_format=self.args.input_data_format)
        return vocabs

    def setup_evaluator(self, evaluator=None):
        evaluator1 = None
        if self.task == constants.TASK_SEG:
            if self.args.evaluation_method == 'normal':
                evaluator1 = FMeasureEvaluator(
                    self.dic.tables[constants.SEG_LABEL].id2str)

        if not evaluator:
            self.evaluator = evaluator1
        else:
            evaluator = evaluator1
