import numpy as np
import sys
import torch

import classifiers.mutant_sequence_tagger
import common
import constants
# from data_loaders import segmentation_data_loader, tagging_data_loader
from data_loaders import segmentation_data_loader
import dictionary
from evaluators.mutant_unit_segmenter_evaluator import AccuracyEvaluatorForAttention, MutantSegmenterEvaluator, MutantTaggerEvaluator
import models.tagger
import models.util
from trainers import trainer
from trainers.tagger_trainer import TaggerTrainerBase


def get_num_candidates(mask_pairs, n_tokens):
    ncand = [0] * n_tokens
    for i, j in mask_pairs:
        ncand[i] += 1
    return ncand


# character cluster mutant segmenter/tagger
class MutantUnitSegmenterTrainer(TaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.unigram_embed_model = None
        self.bigram_embed_model = None
        self.cc_embed_model = None
        self.label_begin_index = 6

    def load_external_embedding_models(self):
        if self.args.unigram_embed_model_path:
            self.unigram_embed_model = trainer.load_embedding_model(
                self.args.unigram_embed_model_path)
        if self.args.bigram_embed_model_path:
            self.bigram_embed_model = trainer.load_embedding_model(
                self.args.bigram_embed_model_path)
        if self.args.cc_embed_model_path:
            self.cc_embed_model = trainer.load_embedding_model(
                self.args.cc_embed_model_path)

    def init_model(self):
        super().init_model()
        if self.unigram_embed_model or self.bigram_embed_model or self.cc_embed_model:
            self.classifier.load_pretrained_embedding_layer(
                self.dic,
                self.unigram_embed_model,
                self.bigram_embed_model,
                self.cc_embed_model,
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
            if 'biaffine_dropout' in self.hparams:
                self.classifier.change_biaffine_dropout_ratio(
                    self.hparams['biaffine_dropout'])
            if 'cc_vector_dropout' in self.hparams:
                self.classifier.change_cc_vector_dropout_ratio(
                    self.hparams['cc_vector_dropout'])
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
            classifier.grow_embedding_layers(
                dic,
                self.unigram_embed_model,
                self.bigram_embed_model,
                self.cc_embed_model,
                # train=train,
                train=(self.args.execute_mode == 'train'),
                fasttext=self.args.gen_oov_cc_for_test)
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

        if self.cc_embed_model:
            pretrained_cc_embed_size = self.cc_embed_model.wv.syn0[0].shape[0]
        else:
            pretrained_cc_embed_size = 0

        self.hparams = {
            'pretrained_unigram_embed_size': pretrained_unigram_embed_size,
            'pretrained_bigram_embed_size': pretrained_bigram_embed_size,
            'pretrained_cc_embed_size': pretrained_cc_embed_size,
            'pretrained_embed_usage': self.args.pretrained_embed_usage,
            'cc_pooling_type': self.args.cc_pooling_type.upper(),
            'biaffine_type': self.args.biaffine_type,
            'use_gold_cc': self.args.use_gold_cc,
            'unuse_nongold_cc': self.args.unuse_nongold_cc,
            'use_unknown_pretrained_cc':
            not self.args.ignore_unknown_pretrained_cc,
            'min_cc_len': self.args.min_cc_len,
            'max_cc_len': self.args.max_cc_len,
            'unigram_embed_size': self.args.unigram_embed_size,
            'bigram_embed_size': self.args.bigram_embed_size,
            'cc_embed_size': self.args.cc_embed_size,
            'attr1_embed_size': self.args.attr1_embed_size,
            'rnn_unit_type': self.args.rnn_unit_type,
            'rnn_bidirection': self.args.rnn_bidirection,
            'rnn_batch_first': self.args.rnn_batch_first,
            'rnn_n_layers': self.args.rnn_n_layers,
            'rnn_hidden_size': self.args.rnn_hidden_size,
            'rnn_n_layers2': self.args.rnn_n_layers2,
            'rnn_hidden_size2': self.args.rnn_hidden_size2,
            'mlp_n_layers': self.args.mlp_n_layers,
            'mlp_hidden_size': self.args.mlp_hidden_size,
            'inference_layer': self.args.inference_layer,
            'embed_dropout': self.args.embed_dropout,
            'rnn_dropout': self.args.rnn_dropout,
            'biaffine_dropout': self.args.biaffine_dropout,
            'cc_vector_dropout': self.args.cc_vector_dropout,
            'mlp_dropout': self.args.mlp_dropout,
            'feature_template': self.args.feature_template,
            'task': self.args.task,
            'tagging_unit': self.args.tagging_unit,
            'token_freq_threshold': self.args.token_freq_threshold,
            'token_max_vocab_size': self.args.token_max_vocab_size,
            'bigram_freq_threshold': self.args.bigram_freq_threshold,
            'bigram_max_vocab_size': self.args.bigram_max_vocab_size,
            'cc_freq_threshold': self.args.cc_freq_threshold,
            'cc_max_vocab_size': self.args.cc_max_vocab_size,
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
                        or key == 'pretrained_cc_embed_size'
                        or key == 'min_cc_len' or key == 'max_cc_len'
                        or key == 'unigram_embed_size'
                        or key == 'bigram_embed_size' or key == 'cc_embed_size'
                        or key == 'additional_feat_size'
                        or key == 'rnn_n_layers' or key == 'rnn_hidden_size'
                        or key == 'rnn_n_layers2' or key == 'rnn_hidden_size2'
                        or key == 'mlp_n_layers' or key == 'mlp_hidden_size'
                        or key == 'token_freq_threshold'
                        or key == 'token_max_vocab_size'
                        or key == 'bigram_freq_threshold'
                        or key == 'bigram_max_vocab_size'
                        or key == 'cc_freq_threshold'
                        or key == 'cc_max_vocab_size'):
                    val = int(val)

                elif (key == 'embed_dropout' or key == 'rnn_dropout'
                      or key == 'biaffine_dropout'
                      or key == 'cc_vector_dropout' or key == 'mlp_dropout'):
                    val = float(val)

                elif (key == 'rnn_batch_first' or key == 'rnn_bidirection'
                      or key == 'lowercasing' or key == 'normalize_digits'
                      or key == 'use_gold_cc' or key == 'unuse_nongold_cc'
                      or key == 'ignore_unknown_pretrained_cc'):
                    val = (val.lower() == 'true')

                hparams[key] = val

        if not 'unuse_nongold_cc' in hparams:
            hparams['unuse_nongold_cc'] = False

        self.hparams = hparams
        self.task = self.hparams['task']

        if self.args.execute_mode != 'interactive':
            if self.hparams[
                    'pretrained_unigram_embed_size'] > 0 and not self.unigram_embed_model:
                self.log('Error: unigram embedding model is necessary.')
                sys.exit()

            if self.hparams[
                    'pretrained_cc_embed_size'] > 0 and not self.cc_embed_model:
                self.log(
                    'Error: character-cluster embedding model is necessary.')
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

        if self.cc_embed_model:
            pretrained_cc_embed_size = self.cc_embed_model.wv.syn0[0].shape[0]
            if hparams['pretrained_cc_embed_size'] != pretrained_cc_embed_size:
                self.log(
                    'Error: pretrained_cc_embed_size and size (dimension) of loaded embedding model are conflicted.'
                    .format(hparams['pretrained_cc_embed_size'],
                            pretrained_cc_embed_size))
                sys.exit()

    def setup_data_loader(self):
        # TODO min/max_cc_len eliminate?
        attr_indexes = common.get_attribute_values(self.args.attr_indexes)

        self.data_loader = segmentation_data_loader.SegmentationDataLoader(
            token_index=self.args.token_index,
            attr_indexes=attr_indexes,
            attr_depths=common.get_attribute_values(self.args.attr_depths,
                                                    len(attr_indexes)),
            attr_target_labelsets=common.get_attribute_labelsets(
                self.args.attr_target_labelsets, len(attr_indexes)),
            use_bigram=(self.hparams['bigram_embed_size'] > 0),
            use_cc_trie=True,
            bigram_max_vocab_size=self.hparams['bigram_max_vocab_size'],
            bigram_freq_threshold=self.hparams['bigram_freq_threshold'],
            cc_max_vocab_size=self.hparams['cc_max_vocab_size'],
            cc_freq_threshold=self.hparams['cc_freq_threshold'],
            min_cc_len=self.hparams['min_cc_len'],
            max_cc_len=self.hparams['max_cc_len'],
            add_gold_cc=self.hparams['use_gold_cc'],
            add_nongold_cc=not self.hparams['unuse_nongold_cc'],
            add_unknown_pretrained_cc=self.
            hparams['use_unknown_pretrained_cc'],
            unigram_vocab=(self.unigram_embed_model.wv
                           if self.unigram_embed_model else set()),
            bigram_vocab=(self.bigram_embed_model.wv
                          if self.bigram_embed_model else set()),
            cc_vocab=(self.cc_embed_model.wv
                      if self.cc_embed_model else set()),
            generate_ccs=self.args.gen_oov_cc_for_test,
            trie_ext=self.dic_ext,
        )

    def load_data(self, data_type):
        super().load_data(data_type)
        if data_type == 'train':
            data = self.train
            dic = self.dic
            evaluate = True
        elif data_type == 'valid':
            data = self.valid
            dic = self.dic_val
            evaluate = True
        elif data_type == 'test':
            data = self.test
            dic = self.dic
            evaluate = True
        elif data_type == 'decode':
            data = self.decode_data
            dic = self.dic
            evaluate = False

        self.log(
            'Start cc search for {} data (min_len={}, max_len={})\n'.format(
                data_type, self.hparams['min_cc_len'],
                self.hparams['max_cc_len']))

        segmentation_data_loader.add_chunk_sequences(
            data,
            dic,
            min_len=self.hparams['min_cc_len'],
            max_len=self.hparams['max_cc_len'],
            evaluate=evaluate,
            model_type=self.hparams['cc_pooling_type'],
            chunk_type=constants.CC)

    def setup_classifier(self):
        dic = self.dic
        hparams = self.hparams

        n_vocab = len(dic.tables['unigram'])
        unigram_embed_size = hparams['unigram_embed_size']
        cc_embed_size = hparams['cc_embed_size']
        n_ccs = len(dic.tries[constants.CC]) if dic.has_trie(
            constants.CC) else 1

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

        if 'pretrained_cc_embed_size' in hparams and hparams[
                'pretrained_cc_embed_size'] > 0:
            pretrained_cc_embed_size = hparams['pretrained_cc_embed_size']
        else:
            pretrained_cc_embed_size = 0

        if 'pretrained_embed_usage' in hparams:
            pretrained_embed_usage = models.util.ModelUsage.get_instance(
                hparams['pretrained_embed_usage'])
        else:
            pretrained_embed_usage = models.util.ModelUsage.NONE

        n_label = len(dic.tables[constants.SEG_LABEL])
        n_labels = [n_label]
        attr1_embed_size = n_attr1 = 0

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

            if pretrained_cc_embed_size > 0 and pretrained_cc_embed_size != cc_embed_size:
                print(
                    'Error: pre-trained and random initialized cc embedding vectors must be the same size (dimension) for {} operation : d1={}, d2={}'
                    .format(hparams['pretrained_embed_usage'],
                            pretrained_cc_embed_size, cc_embed_size),
                    file=sys.stderr)
                sys.exit()

        predictor = models.tagger.construct_RNNTagger(
            n_vocab=n_vocab,
            unigram_embed_size=unigram_embed_size,
            n_bigrams=n_bigrams,
            bigram_embed_size=bigram_embed_size,
            n_attrs=n_attr1,
            attr_embed_size=attr1_embed_size,
            n_chunks=0,
            chunk_embed_size=0,
            n_ccs=n_ccs,
            cc_embed_size=cc_embed_size,
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
            feat_size=hparams['additional_feat_size'],
            mlp_additional_hidden_size=0,
            rnn_dropout=hparams['rnn_dropout'],
            biaffine_dropout=hparams['biaffine_dropout']
            if 'biaffine_dropout' in hparams else 0.0,
            embed_dropout=hparams['embed_dropout']
            if 'embed_dropout' in hparams else 0.0,
            mlp_dropout=hparams['mlp_dropout'],
            cc_vector_dropout=hparams['cc_vector_dropout']
            if 'cc_vector_dropout' in hparams else 0.0,
            pretrained_unigram_embed_size=pretrained_unigram_embed_size,
            pretrained_bigram_embed_size=pretrained_bigram_embed_size,
            pretrained_cc_embed_size=pretrained_cc_embed_size,
            pretrained_embed_usage=pretrained_embed_usage,
            cc_pooling_type=hparams['cc_pooling_type']
            if 'cc_pooling_type' in hparams else '',
            min_cc_len=hparams['min_cc_len'] if 'min_cc_len' in hparams else 0,
            max_cc_len=hparams['max_cc_len'] if 'max_cc_len' in hparams else 0,
            cc_loss_ratio=0.0,
            biaffine_type=hparams['biaffine_type']
            if 'biaffine_type' in hparams else '')

        self.classifier = classifiers.mutant_sequence_tagger.MutantSequenceTagger(
            predictor, task=self.task)

    def gen_vocabs(self):
        if not self.task == constants.TASK_SEG:
            return

        train_path = self.args.path_prefix + self.args.train_data
        char_table = self.dic.tables[constants.UNIGRAM]

        if self.dic_org:  # test
            dics = [self.dic_org, self.dic]
        elif self.dic_dev:  # dev
            dics = [self.dic, self.dic_dev]
        else:  # train
            dics = [self.dic]

        vocabs = self.data_loader.gen_vocabs(
            train_path,
            char_table,
            *dics,
            data_format=self.args.input_data_format)

        return vocabs

    def setup_evaluator(self, evaluator=None):
        evaluator1 = None
        if self.task == constants.TASK_SEG:
            if self.args.evaluation_method == 'normal':
                evaluator1 = MutantSegmenterEvaluator(
                    self.dic.tables[constants.SEG_LABEL].id2str)

        if not evaluator:
            self.evaluator = evaluator1
        else:
            evaluator = evaluator1

    def gen_inputs(self, data, ids, evaluate=True, restrict_memory=False):
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
        cs = [
            torch.tensor(data.inputs[3][j], dtype=int, device=device)
            for j in ids
        ] if data.inputs[3] else None
        ds = [
            torch.tensor(data.inputs[4][j], dtype=int, device=device)
            for j in ids
        ] if data.inputs[4] else None

        if (self.hparams['cc_pooling_type'] == constants.CON
                or self.hparams['cc_pooling_type'] == constants.CCCON):
            feat_size = sum([
                h for h in range(self.hparams['min_cc_len'],
                                 self.hparams['max_cc_len'] + 1)
            ])
            embed_size = self.hparams['cc_embed_size']
        else:
            feat_size = 0
            embed_size = 0
            # ms = [data.inputs[6][j] for j in ids]

        use_attention = (self.hparams['cc_pooling_type'] == constants.CCAVG
                         or self.hparams['cc_pooling_type'] == constants.CCCON)

        ms = [
            segmentation_data_loader.convert_mask_matrix(
                mask=data.inputs[5][j],
                n_tokens=len(us[i]),
                n_chunks=len(cs[i]) if cs else 0,
                feat_size=feat_size,
                emb_size=embed_size,
                use_attention=use_attention,
                device=device) for i, j in enumerate(ids)
        ]

        if self.hparams['feature_template']:
            if self.args.batch_feature_extraction:
                fs = [data.featvecs[j] for j in ids]
            else:
                us_int = [data.inputs[0][j] for j in ids]
                fs = self.feat_extractor.extract_features(
                    us_int, self.dic.tries[constants.CC])
        else:
            fs = None

        gls = [
            torch.tensor(data.outputs[0][j], dtype=int, device=device)
            for j in ids
        ] if evaluate else None
        gcs = [
            torch.tensor(data.outputs[1][j], dtype=int, device=device)
            for j in ids
        ] if evaluate else None

        if evaluate:
            if self.args.evaluation_method == 'attention':
                ncands = [
                    get_num_candidates(data.inputs[5][j][0], len(us[i]))
                    for i, j in enumerate(ids)
                ]
                return us, cs, ds, ms, bs, fs, gls, gcs, ncands
            else:
                return us, cs, ds, ms, bs, fs, gls, gcs
        else:
            return us, cs, ds, ms, bs, fs

    def get_labels(self, inputs):
        if (self.args.evaluation_method == 'each_length'
                or self.args.evaluation_method == 'each_vocab'):
            return inputs[self.label_begin_index:self.label_begin_index + 1]
        else:
            return inputs[self.label_begin_index:]

    def get_outputs(self, ret):
        if (self.args.evaluation_method == 'each_length'
                or self.args.evaluation_method == 'each_vocab'):
            return ret[1:2]
        else:
            return ret[1:]

    def load_external_dictionary(self):
        if self.args.external_dic_path:
            edic_path = self.args.external_dic_path
            # attr_delim = constants.WL_ATTR_DELIM

            trie_ext = dictionary.MapTrie()
            with open(edic_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(constants.COMMENT_SYM):
                        continue

                    # arr = line.split(attr_delim)
                    # if len(arr[0]) == 0:
                    #     continue
                    if len(line) == 0:
                        continue

                    # word = arr[0]
                    word = line
                    trie_ext.get_chunk_id(word, word, update=True)

            self.dic_ext = trie_ext
            self.log('Load external dictionary: {}'.format(edic_path))
            self.log('Num of chunks: {}'.format(len(trie_ext)))
            self.log('')
