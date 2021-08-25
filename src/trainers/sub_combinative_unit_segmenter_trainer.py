import sys
import torch

import classifiers.sub_combinative_sequence_tagger
import constants
from data_loaders import segmentation_data_loader
import dictionary
from evaluators.combinative_unit_segmenter_evaluator import (
    AccuracyEvaluatorForAttention,
    CombinativeSegmenterEvaluator,
)
import models.tagger
import models.util
from trainers import trainer
from trainers.tagger_trainer import TaggerTrainerBase


def get_num_candidates(mask_pairs, n_tokens):
    ncand = [0] * n_tokens
    for i, j in mask_pairs:
        ncand[i] += 1
    return ncand


class SubCombinativeUnitSegmenterTrainer(TaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.unigram_embed_model = None
        self.bigram_embed_model = None
        self.chunk_embed_model = None
        self.subword_embed_model = None
        self.label_begin_index = 9

    def load_external_embedding_models(self):
        if self.args.unigram_embed_model_path:
            self.unigram_embed_model = trainer.load_embedding_model(
                self.args.unigram_embed_model_path)
        if self.args.bigram_embed_model_path:
            self.bigram_embed_model = trainer.load_embedding_model(
                self.args.bigram_embed_model_path)
        if self.args.chunk_embed_model_path:
            self.chunk_embed_model = trainer.load_embedding_model(
                self.args.chunk_embed_model_path)
        if self.args.subword_embed_model_path:
            self.subword_embed_model = trainer.load_embedding_model(
                self.args.subword_embed_model_path)

    def init_model(self):
        super().init_model()
        if self.unigram_embed_model or self.bigram_embed_model or self.chunk_embed_model or self.subword_embed_model:
            self.classifier.load_pretrained_embedding_layer(
                self.dic,
                self.unigram_embed_model,
                self.bigram_embed_model,
                self.chunk_embed_model,
                self.subword_embed_model,
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
            if 'chunk_vector_dropout' in self.hparams:
                self.classifier.change_chunk_vector_dropout_ratio(
                    self.hparams['chunk_vector_dropout'])
            if 'subword_vector_dropout' in self.hparams:
                self.classifier.change_subword_vector_dropout_ratio(
                    self.hparams['subword_vector_dropout'])
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
                self.chunk_embed_model,
                self.subword_embed_model,
                train=(self.args.execute_mode == 'train'),
                fasttext=self.args.gen_oov_chunk_for_test,
            )
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

        if self.chunk_embed_model:
            pretrained_chunk_embed_size = self.chunk_embed_model.wv.syn0[
                0].shape[0]
        else:
            pretrained_chunk_embed_size = 0

        if self.subword_embed_model:
            pretrained_subword_embed_size = self.subword_embed_model.wv.syn0[
                0].shape[0]
        else:
            pretrained_subword_embed_size = 0

        self.hparams = {
            'pretrained_unigram_embed_size': pretrained_unigram_embed_size,
            'pretrained_bigram_embed_size': pretrained_bigram_embed_size,
            'pretrained_chunk_embed_size': pretrained_chunk_embed_size,
            'pretrained_subword_embed_size': pretrained_subword_embed_size,
            'pretrained_embed_usage': self.args.pretrained_embed_usage,
            'chunk_pooling_type': self.args.chunk_pooling_type.upper(),
            'subword_pooling_type': self.args.subword_pooling_type.upper(),
            'biaffine_type': self.args.biaffine_type,
            'use_gold_chunk': self.args.use_gold_chunk,
            'use_gold_subword': self.args.use_gold_subword,
            'unuse_nongold_chunk': self.args.unuse_nongold_chunk,
            'unuse_nongold_subword': self.args.unuse_nongold_subword,
            'use_unknown_pretrained_chunk':
            not self.args.ignore_unknown_pretrained_chunk,
            'use_unknown_pretrained_subword':
            not self.args.ignore_unknown_pretrained_subword,
            'min_chunk_len': self.args.min_chunk_len,
            'max_chunk_len': self.args.max_chunk_len,
            'min_subword_len': self.args.min_subword_len,
            'max_subword_len': self.args.max_subword_len,
            'unigram_embed_size': self.args.unigram_embed_size,
            'bigram_embed_size': self.args.bigram_embed_size,
            'chunk_embed_size': self.args.chunk_embed_size,
            'subword_embed_size': self.args.subword_embed_size,
            'rnn_unit_type': self.args.rnn_unit_type,
            'rnn_bidirection': self.args.rnn_bidirection,
            'rnn_batch_first': self.args.rnn_batch_first,
            'rnn_n_layers': self.args.rnn_n_layers,
            'rnn_hidden_size': self.args.rnn_hidden_size,
            'rnn_n_layers2': self.args.rnn_n_layers2,
            'rnn_hidden_size2': self.args.rnn_hidden_size2,
            'rnn_n_layers3': self.args.rnn_n_layers3,
            'rnn_hidden_size3': self.args.rnn_hidden_size3,
            'mlp_n_layers': self.args.mlp_n_layers,
            'mlp_hidden_size': self.args.mlp_hidden_size,
            'inference_layer': self.args.inference_layer,
            'embed_dropout': self.args.embed_dropout,
            'rnn_dropout': self.args.rnn_dropout,
            'biaffine_dropout': self.args.biaffine_dropout,
            'chunk_vector_dropout': self.args.chunk_vector_dropout,
            'subword_vector_dropout': self.args.subword_vector_dropout,
            'mlp_dropout': self.args.mlp_dropout,
            'task': self.args.task,
            'tagging_unit': self.args.tagging_unit,
            'token_freq_threshold': self.args.token_freq_threshold,
            'token_max_vocab_size': self.args.token_max_vocab_size,
            'bigram_freq_threshold': self.args.bigram_freq_threshold,
            'bigram_max_vocab_size': self.args.bigram_max_vocab_size,
            'chunk_freq_threshold': self.args.chunk_freq_threshold,
            'subword_freq_threshold': self.args.subword_freq_threshold,
            'chunk_max_vocab_size': self.args.chunk_max_vocab_size,
            'subword_max_vocab_size': self.args.subword_max_vocab_size,
            'reverse': self.args.reverse,
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
                        or key == 'pretrained_chunk_embed_size'
                        or key == 'pretrained_subword_embed_size'
                        or key == 'min_chunk_len' or key == 'max_chunk_len'
                        or key == 'min_subword_len' or key == 'max_subword_len'
                        or key == 'unigram_embed_size'
                        or key == 'bigram_embed_size'
                        or key == 'chunk_embed_size'
                        or key == 'subword_embed_size' or key == 'rnn_n_layers'
                        or key == 'rnn_hidden_size' or key == 'rnn_n_layers2'
                        or key == 'rnn_hidden_size2' or key == 'rnn_n_layers3'
                        or key == 'rnn_hidden_size3' or key == 'mlp_n_layers'
                        or key == 'mlp_hidden_size'
                        or key == 'token_freq_threshold'
                        or key == 'token_max_vocab_size'
                        or key == 'bigram_freq_threshold'
                        or key == 'bigram_max_vocab_size'
                        or key == 'chunk_freq_threshold'
                        or key == 'subword_freq_threshold'
                        or key == 'chunk_max_vocab_size'
                        or key == 'subword_max_vocab_size'):
                    val = int(val)

                elif (key == 'embed_dropout' or key == 'rnn_dropout'
                      or key == 'biaffine_dropout'
                      or key == 'chunk_vector_dropout'
                      or key == 'subword_vector_dropout'
                      or key == 'mlp_dropout'):
                    val = float(val)

                elif (key == 'rnn_batch_first' or key == 'rnn_bidirection'
                      or key == 'lowercasing' or key == 'normalize_digits'
                      or key == 'use_gold_chunk' or key == 'use_gold_subword'
                      or key == 'unuse_nongold_chunk'
                      or key == 'unuse_nongold_subword'
                      or key == 'ignore_unknown_pretrained_chunk'
                      or key == 'ignore_unknown_pretrained_subword'
                      or key == 'reverse'):
                    val = (val.lower() == 'true')

                hparams[key] = val

        if not 'unuse_nongold_chunk' in hparams:
            hparams['unuse_nongold_chunk'] = False
        if not 'unuse_nongold_subword' in hparams:
            hparams['unuse_nongold_subword'] = False

        self.hparams = hparams
        self.task = self.hparams['task']

        if self.args.execute_mode != 'interactive':
            if self.hparams[
                    'pretrained_unigram_embed_size'] > 0 and not self.unigram_embed_model:
                self.log('Error: unigram embedding model is necessary.')
                sys.exit()

            if self.hparams[
                    'pretrained_chunk_embed_size'] > 0 and not self.chunk_embed_model:
                self.log('Error: chunk embedding model is necessary.')
                sys.exit()

            if self.hparams[
                    'pretrained_subword_embed_size'] > 0 and not self.subword_embed_model:
                self.log('Error: subword embedding model is necessary.')
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

        if self.chunk_embed_model:
            pretrained_chunk_embed_size = self.chunk_embed_model.wv.syn0[
                0].shape[0]
            if hparams[
                    'pretrained_chunk_embed_size'] != pretrained_chunk_embed_size:
                self.log(
                    'Error: pretrained_chunk_embed_size and size (dimension) of loaded embedding model are conflicted.'
                    .format(hparams['pretrained_chunk_embed_size'],
                            pretrained_chunk_embed_size))
                sys.exit()

        if self.subword_embed_model:
            pretrained_subword_embed_size = self.subword_embed_model.wv.syn0[
                0].shape[0]
            if hparams[
                    'pretrained_subword_embed_size'] != pretrained_subword_embed_size:
                self.log(
                    'Error: pretrained_subword_embed_size and size (dimension) of loaded embedding model are conflicted.'
                    .format(hparams['pretrained_subword_embed_size'],
                            pretrained_subword_embed_size))
                sys.exit()

    def setup_data_loader(self):
        self.data_loader = segmentation_data_loader.SegmentationDataLoader(
            use_bigram=(self.hparams['bigram_embed_size'] > 0),
            use_chunk_trie=True,
            use_subword_trie=True,
            bigram_max_vocab_size=self.hparams['bigram_max_vocab_size'],
            bigram_freq_threshold=self.hparams['bigram_freq_threshold'],
            chunk_max_vocab_size=self.hparams['chunk_max_vocab_size'],
            chunk_freq_threshold=self.hparams['chunk_freq_threshold'],
            subword_max_vocab_size=self.hparams['subword_max_vocab_size'],
            subword_freq_threshold=self.hparams['subword_freq_threshold'],
            min_chunk_len=self.hparams['min_chunk_len'],
            max_chunk_len=self.hparams['max_chunk_len'],
            min_subword_len=self.hparams['min_subword_len'],
            max_subword_len=self.hparams['max_subword_len'],
            add_gold_chunk=self.hparams['use_gold_chunk'],
            add_nongold_chunk=not self.hparams['unuse_nongold_chunk'],
            add_unknown_pretrained_chunk=self.
            hparams['use_unknown_pretrained_chunk'],
            add_gold_subword=self.hparams['use_gold_subword'],
            add_nongold_subword=not self.hparams['unuse_nongold_subword'],
            add_unknown_pretrained_subword=self.
            hparams['use_unknown_pretrained_subword'],
            unigram_vocab=(self.unigram_embed_model.wv
                           if self.unigram_embed_model else set()),
            bigram_vocab=(self.bigram_embed_model.wv
                          if self.bigram_embed_model else set()),
            chunk_vocab=(self.chunk_embed_model.wv
                         if self.chunk_embed_model else set()),
            subword_vocab=(self.subword_embed_model.wv
                           if self.subword_embed_model else set()),
            generate_ngram_chunks=self.args.gen_oov_chunk_for_test,
            generate_subwords=self.args.gen_oov_subword_for_test,
            trie_ext=self.dic_ext,
            trie_subword=self.dic_subword)

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
            'Start chunk search for {} data (min_len={}, max_len={})\n'.format(
                data_type, self.hparams['min_chunk_len'],
                self.hparams['max_chunk_len']))

        segmentation_data_loader.add_chunk_sequences(
            data,
            dic,
            min_len=self.hparams['min_chunk_len'],
            max_len=self.hparams['max_chunk_len'],
            evaluate=evaluate,
            model_type=self.hparams['chunk_pooling_type'],
            chunk_type=constants.CHUNK)

        self.log('Start subword search for {} data (min_len={}, max_len={})\n'.
                 format(data_type, self.hparams['min_subword_len'],
                        self.hparams['max_subword_len']))

        segmentation_data_loader.add_chunk_sequences(
            data,
            dic,
            min_len=self.hparams['min_subword_len'],
            max_len=self.hparams['max_subword_len'],
            evaluate=evaluate,
            model_type=self.hparams['subword_pooling_type'],
            chunk_type=constants.SUBWORD)

    def setup_classifier(self):
        dic = self.dic
        hparams = self.hparams

        n_vocab = len(dic.tables['unigram'])
        unigram_embed_size = hparams['unigram_embed_size']
        chunk_embed_size = hparams['chunk_embed_size']
        subword_embed_size = hparams['subword_embed_size']
        n_chunks = len(dic.tries[constants.CHUNK]) if dic.has_trie(
            constants.CHUNK) else 1
        n_subwords = len(dic.tries[constants.SUBWORD]) if dic.has_trie(
            constants.SUBWORD) else 1

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

        if 'pretrained_chunk_embed_size' in hparams and hparams[
                'pretrained_chunk_embed_size'] > 0:
            pretrained_chunk_embed_size = hparams[
                'pretrained_chunk_embed_size']
        else:
            pretrained_chunk_embed_size = 0

        if 'pretrained_subword_embed_size' in hparams and hparams[
                'pretrained_subword_embed_size'] > 0:
            pretrained_subword_embed_size = hparams[
                'pretrained_subword_embed_size']
        else:
            pretrained_subword_embed_size = 0

        if 'pretrained_embed_usage' in hparams:
            pretrained_embed_usage = models.util.ModelUsage.get_instance(
                hparams['pretrained_embed_usage'])
        else:
            pretrained_embed_usage = models.util.ModelUsage.NONE

        n_label = len(dic.tables[constants.SEG_LABEL])
        n_labels = [n_label]

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

            if pretrained_chunk_embed_size > 0 and pretrained_chunk_embed_size != chunk_embed_size:
                print(
                    'Error: pre-trained and random initialized chunk embedding vectors must be the same size (dimension) for {} operation : d1={}, d2={}'
                    .format(hparams['pretrained_embed_usage'],
                            pretrained_chunk_embed_size, chunk_embed_size),
                    file=sys.stderr)
                sys.exit()

            if pretrained_subword_embed_size > 0 and pretrained_subword_embed_size != subword_embed_size:
                print(
                    'Error: pre-trained and random initialized subword embedding vectors must be the same size (dimension) for {} operation : d1={}, d2={}'
                    .format(hparams['pretrained_embed_usage'],
                            pretrained_subword_embed_size, subword_embed_size),
                    file=sys.stderr)
                sys.exit()

        predictor = models.tagger.construct_RNNTagger(
            n_vocab=n_vocab,
            unigram_embed_size=unigram_embed_size,
            n_bigrams=n_bigrams,
            bigram_embed_size=bigram_embed_size,
            n_chunks=n_chunks,
            chunk_embed_size=chunk_embed_size,
            n_subwords=n_subwords,
            subword_embed_size=subword_embed_size,
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
            mlp_additional_hidden_size=0,
            rnn_dropout=hparams['rnn_dropout'],
            biaffine_dropout=hparams['biaffine_dropout']
            if 'biaffine_dropout' in hparams else 0.0,
            embed_dropout=hparams['embed_dropout']
            if 'embed_dropout' in hparams else 0.0,
            mlp_dropout=hparams['mlp_dropout'],
            chunk_vector_dropout=hparams['chunk_vector_dropout']
            if 'chunk_vector_dropout' in hparams else 0.0,
            subword_vector_dropout=hparams['subword_vector_dropout']
            if 'subword_vector_dropout' in hparams else 0.0,
            pretrained_unigram_embed_size=pretrained_unigram_embed_size,
            pretrained_bigram_embed_size=pretrained_bigram_embed_size,
            pretrained_chunk_embed_size=pretrained_chunk_embed_size,
            pretrained_subword_embed_size=pretrained_subword_embed_size,
            pretrained_embed_usage=pretrained_embed_usage,
            chunk_pooling_type=hparams['chunk_pooling_type']
            if 'chunk_pooling_type' in hparams else '',
            subword_pooling_type=hparams['subword_pooling_type']
            if 'subword_pooling_type' in hparams else '',
            min_chunk_len=hparams['min_chunk_len']
            if 'min_chunk_len' in hparams else 0,
            max_chunk_len=hparams['max_chunk_len']
            if 'max_chunk_len' in hparams else 0,
            min_subword_len=hparams['min_subword_len']
            if 'min_subword_len' in hparams else 0,
            max_subword_len=hparams['max_subword_len']
            if 'max_subword_len' in hparams else 0,
            chunk_loss_ratio=0.0,
            subword_loss_ratio=0.0,
            biaffine_type=hparams['biaffine_type']
            if 'biaffine_type' in hparams else '',
            reverse=self.hparams['reverse'],
        )

        self.classifier = classifiers.sub_combinative_sequence_tagger.SubCombinativeSequenceTagger(
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
                evaluator1 = CombinativeSegmenterEvaluator(
                    self.dic.tables[constants.SEG_LABEL].id2str)

        if not evaluator:
            self.evaluator = evaluator1
        else:
            evaluator = evaluator1

    def gen_inputs(self, data, ids, evaluate=True, restrict_memory=False):
        device = torch.device(
            constants.CUDA_DEVICE) if self.args.gpu >= 0 else torch.device(
                constants.CPU_DEVICE)
        '''
        inputs[0]: unigram sequences
        inputs[1]: bigram sequences
        inputs[2]: None
        inputs[3]: chunk sequences (word)
        inputs[4]: chunk additional feature sequences (word)
        inputs[5]: chunk masking sequences (word)
        inputs[6]: subword sequences
        inputs[7]: subword additional feature sequences
        inputs[8]: subword masking sequences
        outputs[0]: gold segmentation sequences (label)
        outputs[1]: gold chunk sequences (word)
        outputs[2]: gold subword sequences
        Reverse mode:
            inputs
                3 <-> 6
                4 <-> 7
                5 <-> 8
            outputs
                1 <-> 2
        '''

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
        sws = [
            torch.tensor(data.inputs[6][j], dtype=int, device=device)
            for j in ids
        ] if data.inputs[6] else None
        cds = [
            torch.tensor(data.inputs[7][j], dtype=int, device=device)
            for j in ids
        ] if data.inputs[7] else None

        if (self.hparams['chunk_pooling_type'] == constants.CON
                or self.hparams['chunk_pooling_type'] == constants.WCON):
            feat_size = sum([
                h for h in range(self.hparams['min_chunk_len'],
                                 self.hparams['max_chunk_len'] + 1)
            ])
            embed_size = self.hparams['chunk_embed_size']
        else:
            feat_size = 0
            embed_size = 0

        use_chunk_attention = (
            self.hparams['chunk_pooling_type'] == constants.WAVG
            or self.hparams['chunk_pooling_type'] == constants.WCON)

        ms = [
            segmentation_data_loader.convert_mask_matrix(
                mask=data.inputs[5][j],
                n_tokens=len(us[i]),
                n_chunks=len(cs[i]) if cs else 0,
                feat_size=feat_size,
                emb_size=embed_size,
                use_attention=use_chunk_attention,
                device=device) for i, j in enumerate(ids)
        ]

        if (self.hparams['subword_pooling_type'] == constants.CON
                or self.hparams['subword_pooling_type'] == constants.SWCON):
            feat_size = sum([
                h for h in range(self.hparams['min_subword_len'],
                                 self.hparams['max_subword_len'] + 1)
            ])
            embed_size = self.hparams['subword_embed_size']
        else:
            feat_size = 0
            embed_size = 0

        use_subword_attention = (
            self.hparams['subword_pooling_type'] == constants.SWAVG
            or self.hparams['subword_pooling_type'] == constants.SWCON)

        qs = [
            segmentation_data_loader.convert_mask_matrix(
                mask=data.inputs[8][j],
                n_tokens=len(us[i]),
                n_chunks=len(sws[i]) if sws else 0,
                feat_size=feat_size,
                emb_size=embed_size,
                use_attention=use_subword_attention,
                device=device) for i, j in enumerate(ids)
        ]

        fs = None

        gls = [
            torch.tensor(data.outputs[0][j], dtype=int, device=device)
            for j in ids
        ] if evaluate else None
        gcs = [
            torch.tensor(data.outputs[1][j], dtype=int, device=device)
            for j in ids
        ] if evaluate else None
        gsws = [
            torch.tensor(data.outputs[2][j], dtype=int, device=device)
            for j in ids
        ] if evaluate else None

        if evaluate:
            if self.args.evaluation_method == 'attention':
                ncands = [
                    get_num_candidates(data.inputs[5][j][0], len(us[i]))
                    for i, j in enumerate(ids)
                ]
                nswcands = [
                    get_num_candidates(data.inputs[8][j][0], len(us[i]))
                    for i, j in enumerate(ids)
                ]
                if not self.hparams['reverse']:
                    return us, cs, ds, sws, cds, ms, qs, bs, fs, gls, gcs, gsws, ncands, nswcands
                else:
                    return us, sws, cds, cs, ds, qs, ms, bs, fs, gls, gsws, gcs, nswcands, ncands
            else:
                if not self.hparams['reverse']:
                    return us, cs, ds, sws, cds, ms, qs, bs, fs, gls, gcs, gsws
                else:
                    return us, sws, cds, cs, ds, qs, ms, bs, fs, gls, gsws, gcs

        else:
            if not self.hparams['reverse']:
                return us, cs, ds, sws, cds, ms, qs, bs, fs
            else:
                return us, sws, cds, cs, ds, qs, ms, bs, fs

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

            trie_ext = dictionary.MapTrie()
            with open(edic_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(constants.COMMENT_SYM):
                        continue

                    if len(line) == 0:
                        continue

                    word = line
                    trie_ext.get_chunk_id(word, word, update=True)

            self.dic_ext = trie_ext
            self.log('Load external dictionary: {}'.format(edic_path))
            self.log('Num of chunks: {}'.format(len(trie_ext)))
            self.log('')

    def load_subword_dictionary(self):
        if self.args.subword_dic_path:
            swdic_path = self.args.subword_dic_path

            trie_subword = dictionary.MapTrieSubword()
            with open(swdic_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(constants.COMMENT_SYM):
                        continue

                    if len(line) == 0:
                        continue

                    subword = line
                    trie_subword.get_subword_id(subword, subword, update=True)

            self.dic_subword = trie_subword
            self.log('Load subword dictionary: {}'.format(swdic_path))
            self.log('Num of subwords: {}'.format(len(trie_subword)))
            self.log('')
