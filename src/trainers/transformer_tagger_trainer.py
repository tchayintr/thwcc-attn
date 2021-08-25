import copy
from datetime import datetime
import numpy as np
import pickle
import sys
import torch

import classifiers.transformer_sequence_tagger
import common
import constants
from data_loaders import data_loader, segmentation_data_loader
from evaluators.common import AccuracyEvaluator, FMeasureEvaluator, DoubleFMeasureEvaluator
import models.tagger
from trainers import trainer
from trainers.trainer import Trainer
import util


class TransformerTaggerTrainerBase(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)

    def show_data_info(self, data_type):
        dic = self.dic_val if data_type == 'valid' else self.dic
        self.log('### {} dic'.format(data_type))
        self.log('Number of tokens: {}'.format(
            len(dic.tables[constants.UNIGRAM])))
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

        if self.dic.has_table(constants.SEG_LABEL):
            id2seg = {
                v: k
                for k, v in self.dic.tables[
                    constants.SEG_LABEL].str2id.items()
            }
            self.log('# label_set: {}\n'.format(id2seg))

        self.report('[INFO] vocab: {}'.format(
            len(self.dic.tables[constants.UNIGRAM])))
        self.report('[INFO] data length: train={} valid={}'.format(
            len(train.inputs[0]),
            len(valid.inputs[0]) if valid else 0))

    def gen_inputs(self, data, ids, evaluate=True):
        # xp = torch.tensor if self.args.gpu >= 0 else np.asarray
        # create tensor on cpu
        device = torch.device(
            constants.CUDA_DEVICE) if self.args.gpu >= 0 else torch.device(
                constants.CPU_DEVICE)

        us = [
            torch.tensor(data.inputs[0][j], dtype=int, device=device)
            for j in ids
        ]
        ls = [
            torch.tensor(data.outputs[0][j], dtype=int, device=device)
            for j in ids
        ] if evaluate else None

        if evaluate:
            return us, ls
        else:
            return us

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

    def run_train_mode(self):
        # save model
        if not self.args.quiet:
            hparam_path = '{}/{}.hyp'.format(self.model_path_prefix,
                                             self.start_time)
            with open(hparam_path, 'w') as f:
                for key, val in self.hparams.items():
                    print('{}={}'.format(key, val), file=f)
                self.log('Save hyperparameters: {}'.format(hparam_path))

            dic_path = '{}/{}.s2i'.format(self.model_path_prefix,
                                          self.start_time)
            with open(dic_path, 'wb') as f:
                pickle.dump(self.dic, f)
            self.log('Save string2index table: {}'.format(dic_path))

        for e in range(max(1, self.args.epoch_begin), self.args.epoch_end + 1):
            time = datetime.now().strftime('%Y%m%d_%H%M')
            self.log('Start epoch {}: {}\n'.format(e, time))
            self.report('[INFO] Start epoch {} at {}'.format(e, time))

            # learning and evaluation for each epoch
            self.run_epoch(self.train, train=True)

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('Finish: {}\n'.format(time))
        self.log('Finish: {}\n'.format(time))

    def run_epoch(self, data, train=False):
        classifier = self.classifier
        evaluator = self.evaluator
        classifier.train()

        if not train:
            if self.dic_val is not None:
                classifier = copy.deepcopy(self.classifier)
                # classifier = self.classifier
                self.update_model(classifier=classifier, dic=self.dic_val)
                evaluator = copy.deepcopy(self.evaluator)
                self.setup_evaluator(evaluator)
            classifier.eval()
            classifier.change_dropout_ratio(0)

        n_sen = 0
        total_loss = 0
        total_counts = None

        i = 0
        n_ins = len(data.inputs[0])
        shuffle = True if train else False
        for bidx, ids in enumerate(
                trainer.batch_generator(n_ins,
                                        batch_size=self.args.batch_size,
                                        shuffle=shuffle)):
            inputs = self.gen_inputs(data, ids)
            xs = inputs[0]
            golds = self.get_labels(inputs)
            if train:
                ret = classifier(*inputs, train=True)
            else:
                with torch.no_grad():
                    ret = classifier(*inputs, train=False)
            # loss = ret[0]
            loss = ret[0] / self.args.accumulate_grad_batches
            outputs = self.get_outputs(ret)

            if train:
                # set the accumulated gradients to zero
                # self.optimizer.zero_grad()
                loss.backward()  # back propagation
                if (bidx + 1) % self.args.accumulate_grad_batches == 0:
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(),
                                                   self.args.grad_clip)
                    self.optimizer.step()  # update the parameters
                    self.optimizer.zero_grad()
                    # torch.nn.utils.clip_grad_norm_(classifier.parameters(),
                    #                                self.args.grad_clip)
                i_max = min(i + self.args.batch_size, n_ins)

                self.log('* batch %d-%d loss: %.4f' %
                         ((i + 1), i_max, loss.data))
                i = i_max

            n_sen += len(xs)
            total_loss += loss.data
            counts = evaluator.calculate(*[xs], *golds, *outputs)
            if not total_counts:
                total_counts = counts
            else:
                total_counts.merge(counts)

            if ((train and self.n_iter > 0
                 and self.n_iter * self.args.batch_size % self.args.break_point
                 == 0)):
                # and ((bidx + 1) % self.args.accumulate_grad_batches == 0)):
                now_e = '%.3f' % (self.args.epoch_begin - 1 +
                                  (self.n_iter * self.args.batch_size / n_ins))
                time = datetime.now().strftime('%Y%m%d_%H%M')

                self.log('\n### Finish %s iterations (%s examples: %s epoch)' %
                         (self.n_iter,
                          (self.n_iter * self.args.batch_size), now_e))
                self.log('<training result for previous iterations>')
                res = evaluator.report_results(n_sen,
                                               total_counts,
                                               total_loss,
                                               file=self.logger)
                self.report('train\t%d\t%s\t%s' % (self.n_iter, now_e, res))

                if self.args.valid_data:
                    self.log('\n<validation result>')
                    v_res = self.run_epoch(self.valid, train=False)
                    self.report('valid\t%d\t%s\t%s' %
                                (self.n_iter, now_e, v_res))

                # save model
                if not self.args.quiet:
                    model_path = '{}/{}_e{}.pt'.format(self.model_path_prefix,
                                                       self.start_time, now_e)
                    self.log('Save the model: %s\n' % model_path)
                    self.report('[INFO] Save the model: %s\n' % model_path)
                    # torch.save({
                    #     'model': classifier.state_dict(),
                    #     'dic': self.dic,
                    #     'dic_val': self.dic_val,
                    #     'hparams': self.hparams,
                    # }, model_path)
                    torch.save(classifier.state_dict(), model_path)

                if not self.args.quiet:
                    self.reporter.close()
                    self.reporter = open(
                        '{}/{}.log'.format(constants.LOG_DIR, self.start_time),
                        'a')

                # Reset counters
                n_sen = 0
                total_loss = 0
                counts = None
                total_counts = None

            if train:
                self.n_iter += 1

        if train:
            self.scheduler.step()

        res = None if train else evaluator.report_results(
            n_sen, total_counts, total_loss, file=self.logger)
        return res

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
        super().run_eval_mode()

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


class TransformerTaggerTrainer(TransformerTaggerTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.unigram_embed_model = None
        self.label_begin_index = 1

    def load_external_embedding_models(self):
        if self.args.unigram_embed_model_path:
            self.unigram_embed_model = trainer.load_embedding_model(
                self.args.unigram_embed_model_path)

    def init_model(self):
        super().init_model()
        if self.unigram_embed_model:
            self.classifier.load_pretrained_embedding_layer(
                self.dic, self.unigram_embed_model, finetuning=True)

    def reinit_model(self):
        self.log('Re-initialize model from hyperparameters\n')
        self.setup_classifier()

    def load_model(self):
        model_path = self.args.model_path
        dirname = model_path.parent
        basename = model_path.name.split('_e')[0]
        dic_path = '{}/{}.s2i'.format(dirname, basename)
        hparam_path = '{}/{}.hyp'.format(dirname, basename)
        param_path = model_path

        # dictionary
        self.load_dic(dic_path)

        # hyper parameters
        self.load_hyperparameters(hparam_path)
        self.log('Load hyperparameters: {}\n'.format(hparam_path))
        self.show_hyperparameters()

        # model
        self.reinit_model()
        self.classifier.load_state_dict(torch.load(model_path))
        self.log('Load model parameters: {}\n'.format(model_path))

        if self.args.execute_mode == 'train':
            if 'embed_dropout' in self.hparams:
                self.classifier.change_embed_dropout_ratio(
                    self.hparams['embed_dropout'])
            if 'tfm_dropout' in self.hparams:
                self.classifier.change_tfm_dropout_ratio(
                    self.hparams['tfm_dropout'])
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
                                             train=train)
            classifier.grow_inference_layers(dic)

    def init_hyperparameters(self):
        if self.unigram_embed_model:
            pretrained_unigram_embed_size = self.unigram_embed_model.wv.syn0[
                0].shape[0]
        else:
            pretrained_unigram_embed_size = 0

        self.hparams = {
            'pretrained_unigram_embed_size': pretrained_unigram_embed_size,
            'pretrained_embed_usage': self.args.pretrained_embed_usage,
            'unigram_embed_size': self.args.unigram_embed_size,
            'tfm_n_layers': self.args.tfm_n_layers,
            'tfm_ff_hidden_size': self.args.tfm_ff_hidden_size,
            'tfm_hidden_size': self.args.tfm_hidden_size,
            'tfm_n_heads': self.args.tfm_n_heads,
            'mlp_n_layers': self.args.mlp_n_layers,
            'mlp_hidden_size': self.args.mlp_hidden_size,
            'inference_layer': self.args.inference_layer,
            'embed_dropout': self.args.embed_dropout,
            'tfm_dropout': self.args.tfm_dropout,
            'mlp_dropout': self.args.mlp_dropout,
            'task': self.args.task,
            'lowercasing': self.args.lowercasing,
            'normalize_digits': self.args.normalize_digits,
            'token_freq_threshold': self.args.token_freq_threshold,
            'token_max_vocab_size': self.args.token_max_vocab_size,
            'max_seq_len': self.args.max_seq_len,
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
                        or key == 'unigram_embed_size' or key == 'tfm_n_layers'
                        or key == 'tfm_ff_hidden_size'
                        or key == 'tfm_hidden_size' or key == 'tfm_n_heads'
                        or key == 'mlp_n_layers' or key == 'mlp_hidden_size'
                        or key == 'token_freq_threshold'
                        or key == 'token_max_vocab_size'
                        or key == 'max_seq_len'):
                    val = int(val)

                elif (key == 'embed_dropout' or key == 'tfm_dropout'
                      or key == 'mlp_dropout'):
                    val = float(val)

                elif (key == 'lowercasing' or key == 'normalize_digits'):
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

    def setup_data_loader(self):
        if self.task == constants.TASK_SEG:
            self.data_loader = segmentation_data_loader.SegmentationDataLoader(
                # token_index=self.args.token_index,
                unigram_vocab=(self.unigram_embed_model.wv
                               if self.unigram_embed_model else set()), )

    def setup_classifier(self):
        dic = self.dic
        hparams = self.hparams

        n_vocab = len(dic.tables['unigram'])
        unigram_embed_size = hparams['unigram_embed_size']

        if 'pretrained_unigram_embed_size' in hparams and hparams[
                'pretrained_unigram_embed_size'] > 0:
            pretrained_unigram_embed_size = hparams[
                'pretrained_unigram_embed_size']
        else:
            pretrained_unigram_embed_size = 0

        if 'pretrained_embed_usage' in hparams:
            pretrained_embed_usage = models.util.ModelUsage.get_instance(
                hparams['pretrained_embed_usage'])
        else:
            pretrained_embed_usage = models.util.ModelUsage.NONE

        if common.is_segmentation_task(self.task):
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

        predictor = models.tagger.construct_TFMTagger(
            n_vocab=n_vocab,
            unigram_embed_size=unigram_embed_size,
            tfm_n_layers=hparams['tfm_n_layers'],
            tfm_ff_hidden_size=hparams['tfm_ff_hidden_size'],
            tfm_hidden_size=hparams['tfm_hidden_size'],
            tfm_n_heads=hparams['tfm_n_heads'],
            mlp_n_layers=hparams['mlp_n_layers'],
            mlp_hidden_size=hparams['mlp_hidden_size'],
            n_labels=n_labels[0],
            use_crf=hparams['inference_layer'] == 'crf',
            embed_dropout=hparams['embed_dropout']
            if 'embed_dropout' in hparams else 0.0,
            tfm_dropout=hparams['tfm_dropout'],
            mlp_dropout=hparams['mlp_dropout'],
            max_seq_len=hparams['max_seq_len'],
            pretrained_unigram_embed_size=pretrained_unigram_embed_size,
            pretrained_embed_usage=pretrained_embed_usage,
        )

        self.classifier = classifiers.transformer_sequence_tagger.TransformerSequenceTagger(
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
