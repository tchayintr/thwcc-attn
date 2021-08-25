import copy
from datetime import datetime
from gensim.models import keyedvectors, FastText
import numpy as np
import os
import pickle
import sys
import torch
import torchtext

import common
import constants
# import features


class Trainer(object):
    def __init__(self, args, logger=sys.stderr):
        err_msgs = []
        if args.execute_mode == 'train':
            if not args.train_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--train_data')
                err_msgs.append(msg)

            if not args.task and not args.model_path:
                msg = 'Error: the following argument is required for {} mode unless resuming a trained model: {}'.format(
                    'train', '--task')
                err_msgs.append(msg)

        elif args.execute_mode == 'eval':
            if not args.model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--model_path/-m')
                err_msgs.append(msg)

            if not args.test_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--test_data')
                err_msgs.append(msg)

        elif args.execute_mode == 'decode':
            if not args.model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--model_path/-m')
                err_msgs.append(msg)

            if not args.decode_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--decode_data')

        elif args.execute_mode == 'interactive':
            if not args.model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--model_path/-m')
                err_msgs.append(msg)

        else:
            msg = 'Error: invalid execute mode: {}'.format(args.execute_mode)
            err_msgs.append(msg)

        if err_msgs:
            for msg in err_msgs:
                print(msg, file=sys.stderr)
            sys.exit()

        self.args = args
        self.start_time = datetime.now().strftime('%Y%m%d_%H%M')
        self.logger = logger  # output execute log
        self.reporter = None  # output evaluation results
        self.task = args.task
        self.train = None
        self.valid = None
        self.test = None
        self.decode_data = None
        self.hparams = None
        self.dic = None
        self.dic_org = None
        self.dic_val = None
        self.dic_ext = None
        self.dic_subword = None
        self.data_loader = None
        self.classifier = None
        self.evaluator = None
        self.feat_extractor = None
        self.optimizer = None
        self.scheduler = None
        self.model_path_prefix = None
        self.n_iter = 1
        self.label_begin_index = 2

        self.log('Start time: {}\n'.format(self.start_time))
        if not self.args.quiet:
            self.reporter = open('{}/{}.log'.format(constants.LOG_DIR,
                                                    self.start_time),
                                 mode='a')

            if args.execute_mode == 'train':
                self.model_path_prefix = '{}/{}'.format(
                    constants.MODEL_DIR, self.start_time)
                if not os.path.exists(self.model_path_prefix):
                    os.makedirs(self.model_path_prefix)

    def report(self, message):
        if not self.args.quiet:
            print(message, file=self.reporter)

    def log(self, message=''):
        print(message, file=self.logger)

    def close(self):
        if not self.args.quiet:
            self.reporter.close()

    def init_feature_extractor(self, use_gpu=False):
        if 'feature_template' in self.hparams:
            template = self.hparams['feature_template']
            feat_size = 0

            if template:
                self.feat_extractor = features.DictionaryFeatureExtractor(
                    template, use_gpu=use_gpu)

                if 'additional_feat_size' in self.hparams:
                    feat_size = self.hparams['additional_feat_size']
                else:
                    feat_size = self.feat_extractor.size

            self.hparams.update({'additional_feat_size': feat_size})

    def load_external_dictionary(self):
        # to be implemented in sub-class
        pass

    def load_external_embedding_models(self):
        # to be implemented in sub-class
        pass

    def load_subword_dictionary(self):
        # to be implemented in sub-class
        pass

    def init_model(self):
        self.log('Initialize model from hyperparameters\n')
        self.setup_classifier()

    def reinit_model(self):
        self.log('Re-initialize model from hyperparameters\n')
        self.setup_classifier()

    def setup_classifier(self):
        # to be implemented in sub-class
        pass

    def load_dic(self, dic_path):
        with open(dic_path, 'rb') as f:
            self.dic = pickle.load(f)
        self.log('Load dic: {}'.format(dic_path))
        self.log('Num of tokens: {}'.format(
            len(self.dic.tables[constants.UNIGRAM])))
        if self.dic.has_table(constants.BIGRAM):
            self.log('Num of bigrams: {}'.format(
                len(self.dic.tables[constants.BIGRAM])))
        if self.dic.has_trie(constants.CHUNK):
            self.log('Num of chunks: {}'.format(
                len(self.dic.tries[constants.CHUNK])))
        if self.dic.has_trie(constants.CC):
            self.log('Num of character clusters: {}'.format(
                len(self.dic.tries[constants.CC])))
        if self.dic.has_table(constants.SEG_LABEL):
            self.log('Num of segmentation labels: {}'.format(
                len(self.dic.tables[constants.SEG_LABEL])))
        # for i in range(3):  # tmp
        #     if self.dic.has_table(constants.ATTR_LABEL(i)):
        #         self.log('Num of {}-th attribute labels: {}'.format(
        #             i, len(self.dic.tables[constants.ATTR_LABEL(i)])))
        # if self.dic.has_table(constants.ARC_LABEL):
        #     self.log('Num of arc labels: {}'.format(
        #         len(self.dic.tables[constants.ARC_LABEL])))
        self.log('')

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

    def show_hyperparameters(self):
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            if (k in self.hparams
                    and ('dropout' in k or 'freq_threshold' in k
                         or 'max_vocab_size' in k or k == 'attr_indexes')):
                update = self.hparams[k] != v
                message = '{}={}{}'.format(
                    k, v, ' (original value ' + str(self.hparams[k]) +
                    ' was updated)' if update else '')
                self.hparams[k] = v

            elif (k == 'task' and v == 'seg'
                  and self.hparams[k] == constants.TASK_SEGTAG
                  and (self.args.execute_mode == 'decode'
                       or self.args.execute_mode == 'interactive')):
                self.task = self.hparams[k] = v
                message = '{}={}'.format(k, v)

            elif k in self.hparams and v != self.hparams[k]:
                message = '{}={} (input option value {} was discarded)'.format(
                    k, self.hparams[k], v)

            else:
                message = '{}={}'.format(k, v)

            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')

    def update_model(self, classifier=None, dic=None, train=False):
        # to be implemented in sub-class
        pass

    def init_hyperparameters(self):
        # to be implemented in sub-class
        self.hparams = {}

    def load_hyperparameters(self, hparam_path):
        # to be implemented in sub-class
        pass

    def load_training_and_validation_data(self):
        self.load_data('train')
        if self.args.valid_data:
            self.load_data('valid')
        self.show_training_data()

    def load_test_data(self):
        self.dic_org = copy.deepcopy(self.dic)
        self.load_data('test')

    def load_decode_data(self):
        self.load_data('decode')

    def load_data(self, data_type):
        if data_type == 'train':
            self.setup_data_loader()
            data_path = self.args.train_data
            data, self.dic = self.data_loader.load_gold_data(
                data_path,
                self.args.input_data_format,
                dic=self.dic,
                train=True)
            self.dic.create_id2strs()
            self.train = data

        elif data_type == 'valid':
            data_path = self.args.valid_data
            self.dic_val = copy.deepcopy(self.dic)
            data, self.dic_val = self.data_loader.load_gold_data(
                data_path,
                self.args.input_data_format,
                dic=self.dic_val,
                train=False)
            self.dic_val.create_id2strs()
            self.valid = data

        elif data_type == 'test':
            self.setup_data_loader()
            data_path = self.args.test_data
            data, self.dic = self.data_loader.load_gold_data(
                data_path,
                self.args.input_data_format,
                dic=self.dic,
                train=False)
            self.dic.create_id2strs()
            self.test = data

        elif data_type == 'decode':
            self.setup_data_loader()
            data_path = self.args.decode_data
            data = self.data_loader.load_decode_data(
                data_path, self.args.input_data_format, dic=self.dic)
            self.dic.create_id2strs()
            self.decode_data = data

        else:
            print('Error: incorrect data type: {}'.format(), file=sys.stderr)
            sys.exit()

        # if ('feature_template' in self.hparams
        #         and self.hparams['feature_template']
        #         and self.args.batch_feature_extraction):  # for segmentation
        #     data.featvecs = self.feat_extractor.extract_features(
        #         data.inputs[0], self.dic.tries[constants.CHUNK])
        #     self.log('Extract dictionary features for {}'.format(data_type))

        self.log('Load {} data: {}'.format(data_type, data_path))
        self.show_data_info(data_type)

    def show_data_info(self):
        # to be implemented in sub-class
        pass

    def show_training_data(self):
        # to be implemented in sub-class
        pass

    def setup_data_loader(self, use_torchtext):
        # to be implemented in sub-class
        pass

    def setup_optimizer(self):
        mparams = self.classifier.parameters()

        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                params=mparams,
                lr=self.args.adam_alpha,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.adam_weight_decay)

        elif self.args.optimizer == 'adadelta':
            optimizer = torch.optim.Adadelta(
                params=mparams,
                lr=self.args.adadelta_lr,
                rho=self.args.adadelta_rho,
                weight_decay=self.args.adadelta_weight_decay)

        elif self.args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(
                params=mparams,
                lr=self.args.adagrad_lr,
                lr_decay=self.args.adagrad_lr_decay,
                weight_decay=self.args.adagrad_weight_decay)

        elif self.args.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                params=mparams,
                lr=self.args.rmsprop_lr,
                alpha=self.args.rmsprop_alpha,
                weight_decay=self.args.rmsprop_weight_decay)

        elif self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                params=mparams,
                lr=self.args.sgd_lr,
                weight_decay=self.args.sgd_weight_decay)

        self.optimizer = optimizer

        return self.optimizer

    def setup_scheduler(self):
        if self.args.scheduler == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, gamma=self.args.exponential_gamma)

        self.scheduler = scheduler

        return self.scheduler

    def setup_evaluator(self, evaluator=None):
        # to be implemented in sub-class
        pass

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

    def run_eval_mode(self):
        self.log('<test result>')
        res = self.run_epoch(self.test, train=False)
        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('test\t%s\n' % res)
        self.report('Finish: %s\n' % time)
        self.log('Finish: %s\n' % time)

    def run_decode_mode(self):
        self.update_model(classifier=self.classifier, dic=self.dic)
        if self.args.output_data:
            with open(self.args.output_data, 'w') as f:
                self.decode(self.decode_data, file=f)
        else:
            self.decode(self.decode_data, file=sys.stdout)

    def run_interactive_mode(self):
        # to be implemented in sub-class
        pass

    def decode(self, rdata, file=sys.stdout):
        # to be implemented in sub-class
        pass

    def gen_inputs(self, data, ids):
        # to be implemented in sub-class
        return None

    def get_labels(self, inputs):  # called from run_epoch
        return inputs[self.label_begin_index:]

    def get_outputs(self, ret):  # called from run_epoch
        return ret[1:]

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
        for ids in batch_generator(n_ins,
                                   batch_size=self.args.batch_size,
                                   shuffle=shuffle):
            inputs = self.gen_inputs(data, ids)
            xs = inputs[0]
            golds = self.get_labels(inputs)
            if train:
                ret = classifier(*inputs, train=True)
            else:
                with torch.no_grad():
                    ret = classifier(*inputs, train=False)
            loss = ret[0]
            outputs = self.get_outputs(ret)

            if train:
                self.optimizer.zero_grad(
                )  # set the accumulated gradients to zero
                loss.backward()  # back propagation
                torch.nn.utils.clip_grad_norm_(classifier.parameters(),
                                               self.args.grad_clip)
                self.optimizer.step()  # update the parameters
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

            if (train and self.n_iter > 0 and self.n_iter *
                    self.args.batch_size % self.args.break_point == 0):
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


def batch_generator(len_data, batch_size=64, shuffle=True):
    perm = np.random.permutation(len_data) if shuffle else list(
        range(0, len_data))
    for i in range(0, len_data, batch_size):
        i_max = min(i + batch_size, len_data)
        yield perm[i:i_max]


def gen_iterator(data,
                 batch_size,
                 sort_key=None,
                 device=None,
                 train=True,
                 shuffle=None,
                 sort=False,
                 sort_within_batch=None):
    data_iter = torchtext.data.Iterator(dataset=data,
                                        batch_size=batch_size,
                                        sort_key=sort_key,
                                        device=device,
                                        train=train,
                                        shuffle=shuffle,
                                        sort=sort,
                                        sort_within_batch=sort_within_batch)

    return data_iter


def load_embedding_model(model_path):
    # Word2Vec
    if model_path.suffix == 'bin':
        model = keyedvectors.KeyedVectors.load_word2vec_format(model_path,
                                                               binary=True)
    elif model_path.suffix == 'txt' or model_path.suffix == 'vec':
        model = keyedvectors.KeyedVectors.load_word2vec_format(model_path,
                                                               binary=False)

    # FastText
    elif model_path.suffix == 'ftx':
        model = FastText.load(model_path)

    else:
        print('unsuported format of word embedding model', file=sys.stderr)
        sys.exit()

    print('load embedding model: vocab={}'.format(len(model.wv.vocab)),
          file=sys.stderr)
    return model
