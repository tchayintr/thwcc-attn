import argparse
from pathlib import Path
import sys

import constants


class ArgumentLoader(object):
    def __init__(self):
        # self.parser = argparse.ArgumentParser()
        pass

    def parse_args(self):
        all_args = self.get_full_parser().parse_args()
        min_args = self.get_minimum_parser(all_args).parse_args()
        return min_args

    def get_full_parser(self):
        parser = argparse.ArgumentParser()

        ### mode options
        parser.add_argument(
            '--execute_mode',
            '-x',
            choices=['train', 'decode'],
            help='Choose a mode from among \'train\', \'decode\'')
        # choices=['train', 'eval', 'decode'],
        # help='Choose a mode from among \'train\', \'eval\', \'decode\'')
        parser.add_argument('--task', '-t', help='Select a task')
        parser.add_argument(
            '--quiet',
            '-q',
            action='store_true',
            help='Do not output log file and serialized model file')

        ### gpu options
        parser.add_argument(
            '--gpu',
            '-g',
            type=int,
            default=0,
            help='GPU device id (use CPU if specify a negative value)')

        ### training parameters
        parser.add_argument(
            '--epoch_begin',
            type=int,
            default=1,
            help='Conduct training from i-th epoch (Default: 1)')
        parser.add_argument(
            '--epoch_end',
            '-e',
            type=int,
            default=5,
            help='Conduct training up to i-th epoch (Default: 5)')
        parser.add_argument(
            '--break_point',
            type=int,
            default=10000,
            help=
            'The number of instances which trained model is evaluated and saved (Default: 10000)'
        )
        parser.add_argument(
            '--batch_size',
            '-b',
            type=int,
            default=64,
            help='The number of examples in each mini-batch (Default: 64)')
        parser.add_argument(
            '--accumulate_grad_batches',
            type=int,
            default=1,
            help='The number of accumulated gradients (Default: 1)')
        parser.add_argument(
            '--grad_clip',
            type=float,
            default=5.0,
            help='Gradient norm threshold to clip (Default: 5.0)')

        ### optimizer parameters
        parser.add_argument(
            '--optimizer',
            '-o',
            default='adam',
            help=
            'Choose optimizing algorithm from among \'adam\', \'adedelta\', \'adagrad\', \'rmsprop\', and \'sgd\' (Default: adam)'
        )
        parser.add_argument(
            '--adam_alpha',
            type=float,
            default=0.001,
            help='alpha (learning rate) for Adam (Default: 0.001)')
        parser.add_argument('--adam_beta1',
                            type=float,
                            default=0.9,
                            help='beta1 for Adam (Default: 0.9)')
        parser.add_argument('--adam_beta2',
                            type=float,
                            default=0.999,
                            help='beta2 for Adam (Default: 0.999)')
        parser.add_argument(
            '--adam_weight_decay',
            type=float,
            default=0,
            help='Weight decay (L2 penalty) for Adam (Default: 0)')
        parser.add_argument(
            '--adagrad_lr',
            type=float,
            default=0.1,
            help='Initial learning rate for AdaGrad (Default: 0.1)')
        parser.add_argument(
            '--adagrad_lr_decay',
            type=float,
            default=0,
            help='Learning rate decay for AdaGrad (Default: 0)')
        parser.add_argument(
            '--adagrad_weight_decay',
            type=float,
            default=0,
            help='Weight decay (L2 penalty) for AdaGrad (Default: 0)')
        parser.add_argument(
            '--adadelta_lr',
            type=float,
            default=1.0,
            help='Initial learning rate for AdaDelta (Default: 1.0)')
        parser.add_argument('--adadelta_rho',
                            type=float,
                            default=0.9,
                            help='rho for AdaDelta (Default: 0.9)')
        parser.add_argument(
            '--adadelta_weight_decay',
            type=float,
            default=0,
            help='Weight decay (L2 penalty) for AdaDelta (Default: 0)')
        parser.add_argument(
            '--rmsprop_lr',
            type=float,
            default=0.1,
            help='Initial learning rate for RMSprop (Default: 0.1)')
        parser.add_argument('--rmsprop_alpha',
                            type=float,
                            default=0.99,
                            help='alpha for RMSprop (Default: 0.99)')
        parser.add_argument(
            '--rmsprop_weight_decay',
            type=float,
            default=0,
            help='Weight decay (L2 penalty) for RMSprop (Default: 0)')
        parser.add_argument(
            '--sgd_lr',
            type=float,
            default=0.1,
            help='Initial learning rate for SGD optimizers (Default: 0.1)')
        parser.add_argument('--sgd_momentum',
                            type=float,
                            default=0.0,
                            help='Momentum factor for SGD (Default: 0.0)')
        parser.add_argument(
            '--sgd_weight_decay',
            type=float,
            default=0,
            help='Weight decay (L2 penalty) for SGD (Default: 0)')

        ### scheduler parameters
        parser.add_argument(
            '--scheduler',
            '-s',
            default='exponential',
            help=
            'Choose scheduler for optimizer from among \'exponential\' (Default: exponential)'
        )
        parser.add_argument(
            '--exponential_gamma',
            type=float,
            default=0.99,
            help=
            'Multiplicative factor of learning rate decay (Learning rate decay) for Exponential (Default: 0.99)'
        )

        ### data paths and related options
        parser.add_argument(
            '--model_path',
            '-m',
            default=None,
            help=
            'pt/pkl file path of the trained model. \'xxx.hyp\' and \'xxx.s2i\' files are also read simultaneously when you specify \'xxx_izzz.pt/pkl\' file'
        )

        parser.add_argument('--input_data_path_prefix',
                            '-p',
                            dest='path_prefix',
                            default=None,
                            help='Path prefix of input data')
        parser.add_argument(
            '--train_data',
            default=None,
            help=
            'File path succeeding \'input_data_path_prefix\' of training data')
        parser.add_argument(
            '--valid_data',
            default=None,
            help=
            'File path succeeding \'input_data_path_prefix\' of validation data'
        )
        parser.add_argument(
            '--test_data',
            default=None,
            help='File path succeeding \'input_data_path_prefix\' of test data'
        )
        parser.add_argument(
            '--decode_data',
            default='',
            help=
            'File path of input text which succeeds \'input_data_path_prefix\''
        )
        parser.add_argument('--output_data',
                            default=None,
                            help='File path to output parsed text')
        parser.add_argument(
            '--input_data_format',
            '-f',
            choices=['sl'],
            default='sl',
            help='Choose format of input data from among \'sl\' (Default: sl)')
        parser.add_argument(
            '--output_data_format',
            choices=['sl'],
            default='sl',
            help='Choose format of output data from among \'sl\' (Default: sl)'
        )

        ### options for data pre/post-processing
        parser.add_argument('--lowercase_alphabets',
                            dest='lowercasing',
                            action='store_true',
                            help='Lowercase alphabets in input text')
        parser.add_argument(
            '--normalize_digits',
            action='store_true',
            help='Normalize digits by the same symbol in input text')

        ### model parameters
        parser.add_argument(
            '--token_freq_threshold',
            type=int,
            default=1,
            help=
            'Token frequency threshold. Tokens whose frequency are lower than the the threshold are regarded as unknown tokens (Default: 1)'
        )
        parser.add_argument(
            '--token_max_vocab_size',
            type=int,
            default=-1,
            help=
            'Maximum size of token vocaburaly. low frequency tokens are regarded as unknown tokens so that vocaburaly size does not exceed the specified size so much if set positive value (Default: -1)'
        )

        ### neural models
        parser.add_argument(
            '--pretrained_embed_usage',
            default='none',
            help=
            'Specify usage of pretrained embedding model from among \'init\' \'concat\' and \'add\''
        )

        return parser

    def get_minimum_parser(self, args):
        parser = argparse.ArgumentParser()

        # basic options
        self.add_basic_options(parser, args)

        # dependent options
        if args.execute_mode == 'train':
            self.add_train_mode_options(parser, args)
        elif args.execute_mode == 'eval':
            self.add_eval_mode_options(parser, args)
        elif args.execute_mode == 'decode':
            self.add_decode_mode_options(parser, args)
        else:
            print('Error: invalid execute mode: {}'.format(args.execute_mode),
                  file=sys.stderr)
            sys.exit()

        return parser

    def add_basic_options(self, parser, args):
        # mode options
        parser.add_argument('--execute_mode',
                            '-x',
                            required=True,
                            default=args.execute_mode)
        if args.model_path:
            parser.add_argument('--task', '-t', default=args.task)
        else:
            parser.add_argument('--task',
                                '-t',
                                required=True,
                                default=args.task)
        parser.add_argument('--quiet',
                            '-q',
                            action='store_true',
                            default=args.quiet)

        # options for data pre/post-processing
        parser.add_argument('--lowercase_alphabets',
                            dest='lowercasing',
                            action='store_true',
                            default=args.lowercasing)
        parser.add_argument('--normalize_digits',
                            action='store_true',
                            default=args.normalize_digits)

        # gpu options
        parser.add_argument('--gpu', '-g', type=int, default=args.gpu)

    def add_train_mode_options(self, parser, args):
        # training parameters
        parser.add_argument('--epoch_begin',
                            type=int,
                            default=args.epoch_begin)
        parser.add_argument('--epoch_end',
                            '-e',
                            type=int,
                            default=args.epoch_end)
        parser.add_argument('--break_point',
                            type=int,
                            default=args.break_point)
        parser.add_argument('--batch_size',
                            '-b',
                            type=int,
                            default=args.batch_size)
        parser.add_argument('--accumulate_grad_batches',
                            type=int,
                            default=args.accumulate_grad_batches)

        # data paths and related options
        parser.add_argument('--model_path',
                            '-m',
                            type=Path,
                            default=args.model_path)
        parser.add_argument('--input_data_path_prefix',
                            '-p',
                            dest='path_prefix',
                            type=Path,
                            default=args.path_prefix)
        parser.add_argument('--train_data',
                            type=Path,
                            required=True,
                            default=args.train_data)
        parser.add_argument('--valid_data', type=Path, default=args.valid_data)
        self.add_input_data_format_option(parser, args)

        # model parameters
        parser.add_argument('--token_freq_threshold',
                            type=int,
                            default=args.token_freq_threshold)
        parser.add_argument('--token_max_vocab_size',
                            type=int,
                            default=args.token_max_vocab_size)
        parser.add_argument('--pretrained_embed_usage',
                            default=args.pretrained_embed_usage)

        # optimizer parameters
        parser.add_argument('--optimizer', '-o', default=args.optimizer)
        parser.add_argument('--grad_clip', type=float, default=args.grad_clip)

        if args.optimizer == 'adam':
            self.add_adam_options(parser, args)
        elif args.optimizer == 'adadelta':
            self.add_adadelta_options(parser, args)
        elif args.optimizer == 'adagrad':
            self.add_adagrad_options(parser, args)
        elif args.optimizer == 'rmsprop':
            self.add_rmsprop_options(parser, args)
        elif args.optimizer == 'sgd':
            self.add_sgd_options(parser, args)
        else:
            print('Error: invalid optimizer name: {}'.format(args.optimizer),
                  file=sys.stderr)
            sys.exit()

        # scheduler parameters
        parser.add_argument('--scheduler', '-s', default=args.scheduler)

        if args.scheduler == 'exponential':
            self.add_exponential_options(parser, args)
        else:
            print('Error: invalid scheduler name: {}'.format(args.scheduler),
                  file=sys.stderr)
            sys.exit()

    def add_eval_mode_options(self, parser, args):
        # evaluation parameters
        parser.add_argument('--batch_size',
                            '-b',
                            type=int,
                            default=args.batch_size)
        parser.add_argument('--accumulate_grad_batches',
                            type=int,
                            default=args.accumulate_grad_batches)

        # data paths and related options
        parser.add_argument('--model_path',
                            '-m',
                            type=Path,
                            required=True,
                            default=args.model_path)
        parser.add_argument('--input_data_path_prefix',
                            '-p',
                            dest='path_prefix',
                            type=Path,
                            default=args.path_prefix)
        parser.add_argument('--train_data', type=Path, default=args.train_data)
        parser.add_argument('--test_data',
                            type=Path,
                            required=True,
                            default=args.test_data)
        self.add_input_data_format_option(parser, args)

    def add_decode_mode_options(self, parser, args):
        # decoding parameters
        parser.add_argument('--batch_size',
                            '-b',
                            type=int,
                            default=args.batch_size)

        # data paths and related options
        parser.add_argument('--model_path',
                            '-m',
                            type=Path,
                            required=True,
                            default=args.model_path)
        parser.add_argument('--input_data_path_prefix',
                            '-p',
                            dest='path_prefix',
                            type=Path,
                            default=args.path_prefix)
        parser.add_argument('--decode_data',
                            type=Path,
                            required=True,
                            default=args.decode_data)
        parser.add_argument('--output_data',
                            '-o',
                            type=Path,
                            default=args.output_data)
        self.add_input_data_format_option(parser, args)
        self.add_output_data_format_options(parser, args)

    def add_input_data_format_option(self, parser, args):
        parser.add_argument('--input_data_format',
                            '-f',
                            default=args.input_data_format)

    def add_output_data_format_options(self, parser, args):
        if args.output_data_format == 'sl':
            parser.add_argument('--output_data_format',
                                default=args.output_data_format)
            parser.add_argument('--output_token_delim',
                                default=constants.SL_TOKEN_DELIM)
        else:
            print(
                'Error: output data format must be specified from among {sl}: {}'
                .format(args.output_data_format),
                file=sys.stderr)
            sys.exit()

    def add_adam_options(self, parser, args):
        parser.add_argument('--adam_alpha',
                            type=float,
                            default=args.adam_alpha)
        parser.add_argument('--adam_beta1',
                            type=float,
                            default=args.adam_beta1)
        parser.add_argument('--adam_beta2',
                            type=float,
                            default=args.adam_beta2)
        parser.add_argument('--adam_weight_decay',
                            type=float,
                            default=args.adam_weight_decay)

    def add_adadelta_options(self, parser, args):
        parser.add_argument('--adadelta_lr',
                            type=float,
                            default=args.adadelta_lr)
        parser.add_argument('--adadelta_rho',
                            type=float,
                            default=args.adadelta_rho)
        parser.add_argument('--adadelta_weight_decay',
                            type=float,
                            default=args.adadelta_weight_decay)

    def add_adagrad_options(self, parser, args):
        parser.add_argument('--adagrad_lr',
                            type=float,
                            default=args.adagrad_lr)
        parser.add_argument('--adagrad_lr_decay',
                            type=float,
                            default=args.adagrad_lr_decay)
        parser.add_argument('--adagrad_weight_decay',
                            type=float,
                            default=args.adagrad_weight_decay)

    def add_rmsprop_options(self, parser, args):
        parser.add_argument('--rmsprop_lr',
                            type=float,
                            default=args.rmsprop_lr)
        parser.add_argument('--rmsprop_alpha',
                            type=float,
                            default=args.rmsprop_alpha)
        parser.add_argument('--rmsprop_weight_decay',
                            type=float,
                            default=args.rmsprop_weight_decay)

    def add_sgd_options(self, parser, args):
        parser.add_argument('--sgd_lr', type=float, default=args.sgd_lr)
        parser.add_argument('--sgd_momentum',
                            type=float,
                            default=args.sgd_momentum)
        parser.add_argument('--sgd_weight_decay',
                            type=float,
                            default=args.sgd_weight_decay)

    def add_exponential_options(self, parser, args):
        parser.add_argument('--exponential_gamma',
                            type=float,
                            default=args.exponential_gamma)
