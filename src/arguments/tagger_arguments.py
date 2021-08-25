from pathlib import Path
import sys

from arguments.arguments import ArgumentLoader
import constants


class TaggerArgumentLoader(ArgumentLoader):
    def parse_args(self):
        return super().parse_args()

    def get_full_parser(self):
        parser = super().get_full_parser()

        # data paths and related options
        parser.add_argument(
            '--unigram_embed_model_path',
            type=Path,
            default=None,
            help=
            'File path of pretrained model of token (character or word) unigram embedding'
        )
        parser.add_argument(
            '--bigram_embed_model_path',
            type=Path,
            default=None,
            help=
            'File path of pretrained model of token (character or word) bigram embedding'
        )
        parser.add_argument(
            '--external_dic_path',
            type=Path,
            default=None,
            help='File path of external word dictionary listing known words')
        parser.add_argument(
            '--subword_dic_path',
            type=Path,
            default=None,
            help=
            'File path of subword dictionary particulalrly for the \'sub-combinative\' tagging unit'
        )

        # sequence labeling
        parser.add_argument(
            '--tagging_unit',
            default='single',
            help='Specify tagging unit, \'single\' for character-based model, '
            + '\'hybrid\' for hybrid model of character and word, ' +
            '\'mutant\' for mutant model of character and cc' +
            '\'sub-combinative\' for combinative model of character, word, and subword'
            +
            '\'combinative\' for combinative model of character, word and cc' +
            '\'transformer\'')

        # model parameters
        # options for model architecture and parameters
        # common
        parser.add_argument(
            '--embed_dropout',
            type=float,
            default=0.0,
            help='Dropout ratio for embedding layers (Default: 0.0)')
        parser.add_argument(
            '--rnn_dropout',
            type=float,
            default=0.2,
            help='Dropout ratio for RNN vertical layers (Default: 0.2)')
        parser.add_argument(
            '--rnn_unit_type',
            default='lstm',
            help=
            'Choose unit type of RNN from among \'lstm\', \'gru\' and \'plain (tanh) \' (Default: lstm)'
        )
        parser.add_argument('--rnn_bidirection',
                            action='store_true',
                            help='Use bidirectional RNN (Default: False)')
        parser.add_argument(
            '--rnn_batch_first',
            action='store_true',
            help=
            'To provide the input and output tensor as (batch, seq, feature) (Default: False)'
        )
        parser.add_argument('--rnn_n_layers',
                            type=int,
                            default=1,
                            help='The number of RNN layers (Default: 1)')
        parser.add_argument(
            '--rnn_hidden_size',
            type=int,
            default=256,
            help='The size of hidden units (dimension) for RNN (Default: 256)')

        # segmentation evaluation
        parser.add_argument('--evaluation_method',
                            default='normal',
                            help='Evaluation method for segmentation')

        # segmentation/tagging
        parser.add_argument(
            '--bigram_freq_threshold',
            type=int,
            default=1,
            help=
            'Token bigram frequency threshold. Bigrams whose frequency are lower than the the threshold are regarded as unknown bigrams (Default: 1)'
        )
        parser.add_argument(
            '--bigram_max_vocab_size',
            type=int,
            default=-1,
            help=
            'Maximum size of token bigram vocabulary. low frequency bigrams  are regarded as unknown bigrams so that vocabulary size does not exceed the specified size so much if set positive value (Default: -1)'
        )
        parser.add_argument(
            '--mlp_dropout',
            type=float,
            default=0.0,
            help=
            'Dropout ratio for MLP of sequence labeling model (Default: 0.0)')
        parser.add_argument(
            '--unigram_embed_size',
            type=int,
            default=128,
            help=
            'The size (dimension) of token (character or word) unigram embedding (Default: 128)'
        )
        parser.add_argument(
            '--bigram_embed_size',
            type=int,
            default=0,
            help=
            'The size (dimension) of token (character or word) biigram embedding (Default: 0)'
        )
        parser.add_argument(
            '--mlp_n_layers',
            type=int,
            default=1,
            help=
            'The number of layers of MLP of sequence labeling model. The last layer projects input hidden vector to dimensional space of number of labels (Default: 1)'
        )
        parser.add_argument(
            '--mlp_hidden_size',
            type=int,
            default=300,
            help=
            'The size of hidden units (dimension) of MLP of sequence labeling model (Default: 300)'
        )
        parser.add_argument(
            '--mlp_activation',
            default='relu',
            help=
            'Choose type of activation function for Muti-layer perceptron from between'
            + '\'sigmoid\' and \'relu\' (Default: relu)')
        parser.add_argument(
            '--inference_layer_type',
            dest='inference_layer',
            default='crf',
            help=
            'Choose type of inference layer for sequence labeling model from between '
            + '\'softmax\' and \'crf\' (Default: crf)')

        # hybrid/mutant segmentation
        parser.add_argument(
            '--biaffine_type',
            default='',
            help=
            'Specify \'u\', \'v\', and/or \'b\' if use non-zero matrices/vectors in biaffine transformation \'x1*W*x2+x1*U+x2*V+b\' such as \'uv\''
        )
        parser.add_argument('--rnn_n_layers2',
                            type=int,
                            default=0,
                            help='The number of 2nd RNN layers (Default: 0)')
        parser.add_argument(
            '--rnn_hidden_size2',
            type=int,
            default=0,
            help='The size of hidden units (dimension) for 2nd RNN (Default: 0)'
        )
        parser.add_argument(
            '--biaffine_dropout',
            type=float,
            default=0.0,
            help=
            'Dropout ratio for biaffine layer in hybrid model (Default: 0.0)')

        # hybrid segmentation
        parser.add_argument(
            '--chunk_freq_threshold',
            type=int,
            default=1,
            help=
            'Chunk frequency threshold. Chunks whose frequency are lower than the the threshold are regarded as unknown chunks (Default: 1)'
        )
        parser.add_argument(
            '--chunk_max_vocab_size',
            type=int,
            default=-1,
            help=
            'Maximum size of chunk vocaburaly. low frequency chunks are regarded as unknown chunks so that vocaburaly size does not exceed the specified size so much if set positive value (Default: -1)'
        )
        parser.add_argument(
            '--chunk_embed_size',
            type=int,
            default=300,
            help=
            'The size (dimension) of chunk (typically word) embedding (Default: 300)'
        )
        parser.add_argument(
            '--chunk_embed_model_path',
            type=Path,
            default=None,
            help=
            'File path of pretrained model of chunk (typically word) embedding'
        )
        parser.add_argument(
            '--chunk_pooling_type',
            default=constants.WAVG,
            help=
            'Specify integration method of chunk vector from among {avg, wavg, con, wcon} (Default: wavg)'
        )
        parser.add_argument(
            '--min_chunk_len',
            type=int,
            default=1,
            help=
            'Specify minimum chunk length used in hybird segmentation model (generated by character tokens)'
        )
        parser.add_argument(
            '--max_chunk_len',
            type=int,
            default=4,
            help=
            'Specify maximum chunk length used in hybrid segmentation model (generated by character tokens)'
        )
        parser.add_argument('--use_gold_chunk', action='store_true')
        parser.add_argument(
            '--chunk_vector_dropout',
            type=float,
            default=0.0,
            help=
            'Dropout ratio for chunk vectors in hybrid model (Default: 0.0)')
        parser.add_argument('--unuse_nongold_chunk',
                            action='store_true',
                            help='To unuse non-gold chunk')
        parser.add_argument('--ignore_unknown_pretrained_chunk',
                            action='store_true',
                            help='To ignore unknow pretrained chunk')
        parser.add_argument(
            '--gen_oov_chunk_for_test',
            action='store_true',
            help='To generate Out-of-vocabulary chunk for testing')

        # mutant segmentation
        parser.add_argument(
            '--cc_freq_threshold',
            type=int,
            default=1,
            help=
            'Character-cluster frequency threshold. Character-cluster whose frequency are lower than the the threshold are regarded as unknown character-cluster (Default: 1)'
        )
        parser.add_argument(
            '--cc_max_vocab_size',
            type=int,
            default=-1,
            help=
            'Maximum size of character-cluster vocaburaly. low frequency character clusters are regarded as unknown character clusters so that vocaburaly size does not exceed the specified size so much if set positive value (Default: -1)'
        )
        parser.add_argument(
            '--cc_embed_size',
            type=int,
            default=300,
            help=
            'The size (dimension) of character-cluster embedding (Default: 300)'
        )
        parser.add_argument(
            '--cc_embed_model_path',
            type=Path,
            default=None,
            help='File path of pretrained model of character-cluster embedding'
        )
        parser.add_argument(
            '--cc_pooling_type',
            default=constants.CCAVG,
            help=
            'Specify integration method of character-cluster vector from among {avg, ccavg, con, cccon} (Default: ccavg)'
        )
        parser.add_argument(
            '--min_cc_len',
            type=int,
            default=1,
            help=
            'Specify minimum character-cluster length used in mutant segmentation model (generated by character tokens)'
        )
        parser.add_argument(
            '--max_cc_len',
            type=int,
            default=4,
            help=
            'Specify maximum character-cluster length used in mutant segmentation model (generated by character tokens)'
        )
        parser.add_argument('--use_gold_cc', action='store_true')
        parser.add_argument(
            '--cc_vector_dropout',
            type=float,
            default=0.0,
            help=
            'Dropout ratio for character-cluster vectors in mutant model (Default: 0.0)'
        )
        parser.add_argument('--unuse_nongold_cc',
                            action='store_true',
                            help='To unuse non-gold character-cluster')
        parser.add_argument(
            '--ignore_unknown_pretrained_cc',
            action='store_true',
            help='To ignore unknow pretrained character-cluster')
        parser.add_argument(
            '--gen_oov_cc_for_test',
            action='store_true',
            help='To generate Out-of-vocabulary character-cluster for testing')

        # sub-combinative segmentation
        parser.add_argument(
            '--subword_freq_threshold',
            type=int,
            default=1,
            help=
            'Subword frequency threshold. Subword whose frequency are lower than the threshold are regarded as unknown subwords (Default: 1)'
        )
        parser.add_argument(
            '--subword_max_vocab_size',
            type=int,
            default=-1,
            help=
            'Maximum size of subword vocaburaly. low frequency subwords are regarded as unknown subwords so that vocaburaly size does not exceed the specified size so much if set positive value (Default: -1)'
        )
        parser.add_argument(
            '--subword_embed_size',
            type=int,
            default=300,
            help='The size (dimension) of subword embedding (Default: 300)')
        parser.add_argument(
            '--subword_embed_model_path',
            type=Path,
            default=None,
            help='File path of pretrained model of subword embedding')
        parser.add_argument(
            '--subword_pooling_type',
            default=constants.SWAVG,
            help=
            'Specify integration method of subword vector from among {avg, swavg, con, swcon} (Default: swavg)'
        )
        parser.add_argument(
            '--min_subword_len',
            type=int,
            default=1,
            help=
            'Specify minimum subword length used in sub-combinative segmentation model (generated by character tokens)'
        )
        parser.add_argument(
            '--max_subword_len',
            type=int,
            default=4,
            help=
            'Specify maximum subword length used in sub-combinative segmentation model (generated by character tokens)'
        )
        parser.add_argument('--use_gold_subword', action='store_true')
        parser.add_argument(
            '--subword_vector_dropout',
            type=float,
            default=0.0,
            help=
            'Dropout ratio for subword vectors in sub-combinative model (Default: 0.0)'
        )
        parser.add_argument('--unuse_nongold_subword',
                            action='store_true',
                            help='To unuse non-gold subword')
        parser.add_argument('--ignore_unknown_pretrained_subword',
                            action='store_true',
                            help='To ignore unknow pretrained subword')
        parser.add_argument(
            '--gen_oov_subword_for_test',
            action='store_true',
            help='To generate Out-of-vocabulary subowrd for testing')

        # combinative segmentation
        parser.add_argument('--rnn_n_layers3',
                            type=int,
                            default=0,
                            help='The number of 3rd RNN layers (Default: 0)')
        parser.add_argument(
            '--rnn_hidden_size3',
            type=int,
            default=0,
            help='The size (dimension) of hidden units for 3rd RNN (Default: 0)'
        )
        parser.add_argument(
            '--reverse',
            action='store_true',
            help=
            'To reverse the order of word leyer and additional layer (cc and subword)'
        )

        # transformer based segmentation
        parser.add_argument(
            '--tfm_n_layers',
            type=int,
            default=6,
            help='The number of Transformer layers (Default: 12)')
        parser.add_argument(
            '--tfm_ff_hidden_size',
            type=int,
            default=2048,
            help='The size of feed-forward hidden unit (Default: 768)')
        parser.add_argument('--tfm_hidden_size',
                            type=int,
                            default=512,
                            help='The size of hidden unit (Default: 768)')
        parser.add_argument(
            '--tfm_n_heads',
            type=int,
            default=8,
            help=
            'The number of attention heads in Transformer layers (Default: 12)'
        )
        parser.add_argument(
            '--tfm_dropout',
            type=float,
            default=0.0,
            help='Dropout ratio for Transformer layers (Default: 0.0)')
        parser.add_argument(
            '--max_seq_len',
            type=int,
            default=5000,
            help=
            'Maximum length of sequence of in Transformer Positional Encoding (Default: 500)'
        )

        return parser

    def get_minimum_parser(self, args):
        parser = super().get_minimum_parser(args)
        parser.add_argument('--evaluation_method',
                            default=args.evaluation_method)
        if not (args.evaluation_method == 'normal' or args.evaluation_method
                == 'each_length' or args.evaluation_method == 'each_vocab'
                or args.evaluation_method == 'attention'
                or args.evaluation_method == 'stat_test'):
            print(
                'Error: evaluation_method must be specified among from {normal, each_length, each_vocab, attention}:{}'
                .format(args.evaluation_method),
                file=sys.stderr)
            sys.exit()

        parser.add_argument('--unigram_embed_model_path',
                            type=Path,
                            default=args.unigram_embed_model_path)
        parser.add_argument('--embed_dropout',
                            type=float,
                            default=args.embed_dropout)
        parser.add_argument('--rnn_dropout',
                            type=float,
                            default=args.rnn_dropout)

        # specific options for segmentation/tagging
        parser.add_argument('--bigram_embed_model_path',
                            type=Path,
                            default=args.bigram_embed_model_path)
        parser.add_argument('--external_dic_path',
                            type=Path,
                            default=args.external_dic_path)
        parser.add_argument('--subword_dic_path',
                            type=Path,
                            default=args.subword_dic_path)
        parser.add_argument('--mlp_dropout',
                            type=float,
                            default=args.mlp_dropout)
        parser.add_argument('--tagging_unit', default=args.tagging_unit)

        # specific options for transformer based segmentation
        if args.tagging_unit == 'transformer':
            parser.add_argument('--tfm_dropout',
                                type=float,
                                default=args.tfm_dropout)

        # specific options for hybrid/mutant/combinative unit segmentation/tagging
        if args.tagging_unit == 'hybrid':
            parser.add_argument('--biaffine_dropout',
                                type=float,
                                default=args.biaffine_dropout)
            parser.add_argument('--chunk_vector_dropout',
                                type=float,
                                default=args.chunk_vector_dropout)
            parser.add_argument('--chunk_embed_model_path',
                                default=args.chunk_embed_model_path)
            parser.add_argument('--gen_oov_chunk_for_test',
                                action='store_const',
                                const=False,
                                default=False)

        elif args.tagging_unit == 'mutant':
            parser.add_argument('--biaffine_dropout',
                                type=float,
                                default=args.biaffine_dropout)
            parser.add_argument('--cc_vector_dropout',
                                type=float,
                                default=args.cc_vector_dropout)
            parser.add_argument('--cc_embed_model_path',
                                default=args.cc_embed_model_path)
            parser.add_argument('--gen_oov_cc_for_test',
                                action='store_const',
                                const=False,
                                default=False)

        elif args.tagging_unit == 'combinative':
            parser.add_argument('--biaffine_dropout',
                                type=float,
                                default=args.biaffine_dropout)
            parser.add_argument('--chunk_vector_dropout',
                                type=float,
                                default=args.chunk_vector_dropout)
            parser.add_argument('--chunk_embed_model_path',
                                default=args.chunk_embed_model_path)
            parser.add_argument('--gen_oov_chunk_for_test',
                                action='store_const',
                                const=False,
                                default=False)
            parser.add_argument('--cc_vector_dropout',
                                type=float,
                                default=args.cc_vector_dropout)
            parser.add_argument('--cc_embed_model_path',
                                default=args.cc_embed_model_path)
            parser.add_argument('--gen_oov_cc_for_test',
                                action='store_const',
                                const=False,
                                default=False)

        elif args.tagging_unit == 'sub-combinative':
            parser.add_argument('--biaffine_dropout',
                                type=float,
                                default=args.biaffine_dropout)
            parser.add_argument('--chunk_vector_dropout',
                                type=float,
                                default=args.chunk_vector_dropout)
            parser.add_argument('--chunk_embed_model_path',
                                default=args.chunk_embed_model_path)
            parser.add_argument('--gen_oov_chunk_for_test',
                                action='store_const',
                                const=False,
                                default=False)
            parser.add_argument('--subword_vector_dropout',
                                type=float,
                                default=args.subword_vector_dropout)
            parser.add_argument('--subword_embed_model_path',
                                default=args.subword_embed_model_path)
            parser.add_argument('--gen_oov_subword_for_test',
                                action='store_const',
                                const=False,
                                default=False)

        if args.execute_mode == 'train':
            if (not args.model_path and (args.unigram_embed_model_path
                                         or args.bigram_embed_model_path)):
                if not (args.pretrained_embed_usage == 'init'
                        or args.pretrained_embed_usage == 'concat'
                        or args.pretrained_embed_usage == 'add'):
                    print(
                        'Error: pretrained_embed_usage must be specified among from {init, concat, add}:{}'
                        .format(args.pretrained_embed_usage),
                        file=sys.stderr)
                    sys.exit()

                if args.unigram_embed_model_path and args.unigram_embed_size <= 0:
                    print(
                        'Error: unigram_embed_size must be positive value to use pretrained unigram embed model: {}'
                        .format(args.unigram_embed_size),
                        file=sys.stderr)
                    sys.exit()

                if args.bigram_embed_model_path and args.bigram_embed_size <= 0:
                    print(
                        'Error: bigram_embed_size must be positive value to use pretrained bigram embed model: {}'
                        .format(args.bigram_embed_size),
                        file=sys.stderr)
                    sys.exit()

            parser.add_argument('--unigram_embed_size',
                                type=int,
                                default=args.unigram_embed_size)
            parser.add_argument('--rnn_unit_type', default=args.rnn_unit_type)
            parser.add_argument('--rnn_bidirection',
                                action='store_true',
                                default=args.rnn_bidirection)
            parser.add_argument('--rnn_batch_first',
                                action='store_true',
                                default=args.rnn_batch_first)
            parser.add_argument('--rnn_n_layers',
                                type=int,
                                default=args.rnn_n_layers)
            parser.add_argument('--rnn_hidden_size',
                                type=int,
                                default=args.rnn_hidden_size)

            # specific options for segmentation/tagging
            parser.add_argument('--bigram_freq_threshold',
                                type=int,
                                default=args.bigram_freq_threshold)
            parser.add_argument('--bigram_max_vocab_size',
                                type=int,
                                default=args.bigram_max_vocab_size)
            parser.add_argument('--bigram_embed_size',
                                type=int,
                                default=args.bigram_embed_size)
            parser.add_argument('--mlp_n_layers',
                                type=int,
                                default=args.mlp_n_layers)
            parser.add_argument('--mlp_hidden_size',
                                type=int,
                                default=args.mlp_hidden_size)
            parser.add_argument('--mlp_activation',
                                default=args.mlp_activation)
            parser.add_argument('--inference_layer_type',
                                dest='inference_layer',
                                default=args.inference_layer)

            # specific options for transformer unit segmentation
            if args.tagging_unit == 'transformer':
                parser.add_argument('--tfm_n_layers',
                                    type=int,
                                    default=args.tfm_n_layers)
                parser.add_argument('--tfm_ff_hidden_size',
                                    type=int,
                                    default=args.tfm_ff_hidden_size)
                parser.add_argument('--tfm_hidden_size',
                                    type=int,
                                    default=args.tfm_hidden_size)
                parser.add_argument('--tfm_n_heads',
                                    type=int,
                                    default=args.tfm_n_heads)
                parser.add_argument('--max_seq_len',
                                    type=int,
                                    default=args.max_seq_len)

            # specific options for hybrid/combinative unit segmentation
            elif args.tagging_unit == 'hybrid':
                parser.add_argument('--biaffine_type',
                                    default=args.biaffine_type)
                parser.add_argument('--chunk_embed_size',
                                    type=int,
                                    default=args.chunk_embed_size)
                parser.add_argument('--chunk_freq_threshold',
                                    type=int,
                                    default=args.chunk_freq_threshold)
                parser.add_argument('--chunk_max_vocab_size',
                                    type=int,
                                    default=args.chunk_max_vocab_size)
                parser.add_argument('--rnn_n_layers2',
                                    type=int,
                                    default=args.rnn_n_layers2)
                parser.add_argument('--rnn_hidden_size2',
                                    type=int,
                                    default=args.rnn_hidden_size2)

                chunk_pooling_type = args.chunk_pooling_type.upper()
                if (chunk_pooling_type == constants.AVG
                        or chunk_pooling_type == constants.WAVG
                        or chunk_pooling_type == constants.CON
                        or chunk_pooling_type == constants.WCON):
                    parser.add_argument('--chunk_pooling_type',
                                        default=chunk_pooling_type)
                else:
                    print(
                        'Error: chunk pooling type must be specified among from {AVG, WAVG, CON, WCON}. Input: {}'
                        .format(args.chunk_pooling_type),
                        file=sys.stderr)
                    sys.exit()

                parser.add_argument('--min_chunk_len',
                                    type=int,
                                    default=args.min_chunk_len)
                parser.add_argument('--max_chunk_len',
                                    type=int,
                                    default=args.max_chunk_len)
                parser.add_argument('--use_gold_chunk',
                                    action='store_true',
                                    default=args.use_gold_chunk)
                parser.add_argument('--unuse_nongold_chunk',
                                    action='store_true',
                                    default=args.unuse_nongold_chunk)
                parser.add_argument(
                    '--ignore_unknown_pretrained_chunk',
                    action='store_true',
                    default=args.ignore_unknown_pretrained_chunk)

            elif args.tagging_unit == 'mutant':
                parser.add_argument('--biaffine_type',
                                    default=args.biaffine_type)
                parser.add_argument('--cc_embed_size',
                                    type=int,
                                    default=args.cc_embed_size)
                parser.add_argument('--cc_freq_threshold',
                                    type=int,
                                    default=args.cc_freq_threshold)
                parser.add_argument('--cc_max_vocab_size',
                                    type=int,
                                    default=args.cc_max_vocab_size)
                parser.add_argument('--rnn_n_layers2',
                                    type=int,
                                    default=args.rnn_n_layers2)
                parser.add_argument('--rnn_hidden_size2',
                                    type=int,
                                    default=args.rnn_hidden_size2)

                cc_pooling_type = args.cc_pooling_type.upper()
                if (cc_pooling_type == constants.AVG
                        or cc_pooling_type == constants.CCAVG
                        or cc_pooling_type == constants.CON
                        or cc_pooling_type == constants.CCCON):
                    parser.add_argument('--cc_pooling_type',
                                        default=cc_pooling_type)
                else:
                    print(
                        'Error: character-cluster pooling type must be specified among from {AVG, CCAVG, CON, CCCON}. Input: {}'
                        .format(args.cc_pooling_type),
                        file=sys.stderr)
                    sys.exit()

                parser.add_argument('--min_cc_len',
                                    type=int,
                                    default=args.min_cc_len)
                parser.add_argument('--max_cc_len',
                                    type=int,
                                    default=args.max_cc_len)
                parser.add_argument('--use_gold_cc',
                                    action='store_true',
                                    default=args.use_gold_cc)
                parser.add_argument('--unuse_nongold_cc',
                                    action='store_true',
                                    default=args.unuse_nongold_cc)
                parser.add_argument('--ignore_unknown_pretrained_cc',
                                    action='store_true',
                                    default=args.ignore_unknown_pretrained_cc)

            elif args.tagging_unit == 'sub-combinative':
                parser.add_argument('--biaffine_type',
                                    default=args.biaffine_type)
                parser.add_argument('--chunk_embed_size',
                                    type=int,
                                    default=args.chunk_embed_size)
                parser.add_argument('--subword_embed_size',
                                    type=int,
                                    default=args.subword_embed_size)
                parser.add_argument('--chunk_freq_threshold',
                                    type=int,
                                    default=args.chunk_freq_threshold)
                parser.add_argument('--subword_freq_threshold',
                                    type=int,
                                    default=args.subword_freq_threshold)
                parser.add_argument('--chunk_max_vocab_size',
                                    type=int,
                                    default=args.chunk_max_vocab_size)
                parser.add_argument('--subword_max_vocab_size',
                                    type=int,
                                    default=args.subword_max_vocab_size)
                parser.add_argument('--rnn_n_layers2',
                                    type=int,
                                    default=args.rnn_n_layers2)
                parser.add_argument('--rnn_hidden_size2',
                                    type=int,
                                    default=args.rnn_hidden_size2)
                parser.add_argument('--rnn_n_layers3',
                                    type=int,
                                    default=args.rnn_n_layers3)
                parser.add_argument('--rnn_hidden_size3',
                                    type=int,
                                    default=args.rnn_hidden_size3)
                parser.add_argument('--reverse',
                                    action='store_true',
                                    default=args.reverse)

                chunk_pooling_type = args.chunk_pooling_type.upper()
                if (chunk_pooling_type == constants.AVG
                        or chunk_pooling_type == constants.WAVG
                        or chunk_pooling_type == constants.CON
                        or chunk_pooling_type == constants.WCON):
                    parser.add_argument('--chunk_pooling_type',
                                        default=chunk_pooling_type)
                else:
                    print(
                        'Error: chunk pooling type must be specified among from {AVG, WAVG, CON, WCON}. Input: {}'
                        .format(args.chunk_pooling_type),
                        file=sys.stderr)
                    sys.exit()

                subword_pooling_type = args.subword_pooling_type.upper()
                if (subword_pooling_type == constants.AVG
                        or subword_pooling_type == constants.SWAVG
                        or subword_pooling_type == constants.CON
                        or subword_pooling_type == constants.SWCON):
                    parser.add_argument('--subword_pooling_type',
                                        default=subword_pooling_type)
                else:
                    print(
                        'Error: subword pooling type must be specified among from {AVG, SWAVG, CON, SWCON}. Input: {}'
                        .format(args.subword_pooling_type),
                        file=sys.stderr)
                    sys.exit()

                parser.add_argument('--min_chunk_len',
                                    type=int,
                                    default=args.min_chunk_len)
                parser.add_argument('--max_chunk_len',
                                    type=int,
                                    default=args.max_chunk_len)
                parser.add_argument('--min_subword_len',
                                    type=int,
                                    default=args.min_subword_len)
                parser.add_argument('--max_subword_len',
                                    type=int,
                                    default=args.max_subword_len)
                parser.add_argument('--use_gold_chunk',
                                    action='store_true',
                                    default=args.use_gold_chunk)
                parser.add_argument('--use_gold_subword',
                                    action='store_true',
                                    default=args.use_gold_subword)
                parser.add_argument('--unuse_nongold_chunk',
                                    action='store_true',
                                    default=args.unuse_nongold_chunk)
                parser.add_argument('--unuse_nongold_subword',
                                    action='store_true',
                                    default=args.unuse_nongold_subword)
                parser.add_argument(
                    '--ignore_unknown_pretrained_chunk',
                    action='store_true',
                    default=args.ignore_unknown_pretrained_chunk)
                parser.add_argument(
                    '--ignore_unknown_pretrained_subword',
                    action='store_true',
                    default=args.ignore_unknown_pretrained_subword)

            elif args.tagging_unit == 'combinative':
                parser.add_argument('--biaffine_type',
                                    default=args.biaffine_type)
                parser.add_argument('--chunk_embed_size',
                                    type=int,
                                    default=args.chunk_embed_size)
                parser.add_argument('--cc_embed_size',
                                    type=int,
                                    default=args.cc_embed_size)
                parser.add_argument('--chunk_freq_threshold',
                                    type=int,
                                    default=args.chunk_freq_threshold)
                parser.add_argument('--cc_freq_threshold',
                                    type=int,
                                    default=args.cc_freq_threshold)
                parser.add_argument('--chunk_max_vocab_size',
                                    type=int,
                                    default=args.chunk_max_vocab_size)
                parser.add_argument('--cc_max_vocab_size',
                                    type=int,
                                    default=args.cc_max_vocab_size)
                parser.add_argument('--rnn_n_layers2',
                                    type=int,
                                    default=args.rnn_n_layers2)
                parser.add_argument('--rnn_hidden_size2',
                                    type=int,
                                    default=args.rnn_hidden_size2)
                parser.add_argument('--rnn_n_layers3',
                                    type=int,
                                    default=args.rnn_n_layers3)
                parser.add_argument('--rnn_hidden_size3',
                                    type=int,
                                    default=args.rnn_hidden_size3)
                parser.add_argument('--reverse',
                                    action='store_true',
                                    default=args.reverse)

                chunk_pooling_type = args.chunk_pooling_type.upper()
                if (chunk_pooling_type == constants.AVG
                        or chunk_pooling_type == constants.WAVG
                        or chunk_pooling_type == constants.CON
                        or chunk_pooling_type == constants.WCON):
                    parser.add_argument('--chunk_pooling_type',
                                        default=chunk_pooling_type)
                else:
                    print(
                        'Error: chunk pooling type must be specified among from {AVG, WAVG, CON, WCON}. Input: {}'
                        .format(args.chunk_pooling_type),
                        file=sys.stderr)
                    sys.exit()

                cc_pooling_type = args.cc_pooling_type.upper()
                if (cc_pooling_type == constants.AVG
                        or cc_pooling_type == constants.CCAVG
                        or cc_pooling_type == constants.CON
                        or cc_pooling_type == constants.CCCON):
                    parser.add_argument('--cc_pooling_type',
                                        default=cc_pooling_type)
                else:
                    print(
                        'Error: character-cluster pooling type must be specified among from {AVG, CCAVG, CON, CCCON}. Input: {}'
                        .format(args.cc_pooling_type),
                        file=sys.stderr)
                    sys.exit()

                parser.add_argument('--min_chunk_len',
                                    type=int,
                                    default=args.min_chunk_len)
                parser.add_argument('--max_chunk_len',
                                    type=int,
                                    default=args.max_chunk_len)
                parser.add_argument('--min_cc_len',
                                    type=int,
                                    default=args.min_cc_len)
                parser.add_argument('--max_cc_len',
                                    type=int,
                                    default=args.max_cc_len)
                parser.add_argument('--use_gold_chunk',
                                    action='store_true',
                                    default=args.use_gold_chunk)
                parser.add_argument('--use_gold_cc',
                                    action='store_true',
                                    default=args.use_gold_cc)
                parser.add_argument('--unuse_nongold_chunk',
                                    action='store_true',
                                    default=args.unuse_nongold_chunk)
                parser.add_argument('--unuse_nongold_cc',
                                    action='store_true',
                                    default=args.unuse_nongold_cc)
                parser.add_argument(
                    '--ignore_unknown_pretrained_chunk',
                    action='store_true',
                    default=args.ignore_unknown_pretrained_chunk)
                parser.add_argument('--ignore_unknown_pretrained_cc',
                                    action='store_true',
                                    default=args.ignore_unknown_pretrained_cc)

        return parser

    def add_input_data_format_option(self, parser, args):
        if args.task == constants.TASK_SEG:
            if args.input_data_format == 'sl':
                parser.add_argument('--input_data_format',
                                    '-f',
                                    default=args.input_data_format)
            else:
                if args.execute_mode == 'decode':
                    parser.add_argument('--input_data_format',
                                        '-f',
                                        default='sl')
                else:
                    print(
                        'Error: input data format for task={}/mode={} must be specified among from {sl}. Input: {}'
                        .format(args.task, args.execute_mode,
                                args.input_data_format),
                        file=sys.stderr)
                    sys.exit()
