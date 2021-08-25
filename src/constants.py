global __version
__version__ = 'v0.1.1'

# common

GPU_DEVICE = 'GPU'
CUDA_VISIBLE_DEVICE = 'CUDA_VISIBLE_DEVICES'
CUDA_DEVICE = 'cuda'
CPU_DEVICE = 'cpu'

# task

TASK_SEG = 'seg'
TASK_SEGTAG = 'segtag'
TASK_TAG = 'tag'

# tagging unit

SINGLE_TAGGING = 'single'
HYBRID_TAGGING = 'hybrid'
SUB_COMBINATIVE_TAGGING = 'sub-combinative'
COMBINATIVE_TAGGING = 'combinative'

# for analyzer

LOG_DIR = 'log'
MODEL_DIR = 'models/main'

# model

ATTENTION_MAX_LENGTH = 300

# for dictionary

UNK_SYMBOL = '<UNK>'
NUM_SYMBOL = '<NUM>'
NONE_SYMBOL = '<NONE>'
ROOT_SYMBOL = '<ROOT>'

CHAR = 'char'
WORD = 'word'
UNIGRAM = 'unigram'
BIGRAM = 'bigram'
CHUNK = 'chunk'
SUBWORD = 'subword'
CC = 'cc'
SEG_LABEL = 'seg_label'
ARC_LABEL = 'arc_label'
ATTR_LABEL = 'attr{}_label'.format
DOMAIN = 'domain'

TYPE_THAI = '<TH>'
TYPE_ENG = '<EN>'
TYPE_HIRA = '<HR>'
TYPE_KATA = '<KT>'
TYPE_LONG = '<LG>'
TYPE_KANJI = '<KJ>'
TYPE_ALPHA = '<AL>'
TYPE_DIGIT = '<DG>'
TYPE_SPACE = '<SC>'
TYPE_SYMBOL = '<SY>'
TYPE_ASCII_OTHER = '<AO>'

BOS = '<B>'  # unused
EOS = '<E>'

# for character

SEG_LABELS = 'BIES'
B = 'B'
I = 'I'
E = 'E'
S = 'S'
O = 'O'

### for hybrid/mutant/combinative segmentation

CON = 'CON'
WCON = 'WCON'
CCCON = 'CCCON'
SWCON = 'SWCON'
AVG = 'AVG'
WAVG = 'WAVG'
CCAVG = 'CCAVG'
SWAVG = 'SWAVG'

# for data i/o

PADDING_LABEL = -1
NUM_FOR_REPORTING = 100000

SL_COLUMN_DELIM = '\t'
SL_TOKEN_DELIM = ' '
SL_ATTR_DELIM = '_'
WL_TOKEN_DELIM = '\n'
WL_ATTR_DELIM = '\t'
KEY_VALUE_DELIM = '='
SUBATTR_SEPARATOR = '-'
COMMENT_SYM = '#'
ATTR_INFO_DELIM = ','
ATTR_INFO_DELIM2 = ':'
ATTR_INFO_DELIM3 = '_'

JSON_FORMAT = 'json'
SL_FORMAT = 'sl'
TXT_FORMAT = 'txt'
WL_FORMAT = 'wl'

# dataset

MAX_VOCAB_SIZE = 128000

# token

INIT_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

### for feature extraction
# example of expected input: 'seg:L:2-3,4-5,6-10'

FEAT_TEMP_DELIM = ':'
FEAT_TEMP_RANGE_DELIM = ','
FEAT_TEMP_RANGE_SYM = '-'
L = 'L'
R = 'R'

### for character-cluster
# c: consonant
# t: tone

CONSONANT_SYMBOL = 'c'
PIPE_SYMBOL = '|'
TONE_SYMBOL = 't'
CONSONANTS = '[ก-ฮ]'
TONES = '[่-๋]?'
CHAR_CLUSTERS = '''
    เc็c
    เcctาะ
    เccีtยะ
    เccีtย(?=[เ-ไก-ฮ]|$)
    เccอะ
    เcc็c
    เcิc์c
    เcิtc
    เcีtยะ?
    เcืtอะ?
    เc[ิีุู]tย(?=[เ-ไก-ฮ]|$)
    เctา?ะ?
    cัtวะ
    c[ัื]tc[ุิะ]?
    c[ิุู]์
    c[ะ-ู]t
    c็
    ct[ะาำ]?
    แc็c
    แcc์
    แctะ
    แcc็c
    แccc์
    โctะ
    [เ-ไ]ct
    ๆ
    ฯลฯ
    ฯ
'''
