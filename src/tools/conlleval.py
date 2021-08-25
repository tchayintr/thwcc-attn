# This file is the modified version of the following software.
#
# - conlleval.py <https://github.com/spyysalo/conlleval.py>
#     - Copyright (c) 2016, Sampo Pyysalo
#     - Released under the MIT license https://opensource.org/licenses/mit-license.php


# Python version of the evaluation script from CoNLL'00-

# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported

import sys
import re

from collections import defaultdict, namedtuple

ANY_SPACE = '<SPACE>'



class FormatError(Exception):
    pass



Metrics = namedtuple('Metrics', 'tp fp fn acc prec rec fscore')



class FCounts(object):
    def __init__(self):
        self.correct_chunk = 0    # number of correctly identified chunks
        self.correct_tags = 0     # number of correct chunk tags
        self.found_correct = 0    # number of chunks in corpus
        self.found_guessed = 0    # number of identified chunks
        self.token_counter = 0    # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)

    def __str__(self):
        return '[%s, %s, %s, %s, %s]' % (str(self.correct_chunk),
                                         str(self.correct_tags),
                                         str(self.found_correct),
                                         str(self.found_guessed),
                                         str(self.token_counter))


    def merge(self, counts):
        if not isinstance(counts, FCounts):
            print('Invalid count object', file=sys.stderr)
            return

        self.correct_chunk += counts.correct_chunk
        self.correct_tags += counts.correct_tags
        self.found_correct += counts.found_correct
        self.found_guessed += counts.found_guessed
        self.token_counter += counts.token_counter



def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(
        description='evaluate tagging results using CoNLL criteria',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg = parser.add_argument
    arg('-b', '--boundary', metavar='STR', default='-X-',
        help='sentence boundary')
    arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
        help='character delimiting items in input')
    arg('-o', '--otag', metavar='CHAR', default='O',
        help='alternative outside tag')
    arg('file', nargs='?', default=None)
    return parser.parse_args(argv)


def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')


def evaluate(iterable, counts=None, options=None):
    if options is None:
        options = parse_args([])    # use defaults

    if counts is None:
        counts = FCounts()

    num_features = None       # number of features per line
    in_correct = False        # currently processed chunks is correct until now
    last_correct = 'O'        # previous chunk tag in corpus
    last_correct_type = ''    # type of previously identified chunk tag
    last_guessed = 'O'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus

    for line in iterable:

        if type(line) is str:
            line = line.rstrip('\r\n')
            if options.delimiter == ANY_SPACE:
                features = line.split()
            else:
                features = line.split(options.delimiter)
        elif type(line) is list:
            features = line
        else:
            raise FormatError('unexpected format of lines: ' + str(line))

        if num_features is None:
            num_features = len(features)
        elif num_features != len(features) and len(features) != 0:
            continue
            raise FormatError('unexpected number of features: %d (%d)' %
                              (len(features), num_features))

        if len(features) == 0 or features[0] == options.boundary:
            features = [options.boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)

        guessed, guessed_type = parse_tag(features.pop())
        correct, correct_type = parse_tag(features.pop())
        first_item = features.pop(0)

        if first_item == options.boundary:
            guessed = 'O'

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)

        if in_correct:
            if (end_correct and end_guessed and last_guessed_type == last_correct_type):
                in_correct = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
            elif (end_correct != end_guessed or guessed_type != correct_type):
                in_correct = False

        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct = True

        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        if first_item != options.boundary:
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            counts.token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts


def uniq(iterable):
    seen = set()
    return [i for i in iterable if not (i in seen or seen.add(i))]


def calculate_metrics(correct, guessed, total, correct_tokens=0, total_tokens=0):
    tp, fp, fn = correct, guessed - correct, total - correct
    a = (1. * correct_tokens / total_tokens) if total_tokens > 0 else 0
    p = 0 if tp + fp == 0 else 1. * tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1. * tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, a, p, r, f)


def metrics(counts):
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct.keys()) + list(c.t_found_guessed.keys())):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )
    return overall, by_type


def report(counts, out=None):
    if out is None:
        out = sys.stdout

    overall, by_type = metrics(counts)

    c = counts
    out.write('processed %d tokens with %d phrases; ' %
              (c.token_counter, c.found_correct))
    out.write('found: %d phrases; correct: %d.\n' %
              (c.found_guessed, c.correct_chunk))

    if c.token_counter > 0:
        out.write('accuracy: %6.2f%%; ' %
                  (100. * c.correct_tags / c.token_counter))
        out.write('precision: %6.2f%%; ' % (100. * overall.prec))
        out.write('recall: %6.2f%%; ' % (100. * overall.rec))
        out.write('FB1: %6.2f\n' % (100. * overall.fscore))

    for i, m in sorted(by_type.items()):
        out.write('%17s: ' % i)
        out.write('precision: %6.2f%%; ' % (100. * m.prec))
        out.write('recall: %6.2f%%; ' % (100. * m.rec))
        out.write('FB1: %6.2f  %d\n' % (100. * m.fscore, c.t_found_guessed[i]))


def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start


def main(argv):
    args = parse_args(argv[1:])

    if args.file is None:
        counts = evaluate(sys.stdin, None, args)
    else:
        with open(args.file) as f:
            counts = evaluate(f, None, args)
    report(counts)




if __name__ == '__main__':
    sys.exit(main(sys.argv))
