import sys

import constants
from tools import conlleval
from tools.conlleval import FCounts


class Counts(object):
    def merge(self, counts):
        # to be implemented in sub-class
        pass


class ACounts(Counts):
    def __init__(self):
        self.total = 0
        self.correct = 0

    def merge(self, counts):
        if not isinstance(counts, ACounts):
            print('Invalid count object', file=sys.stderr)
            return

        self.total += counts.total
        self.correct += counts.correct


class DACounts(Counts):
    def __init__(self):
        self.l1 = ACounts()
        self.l2 = ACounts()

    def merge(self, counts):
        if not isinstance(counts, DACounts):
            print('Invalid count object', file=sys.stderr)
            return

        self.l1.merge(counts.l1)
        self.l2.merge(counts.l2)


class TACounts(Counts):
    def __init__(self):
        self.l1 = ACounts()
        self.l2 = ACounts()
        self.l3 = ACounts()

    def merge(self, counts):
        if not isinstance(counts, TACounts):
            print('Invalid count object', file=sys.stderr)
            return

        self.l1.merge(counts.l1)
        self.l2.merge(counts.l2)
        self.l3.merge(counts.l3)


class DFCounts(Counts):
    def __init__(self):
        self.l1 = FCounts()
        self.l2 = FCounts()

    def merge(self, counts):
        if not isinstance(counts, DFCounts):
            print('Invalid count object', file=sys.stderr)
            return

        self.l1.merge(counts.l1)
        self.l2.merge(counts.l2)


class FACounts(Counts):
    def __init__(self):
        self.l1 = FCounts()
        self.l2 = ACounts()
        self.l3 = ACounts()

    def merge(self, counts):
        if not isinstance(counts, FACounts):
            print('Invalid count object', file=sys.stderr)
            return

        self.l1.merge(counts.l1)
        self.l2.merge(counts.l2)
        self.l3.merge(counts.l3)


# Calculators


class AccuracyCalculator(object):
    def __init__(self, ignore_head=False, ignored_labels=set()):
        self.ignore_head = ignore_head
        self.ignored_labels = ignored_labels

    def __call__(self, ts, ys):
        counts = ACounts()
        for t, y in zip(ts, ys):
            if self.ignore_head:
                t = t[1:]
                y = y[1:]

            for ti, yi in zip(t, y):
                if int(ti) in self.ignored_labels:
                    continue

                counts.total += 1
                if ti == yi:
                    counts.correct += 1

        return counts


class DoubleAccuracyCalculator(object):
    def __init__(self, ignore_head=False, ignored_labels=set()):
        self.ignore_head = ignore_head
        self.ignored_labels = ignored_labels

    def __call__(self, t1s, t2s, y1s, y2s):
        counts = DACounts()
        for t1, t2, y1, y2 in zip(t1s, t2s, y1s, y2s):

            if self.ignore_head:
                t1 = t1[1:]
                t2 = t2[1:]
                y1 = y1[1:]
                y2 = y2[1:]

            for t1i, t2i, y1i, y2i in zip(t1, t2, y1, y2):
                if int(t2i) in self.ignored_labels:
                    continue

                counts.l1.total += 1
                counts.l2.total += 1
                if t1i == y1i:
                    counts.l1.correct += 1

                    if t2i == y2i:  # depend on t1i equals to y1i
                        counts.l2.correct += 1

        return counts


class TripleAccuracyCalculator(object):
    def __init__(self, ignore_head=False, ignored_labels=set()):
        self.ignore_head = ignore_head
        self.ignored_labels = ignored_labels

    def __call__(self, t1s, t2s, t3s, y1s, y2s, y3s):
        counts = TACounts()
        for t1, t2, t3, y1, y2, y3 in zip(t1s, t2s, t3s, y1s, y2s, y3s):

            if self.ignore_head:
                t1 = t1[1:]
                t2 = t2[1:]
                t3 = t3[1:]
                y1 = y1[1:]
                y2 = y2[1:]
                y3 = y3[1:]

            for t1i, t2i, t3i, y1i, y2i, y3i in zip(t1, t2, t3, y1, y2, y3):
                if int(t3i) in self.ignored_labels:
                    continue

                counts.l1.total += 1
                counts.l2.total += 1
                counts.l3.total += 1
                if t1i == y1i:
                    counts.l1.correct += 1

                if t2i == y2i:
                    counts.l2.correct += 1

                    if t3i == y3i:  # depend on t2i equals to y2i
                        counts.l3.correct += 1

        return counts


class FMeasureCalculator(object):
    def __init__(self, id2label):
        self.id2label = id2label

    def __call__(self, xs, ts, ys):
        counts = FCounts()
        for x, t, y in zip(xs, ts, ys):
            generator = self.generate_lines(x, t, y)
            counts = conlleval.evaluate(generator, counts=counts)

        return counts

    def generate_lines(self, x, t, y):
        i = 0
        while True:
            if i == len(x):
                break
            elif int(x[i]) == constants.PADDING_LABEL:
                break

            x_str = str(x[i])
            t_str = self.id2label[int(t[i])] if int(t[i]) > -1 else 'NONE'
            y_str = self.id2label[int(y[i])] if int(y[i]) > -1 else 'NONE'

            yield [x_str, t_str, y_str]
            i += 1


class DoubleFMeasureCalculator(object):
    def __init__(self, id2label):
        self.id2label = id2label

    def __call__(self, xs, ts, ys):
        counts = DFCounts()

        for x, t, y in zip(xs, ts, ys):
            generator_seg = self.generate_lines_seg(x, t, y)
            generator_tag = self.generate_lines(x, t, y)
            counts.l1 = conlleval.evaluate(generator_seg, counts=counts.l1)
            counts.l2 = conlleval.evaluate(generator_tag, counts=counts.l2)

        return counts

    def generate_lines_seg(self, x, t, y):
        i = 0
        while True:
            if i == len(x):
                break
            x_str = str(x[i])
            # if x_str == '＄':   # TMP: ignore special token for BCCWJ
            #     i += 1
            #     continue

            t_str = self.id2label[int(t[i])] if int(t[i]) > -1 else 'NONE'
            y_str = self.id2label[int(y[i])]
            tseg_str = t_str.split('-')[0]
            yseg_str = y_str.split('-')[0]

            yield [x_str, tseg_str, yseg_str]
            i += 1

    def generate_lines(self, x, t, y):
        i = 0
        while True:
            if i == len(x):
                break

            x_str = str(x[i])
            # if x_str == '＄':   # TMP: ignore special token for BCCWJ
            #     i += 1
            #     continue

            t_str = self.id2label[int(t[i])] if int(t[i]) > -1 else 'NONE'
            y_str = self.id2label[int(y[i])]

            yield [x_str, t_str, y_str]
            i += 1


class FMeasureAndAccuracyCalculator(object):
    def __init__(self, id2label):
        self.id2label = id2label

    def __call__(self, xs, t1s, t2s, y1s, y2s):
        counts = FACounts()
        if not t2s or not y2s:
            t2s = [None] * len(xs)
            y2s = [None] * len(xs)

        for x, t1, t2, y1, y2 in zip(xs, t1s, t2s, y1s, y2s):
            generator_seg = self.generate_lines_seg(x, t1, y1)
            counts.l1 = conlleval.evaluate(generator_seg, counts=counts.l1)

            if t2 is not None:
                for ti, yi in zip(t2, y2):
                    counts.l2.total += 1
                    if ti == yi:
                        counts.l2.correct += 1

        return counts

    def generate_lines_seg(self, x, t, y):
        i = 0
        while True:
            if i == len(x):
                break
            x_str = str(x[i])
            t_str = self.id2label[int(t[i])] if int(t[i]) > -1 else 'NONE'
            y_str = self.id2label[int(y[i])]
            tseg_str = t_str.split('-')[0]
            yseg_str = y_str.split('-')[0]

            yield [x_str, tseg_str, yseg_str]
            i += 1


class FMeasureAndAccuraciesCalculator(object):
    def __init__(self, id2label):
        self.id2label = id2label

    def __call__(self, xs, t1s, t2s, t3s, y1s, y2s, y3s):
        counts = FACounts()
        if not t2s or not y2s:
            t2s = [None] * len(xs)
            y2s = [None] * len(xs)
        if not t3s or not y3s:
            t3s = [None] * len(xs)
            y3s = [None] * len(xs)

        for x, t1, t2, t3, y1, y2, y3 in zip(xs, t1s, t2s, t3s, y1s, y2s, y3s):
            generator_seg = self.generate_lines_seg(x, t1, y1)
            counts.l1 = conlleval.evaluate(generator_seg, counts=counts.l1)

            if t2 is not None:
                for ti, yi in zip(t2, y2):
                    counts.l2.total += 1
                    if ti == yi:
                        counts.l2.correct += 1
            if t3 is not None:
                for ti, yi in zip(t3, y3):
                    counts.l3.total += 1
                    if ti == yi:
                        counts.l3.correct += 1

        return counts

    def generate_lines_seg(self, x, t, y):
        i = 0
        while True:
            if i == len(x):
                break
            x_str = str(x[i])
            t_str = self.id2label[int(t[i])] if int(t[i]) > -1 else 'NONE'
            y_str = self.id2label[int(y[i])]
            tseg_str = t_str.split('-')[0]
            yseg_str = y_str.split('-')[0]

            yield [x_str, tseg_str, yseg_str]
            i += 1


# Evaluators


class FMeasureEvaluator(object):
    def __init__(self, id2label):
        self.calculator = FMeasureCalculator(id2label)

    def calculate(self, *inputs):
        counts = self.calculator(*inputs)
        return counts

    def report_results(self, sen_counter, counts, loss, file=sys.stderr):
        ave_loss = loss / counts.token_counter
        met = conlleval.calculate_metrics(counts.correct_chunk,
                                          counts.found_guessed,
                                          counts.found_correct,
                                          counts.correct_tags,
                                          counts.token_counter)

        print('ave loss: %.5f' % ave_loss, file=file)
        print('sen, token, chunk, chunk_pred: {} {} {} {}'.format(
            sen_counter, counts.token_counter, counts.found_correct,
            counts.found_guessed),
              file=file)
        print('TP, FP, FN: %d %d %d' % (met.tp, met.fp, met.fn), file=file)
        print('A, P, R, F:%6.2f %6.2f %6.2f %6.2f' %
              (100. * met.acc, 100. * met.prec, 100. * met.rec,
               100. * met.fscore),
              file=file)

        res = '%.2f\t%.2f\t%.2f\t%.2f\t%.4f' % ((100. * met.acc),
                                                (100. * met.prec),
                                                (100. * met.rec),
                                                (100. * met.fscore), ave_loss)
        return res


class DoubleFMeasureEvaluator(object):
    def __init__(self, id2label):
        self.calculator = DoubleFMeasureCalculator(id2label)

    def calculate(self, *inputs):
        counts = self.calculator(*inputs)
        return counts

    def report_results(self, sen_counter, counts, loss, file=sys.stderr):
        ave_loss = loss / counts.l1.token_counter
        met1 = conlleval.calculate_metrics(counts.l1.correct_chunk,
                                           counts.l1.found_guessed,
                                           counts.l1.found_correct,
                                           counts.l1.correct_tags,
                                           counts.l1.token_counter)
        met2 = conlleval.calculate_metrics(counts.l2.correct_chunk,
                                           counts.l2.found_guessed,
                                           counts.l2.found_correct,
                                           counts.l2.correct_tags,
                                           counts.l2.token_counter)

        print('ave loss: %.5f' % ave_loss, file=file)
        print('sen, token, chunk, chunk_pred: {} {} {} {}'.format(
            sen_counter, counts.l1.token_counter, counts.l1.found_correct,
            counts.l1.found_guessed),
              file=file)
        print(
            'SEG - TP, FP, FN / A, P, R, F: %d %d %d /\t%6.2f %6.2f %6.2f %6.2f'
            % (met1.tp, met1.fp, met1.fn, 100. * met1.acc, 100. * met1.prec,
               100. * met1.rec, 100. * met1.fscore),
            file=file)
        print(
            'TAG - TP, FP, FN / A, P, R, F: %d %d %d /\t%6.2f %6.2f %6.2f %6.2f'
            % (met2.tp, met2.fp, met2.fn, 100. * met2.acc, 100. * met2.prec,
               100. * met2.rec, 100. * met2.fscore),
            file=file)

        res = '%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f' % (
            (100. * met1.acc), (100. * met1.prec), (100. * met1.rec),
            (100. * met1.fscore), (100. * met2.acc), (100. * met2.prec),
            (100. * met2.rec), (100. * met2.fscore), ave_loss)

        return res


class AccuracyEvaluator(object):
    def __init__(self, ignore_head=False, ignored_labels=set()):
        self.calculator = AccuracyCalculator(ignore_head, ignored_labels)

    def calculate(self, *inputs):
        ts = inputs[1]
        ys = inputs[2]
        counts = self.calculator(ts, ys)
        return counts

    def report_results(self, sen_counter, counts, loss=None, file=sys.stderr):
        ave_loss = loss / counts.total if loss is not None else None
        acc = 1. * counts.correct / counts.total

        if ave_loss is not None:
            print('ave loss: %.5f' % ave_loss, file=file)
        print('sen, token, correct: {} {} {}'.format(sen_counter, counts.total,
                                                     counts.correct),
              file=file)
        print('A:%6.2f' % (100. * acc), file=file)

        if ave_loss is not None:
            res = '%.2f\t%.4f' % ((100. * acc), ave_loss)
        else:
            res = '%.2f' % (100. * acc)

        return res


class DoubleAccuracyEvaluator(object):
    def __init__(self, ignore_head=False, ignored_labels=set()):
        self.calculator = DoubleAccuracyCalculator(ignore_head, ignored_labels)

    def calculate(self, *inputs):
        t1s = inputs[1]
        t2s = inputs[2]
        y1s = inputs[3]
        y2s = inputs[4]
        counts = self.calculator(t1s, t2s, y1s, y2s)
        return counts


class TripleAccuracyEvaluator(object):
    def __init__(self, ignore_head=False, ignored_labels=set()):
        self.calculator = TripleAccuracyCalculator(ignore_head, ignored_labels)

    def calculate(self, *inputs):
        t1s = inputs[1]
        t2s = inputs[2]
        t3s = inputs[3]
        y1s = inputs[4]
        y2s = inputs[5]
        y3s = inputs[6]
        counts = self.calculator(t1s, t2s, t3s, y1s, y2s, y3s)
        return counts
