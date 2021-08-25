from copy import copy
import sys

from evaluators.common import Counts, FMeasureAndAccuracyCalculator, DoubleFMeasureEvaluator
from tools import conlleval


class HybridSegmenterEvaluator(object):
    def __init__(self, id2label):
        self.calculator = FMeasureAndAccuracyCalculator(id2label)

    def calculate(self, *inputs):
        counts = self.calculator(*inputs)
        return counts

    def report_results(self, sen_counter, counts, loss, file=sys.stderr):
        ave_loss = loss / counts.l1.token_counter
        l1_met = conlleval.calculate_metrics(counts.l1.correct_chunk,
                                             counts.l1.found_guessed,
                                             counts.l1.found_correct,
                                             counts.l1.correct_tags,
                                             counts.l1.token_counter)
        l2_acc = 1. * counts.l2.correct / counts.l2.total if counts.l2.total > 0 else 0

        print('ave loss: %.5f' % ave_loss, file=file)
        print('sen, token, chunk, chunk_pred: {} {} {} {}'.format(
            sen_counter, counts.l1.token_counter, counts.l1.found_correct,
            counts.l1.found_guessed),
              file=file)
        print('TP, FP, FN: %d %d %d' % (l1_met.tp, l1_met.fp, l1_met.fn),
              file=file)
        print('A, P, R, F, AW:%6.2f %6.2f %6.2f %6.2f %6.2f' %
              (100. * l1_met.acc, 100. * l1_met.prec, 100. * l1_met.rec,
               100. * l1_met.fscore, (100. * l2_acc)),
              file=file)

        res = '%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f' % (
            (100. * l1_met.acc), (100. * l1_met.prec), (100. * l1_met.rec),
            (100. * l1_met.fscore), (100. * l2_acc), ave_loss)

        return res


class HybridTaggerEvaluator(DoubleFMeasureEvaluator):  # tmp
    def __init__(self, id2label):
        super().__init__(id2label)

    def calculate(self, *inputs):
        xs = inputs[0]
        ts = inputs[1]
        ys = inputs[3]
        counts = self.calculator(xs, ts, ys)
        return counts


# attention-based evaluation


class ACountsA(object):
    def __init__(self):
        self.all = 0
        self.possible = 0  # correct word exists in candidates
        self.wcorrect = 0  # for word prediction
        self.scorrect = 0  # for segmentation label

    def __str__(self):
        return '%d %d %d %d' % (self.all, self.possible, self.wcorrect,
                                self.scorrect)

    def merge(self, counts):
        if not isinstance(counts, ACountsA):
            print('Invalid count object', file=sys.stderr)
            return

        self.all += counts.all
        self.possible += counts.possible
        self.wcorrect += counts.wcorrect
        self.scorrect += counts.scorrect

    def increment(self, possible=True, wcorrect=True, scorrect=True):
        self.all += 1
        if possible:
            self.possible += 1
        if wcorrect:
            self.wcorrect += 1
        if scorrect:
            self.scorrect += 1


class ACountsAX(object):
    def __init__(self):
        self.key2counts = {
            'total': ACountsA(),
            'atn_cor': ACountsA(),
            'atn_inc': ACountsA(),
            'atn_inc_unk': ACountsA()
        }

    def merge(self, counts):
        if not isinstance(counts, ACountsAX):
            print('Invalid count object', file=sys.stderr)
            return

        for key in self.key2counts.keys() | counts.key2counts.keys():
            if key in self.key2counts and key in counts.key2counts:
                self.key2counts[key].merge(counts.key2counts[key])

            elif key in self.key2counts:
                pass

            else:
                self.key2counts[key] = copy(counts.key2counts[key])

    def increment(self, key, possible=True, wcorrect=True, scorrect=True):
        self.key2counts[key].increment(possible, wcorrect, scorrect)


class AccuracyCalculatorForAttention(object):
    def __init__(self):
        pass

    def __call__(self, xs, gls, gcs, ncands, pls, pcs):
        countsx = ACountsAX()

        for gc, pc, ncand, gl, pl in zip(gcs, pcs, ncands, gls, pls):
            for gci, pci, nci, gli, pli in zip(gc, pc, ncand, gl, pl):
                if nci not in countsx.key2counts:
                    countsx.key2counts[nci] = ACountsA()

                countsx.increment(nci,
                                  possible=gci >= 0,
                                  wcorrect=gci == pci,
                                  scorrect=gli == pli)

                countsx.increment('total',
                                  possible=gci >= 0,
                                  wcorrect=gci == pci,
                                  scorrect=gli == pli)

                if pci is not None:
                    if pci == gci:
                        countsx.increment('atn_cor',
                                          possible=True,
                                          wcorrect=True,
                                          scorrect=gli == pli)

                    else:
                        countsx.increment('atn_inc',
                                          possible=gci >= 0,
                                          wcorrect=False,
                                          scorrect=gli == pli)

                        if gci < 0:
                            countsx.increment('atn_inc_unk',
                                              possible=False,
                                              wcorrect=False,
                                              scorrect=gli == pli)

        return countsx


class AccuracyEvaluatorForAttention(object):
    def __init__(self):
        self.calculator = AccuracyCalculatorForAttention()

    def calculate(self, *inputs):
        countsx = self.calculator(*inputs)
        return countsx

    def report_results(self, sen_counter, countsx, loss, file=sys.stderr):
        res = None
        ave_loss = loss / countsx.key2counts['total'].all
        print('ave loss: %.5f' % ave_loss, file=file)
        print('ncand\tall\tw_pos\tw_cor\ts_cor\tw_upper\tw_acc\ts_acc',
              file=file)

        for key, counts in countsx.key2counts.items():
            wacc = counts.wcorrect / counts.all * 100
            wupper = counts.possible / counts.all * 100
            sacc = counts.scorrect / counts.all * 100
            print('%s\t%d\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f' %
                  (str(key), counts.all, counts.possible, counts.wcorrect,
                   counts.scorrect, wupper, wacc, sacc),
                  file=file)
            if key == 'total':
                res = '%.2f\t%.2f\t%.2f\t%.4f' % (wacc, wupper, sacc, ave_loss)

        if 1 in countsx.key2counts:
            c1 = countsx.key2counts[1]
            ct = countsx.key2counts['total']
            print('lower(1): %.2f' % (c1.possible / c1.all * 100), file=file)
            print('lower(total): %.2f' % (c1.possible / ct.all * 100),
                  file=file)

        return res
