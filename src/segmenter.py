import sys

from core import Core
from arguments import tagger_arguments
from trainers import (tagger_trainer, hybrid_unit_segmenter_trainer,
                      mutant_unit_segmenter_trainer,
                      sub_combinative_unit_segmenter_trainer,
                      combinative_unit_segmenter_trainer,
                      transformer_tagger_trainer)


class Segmenter(Core):
    def __init__(self):
        super().__init__()

    def get_args(self):
        parser = tagger_arguments.TaggerArgumentLoader()
        args = parser.parse_args()
        return args

    def get_trainer(self, args):
        if args.tagging_unit == 'single':
            trainer = tagger_trainer.TaggerTrainer(args)
        elif args.tagging_unit == 'hybrid':
            trainer = hybrid_unit_segmenter_trainer.HybridUnitSegmenterTrainer(
                args)
        elif args.tagging_unit == 'mutant':
            trainer = mutant_unit_segmenter_trainer.MutantUnitSegmenterTrainer(
                args)
        elif args.tagging_unit == 'sub-combinative':
            trainer = sub_combinative_unit_segmenter_trainer.SubCombinativeUnitSegmenterTrainer(
                args)
        elif args.tagging_unit == 'combinative':
            trainer = combinative_unit_segmenter_trainer.CombinativeUnitSegmenterTrainer(
                args)
        elif args.tagging_unit == 'transformer':
            trainer = transformer_tagger_trainer.TransformerTaggerTrainer(args)
        else:
            print('Error: the following argument is invalid for {} mode: {}'.
                  format(args.execute_mode, '--tagging_unit'))
            sys.exit()

        return trainer


if __name__ == '__main__':
    analyzer = Segmenter()
    analyzer.run()
