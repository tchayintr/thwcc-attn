import sys

from classifiers.sequence_tagger import SequenceTagger
import constants
import models.util


class SubCombinativeSequenceTagger(SequenceTagger):
    def __init__(self, predictor, task=constants.TASK_SEG):
        super().__init__(predictor=predictor)
        self.task = task

    def forward(self, *inputs, train=False):
        ret = self.predictor(*inputs[:11])  # exclude ncands and nswcands
        return ret

    def change_dropout_ratio(self, dropout_ratio, file=sys.stderr):
        self.change_embed_dropout_ratio(dropout_ratio)
        self.change_rnn_dropout_ratio(dropout_ratio)
        self.change_biaffine_dropout_ratio(dropout_ratio)
        self.change_mlp_dropout_ratio(dropout_ratio)
        self.change_chunk_vector_dropout_ratio(dropout_ratio)
        self.change_subword_vector_dropout_ratio(dropout_ratio)
        print('', file=sys.stderr)

    def change_biaffine_dropout_ratio(self, dropout_ratio):
        self.predictor.biaffine_dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format(
            'Biaffine', self.predictor.biaffine_dropout),
              file=sys.stderr)

    def change_chunk_vector_dropout_ratio(self, dropout_ratio):
        self.predictor.chunk_vector_dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format(
            'Chunk vector', self.predictor.chunk_vector_dropout),
              file=sys.stderr)

    def change_subword_vector_dropout_ratio(self, dropout_ratio):
        self.predictor.subword_vector_dropout = dropout_ratio
        print('Set {} dropout ratio to {}'.format(
            'Subword vector', self.predictor.subword_vector_dropout),
              file=sys.stderr)

    def load_pretrained_embedding_layer(self,
                                        dic,
                                        external_unigram_model,
                                        external_bigram_model,
                                        external_chunk_model,
                                        external_subword_model,
                                        finetuning=False):
        usage = self.predictor.pretrained_embed_usage

        if external_unigram_model:
            id2unigram = dic.tables[constants.UNIGRAM].id2str
            if usage == models.util.ModelUsage.INIT:
                models.util.load_pretrained_embedding_layer(
                    id2unigram,
                    self.predictor.unigram_embed,
                    external_unigram_model,
                    finetuning=finetuning)
            elif usage == models.util.ModelUsage.ADD or models.util.ModelUsage.CONCAT:
                models.util.load_pretrained_embedding_layer(
                    id2unigram,
                    self.predictor.pretrained_unigram_embed,
                    external_unigram_model,
                    finetuning=finetuning)

        if external_bigram_model:
            id2bigram = dic.tables[constants.BIGRAM].id2str
            if usage == models.util.ModelUsage.INIT:
                models.util.load_pretrained_embedding_layer(
                    id2bigram,
                    self.predictor.bigram_embed,
                    external_bigram_model,
                    finetuning=finetuning)
            elif usage == models.util.ModelUsage.ADD or models.util.ModelUsage.CONCAT:
                models.util.load_pretrained_embedding_layer(
                    id2bigram,
                    self.predictor.pretrained_bigram_embed,
                    external_bigram_model,
                    finetuning=finetuning)

        if external_chunk_model:
            id2chunk = dic.tries[constants.CHUNK].id2chunk
            if usage == models.util.ModelUsage.INIT:
                models.util.load_pretrained_embedding_layer(
                    id2chunk,
                    self.predictor.chunk_embed,
                    external_chunk_model,
                    finetuning=finetuning)
            elif usage == models.util.ModelUsage.ADD or models.util.ModelUsage.CONCAT:
                models.util.load_pretrained_embedding_layer(
                    id2chunk,
                    self.predictor.pretrained_chunk_embed,
                    external_chunk_model,
                    finetuning=finetuning)

        if external_subword_model:
            id2subword = dic.tries[constants.SUBWORD].id2subword
            if usage == models.util.ModelUsage.INIT:
                models.util.load_pretrained_embedding_layer(
                    id2subword,
                    self.predictor.subword_embed,
                    external_subword_model,
                    finetuning=finetuning)
            elif usage == models.util.ModelUsage.ADD or models.util.ModelUsage.CONCAT:
                models.util.load_pretrained_embedding_layer(
                    id2subword,
                    self.predictor.pretrained_subword_embed,
                    external_subword_model,
                    finetuning=finetuning)

    def grow_embedding_layers(self,
                              dic_grown,
                              external_unigram_model=None,
                              external_bigram_model=None,
                              external_chunk_model=None,
                              external_subword_model=None,
                              train=True,
                              fasttext=False):
        if (self.predictor.pretrained_embed_usage == models.util.ModelUsage.ADD
                or self.predictor.pretrained_embed_usage
                == models.util.ModelUsage.CONCAT):
            pretrained_unigram_embed = self.predictor.pretrained_unigram_embed
            pretrained_bigram_embed = self.predictor.pretrained_bigram_embed
            pretrained_chunk_embed = self.predictor.pretrained_chunk_embed
            pretrained_subword_embed = self.predictor.pretrained_subword_embed
        else:
            pretrained_unigram_embed = None
            pretrained_bigram_embed = None
            pretrained_chunk_embed = None
            pretrained_subword_embed = None

        id2unigram_grown = dic_grown.tables[constants.UNIGRAM].id2str
        n_unigrams_org = self.predictor.unigram_embed.weight.shape[0]
        n_unigrams_grown = len(id2unigram_grown)
        models.util.grow_embedding_layers(
            n_unigrams_org,
            n_unigrams_grown,
            self.predictor.unigram_embed,
            pretrained_unigram_embed,
            external_unigram_model,
            id2unigram_grown,
            self.predictor.pretrained_embed_usage,
            train=train)

        if external_bigram_model:
            id2bigram_grown = dic_grown.tables[constants.BIGRAM].id2str
            n_bigrams_org = self.predictor.bigram_embed.weight.shape[0]
            n_bigrams_grown = len(id2bigram_grown)
            models.util.grow_embedding_layers(
                n_bigrams_org,
                n_bigrams_grown,
                self.predictor.bigram_embed,
                pretrained_bigram_embed,
                external_bigram_model,
                id2bigram_grown,
                self.predictor.pretrained_embed_usage,
                train=train)

        id2chunk_grown = dic_grown.tries[constants.CHUNK].id2chunk
        n_chunks_org = self.predictor.chunk_embed.weight.shape[0]
        n_chunks_grown = len(id2chunk_grown)
        models.util.grow_embedding_layers(
            n_chunks_org,
            n_chunks_grown,
            self.predictor.chunk_embed,
            pretrained_chunk_embed,
            external_chunk_model,
            id2chunk_grown,
            self.predictor.pretrained_embed_usage,
            train=train,
            fasttext=fasttext)

        id2subword_grown = dic_grown.tries[constants.SUBWORD].id2subword
        n_sws_org = self.predictor.subword_embed.weight.shape[0]
        n_sws_grown = len(id2subword_grown)
        models.util.grow_embedding_layers(
            n_sws_org,
            n_sws_grown,
            self.predictor.subword_embed,
            pretrained_subword_embed,
            external_subword_model,
            id2subword_grown,
            self.predictor.pretrained_embed_usage,
            train=train,
            fasttext=fasttext)

    def copy_parameters_from_basemodel(self, base):
        self.predictor.unigram_embed = base.unigram_embed
        self.predictor.rnn = base.rnn
        if self.predictor.use_crf:
            self.predictor.crf = base.crf
