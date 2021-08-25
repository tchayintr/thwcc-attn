import re

import constants


class CC(object):
    def __init__(self, clusters):
        self.ccs = clusters.replace(constants.CONSONANT_SYMBOL,
                                    constants.CONSONANTS).replace(
                                        constants.TONE_SYMBOL,
                                        constants.TONES).split()


class CCExtractor(CC):
    def __init__(self, clusters):
        super().__init__(clusters)
        self.ccp = re.compile(constants.PIPE_SYMBOL.join(self.ccs))

    def gen_char_clusters(self, seq):
        p = 0
        while p < len(seq):
            m = self.ccp.match(seq[p:])
            if m:
                n = m.span()[1]
            else:
                n = 1
            yield seq[p:p + n]
            p += n

    def create_all_char_clusters(self, seq):
        res = self.gen_char_clusters(seq)
        return list(res)

    def create_all_char_cluster_indexes(self, seq, max_length=None):
        p = 0
        index_pairs = []
        for cc in self.gen_char_clusters(seq):
            n = len(cc) + p
            index_pairs.append(
                (p,
                 n)) if not max_length or (max_length
                                           and len(cc) <= max_length) else None
            p = n
        return index_pairs


if __name__ == '__main__':
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

    sent = 'ฉันกินข้าวที่บ้าน'
    max_len = 4
    extractor = CCExtractor(CHAR_CLUSTERS)
    ccs = extractor.create_all_char_clusters(sent)
    print('sentence: {}'.format(ccs))
