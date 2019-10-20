import os
from datautil.dependency import read_deps
from collections import Counter
import numpy as np


def create_vocab(data_path, min_count=2):
    assert os.path.exists(data_path)

    root_rel = ''
    wd_counter, pos_counter, rel_counter = Counter(), Counter(), Counter()
    with open(data_path, 'r', encoding='utf-8') as fr:
        for deps in read_deps(fr):
            for dep in deps:
                wd_counter[dep.form] += 1
                pos_counter[dep.pos] += 1

                if dep.head != 0:
                    rel_counter[dep.dep_rel] += 1
                elif root_rel == '':
                    root_rel = dep.dep_rel
                    rel_counter[dep.dep_rel] += 1
                elif root_rel != dep.dep_rel:
                    print('root = ' + root_rel + ', rel for root = ' + dep.dep_rel)

    return DepVocab(wd_counter, pos_counter, rel_counter, root_rel, min_count=min_count)


class DepVocab(object):
    '''
        词表
        词性表
        依存关系表
    '''
    def __init__(self, wd_counter: Counter,
                 pos_counter: Counter,
                 rel_counter: Counter,
                 root_rel='root',
                 padding='<pad>',
                 unknown='<unk>',
                 min_count=2):
        
        self.root_rel = root_rel
        self.root_form = '<'+root_rel.lower()+'>'
        self.padding = padding
        self.unknown = unknown

        self._word2idx = dict()
        if self.padding is not None:
            self._word2idx[self.padding] = len(self._word2idx)
        if self.root_form is not None:
            self._word2idx[self.root_form] = len(self._word2idx)
        if self.unknown is not None:
            self._word2idx[self.unknown] = len(self._word2idx)

        self._pos2idx = dict()
        if self.padding is not None:
            self._pos2idx[self.padding] = len(self._pos2idx)
        if self.root_rel is not None:
            self._pos2idx[self.root_rel] = len(self._pos2idx)
        if self.unknown is not None:
            self._pos2idx[self.unknown] = len(self._pos2idx)

        self._rel2idx = dict()
        if self.padding is not None:
            self._rel2idx[self.padding] = len(self._rel2idx)
        if self.root_rel is not None:
            self._rel2idx[self.root_rel] = len(self._rel2idx)

        for wd, freq in wd_counter.items():
            if freq > min_count and wd not in self._word2idx:
                self._word2idx[wd] = len(self._word2idx)

        for pos in pos_counter.keys():
            if pos not in self._pos2idx:
                self._pos2idx[pos] = len(self._pos2idx)

        for rel in rel_counter.keys():
            if rel not in self._rel2idx:
                self._rel2idx[rel] = len(self._rel2idx)

        self._idx2wd = dict((idx, wd) for wd, idx in self._word2idx.items())
        self._idx2pos = dict((idx, pos) for pos, idx in self._pos2idx.items())
        self._idx2rel = dict((idx, rel) for rel, idx in self._rel2idx.items())

        self._extwd2idx = None
        self._extidx2wd = None

    def get_embedding_weights(self, embed_path):
        if not os.path.exists(embed_path):
            print('embedding path does not exist!')
            return None

        vec_tab = dict()
        vec_size = 0
        with open(embed_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                tokens = line.strip().split()
                wd, vec = tokens[0], tokens[1:]
                if vec_size == 0:
                    vec_size = len(vec)
                vec_tab[wd] = np.asarray(vec, dtype=np.float32)

        oov_ratio = 0
        for wd in self._word2idx.keys():
            if wd not in vec_tab:
                oov_ratio += 1
        print('oov ratio: %.2f%%' % (100 * (oov_ratio-3) / (len(self._word2idx)-3)))

        if self._extwd2idx is None:
            self._extwd2idx = dict()
            if self.padding is not None:
                self._extwd2idx[self.padding] = len(self._extwd2idx)
            if self.root_form is not None:
                self._extwd2idx[self.root_form] = len(self._extwd2idx)
            if self.unknown is not None:
                self._extwd2idx[self.unknown] = len(self._extwd2idx)

        for wd in vec_tab.keys():
            if wd not in self._extwd2idx:
                self._extwd2idx[wd] = len(self._extwd2idx)

        self._extidx2wd = dict((idx, wd) for wd, idx in self._extwd2idx.items())

        embed_weights = np.zeros((len(self._extwd2idx), vec_size), dtype=np.float32)
        for idx, wd in self._extidx2wd.items():
            if wd in vec_tab:
                embed_weights[idx] = vec_tab[wd]

        embed_weights[self._extwd2idx[self.unknown]] = np.mean(embed_weights, 0) / np.std(embed_weights)

        return embed_weights

    def word2index(self, wds):
        if isinstance(wds, list):
            return [self._word2idx.get(wd, self._word2idx[self.unknown]) for wd in wds]
        else:
            return self._word2idx.get(wds, self._word2idx[self.unknown])

    def index2word(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2wd.get(i, self.unknown) for i in idxs]
        else:
            return self._idx2wd.get(idxs, self.unknown)

    def extwd2index(self, wds):
        if isinstance(wds, list):
            return [self._extwd2idx.get(wd, self._extwd2idx[self.unknown]) for wd in wds]
        else:
            return self._extwd2idx.get(wds, self._extwd2idx[self.unknown])

    def extidx2word(self, idxs):
        if isinstance(idxs, list):
            return [self._extidx2wd.get(i, self.unknown) for i in idxs]
        else:
            return self._extidx2wd.get(idxs, self.unknown)

    def pos2index(self, pos):
        if isinstance(pos, list):
            return [self._pos2idx.get(p, self._pos2idx[self.unknown]) for p in pos]
        else:
            return self._pos2idx.get(pos, self._pos2idx[self.unknown])

    def index2pos(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2pos.get(i, self.unknown) for i in idxs]
        else:
            return self._idx2pos.get(idxs, self.unknown)

    def rel2index(self, rels):
        if isinstance(rels, list):
            return [self._rel2idx.get(rel) for rel in rels]
        else:
            return self._rel2idx.get(rels)

    def index2rel(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2rel.get(i) for i in idxs]
        else:
            return self._idx2rel.get(idxs)

    @property
    def vocab_size(self):
        return len(self._word2idx)
    
    @property
    def extvocab_size(self):
        return len(self._extwd2idx)

    @property
    def pos_size(self):
        return len(self._pos2idx)

    @property
    def rel_size(self):
        return len(self._rel2idx)

