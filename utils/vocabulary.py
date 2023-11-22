import os
from collections import Counter
import numpy as np
from .file_io import save_pickle, load_pickle, load_json


class Vocabulary:
    def __init__(self, word2idx, unk_token="[UNK]"):
        self.word2idx = word2idx
        self.idx2word = self.build_reverse_vocab(word2idx)
        self.unk_token = unk_token

    def build_reverse_vocab(self, word2idx):
        return {i: w for w, i in word2idx.items()}

    def has_word(self, word):
        '''
        检查词是否被记录
        :param word:
        :return:
        '''
        return word in self.word2idx

    def to_index(self, word):
        '''
        将词转为数字. 若词不再词典中被记录, 将视为 unknown, 若 ``unknown=None`` , 将抛出
        :param word:
        :return:
        '''
        if word in self.word2idx:
            return self.word2idx[word]
        if self.unk_token is not None:
            return self.word2idx[self.unk_token]
        else:
            raise ValueError("word {} not in vocabulary".format(word))

    def to_word(self, idx):
        """
        给定一个数字, 将其转为对应的词.

        :param int idx: the index
        :return str word: the word
        """
        return self.idx2word[idx]

    def __len__(self):
        return len(self.idx2word)
    
    def save(self, file_path):
        '''
        保存vocab
        :param file_name:
        :param pickle_path:
        :return:
        '''
        mappings = {
            "word2idx": self.word2idx,
            'unk_token': self.unk_token
        }
        save_pickle(data=mappings, file_path=file_path)

    @classmethod
    def load_from_file(cls, file_path):
        '''
        从文件组红加载vocab
        :param file_name:
        :param pickle_path:
        :return:
        '''
        mappings = load_pickle(input_file=file_path)
        word2idx = mappings['word2idx']
        unk_token = mappings.get('unk_token',"[UNK]")
        return cls(word2idx, unk_token)


class VocabularyBuilder(object):
    def __init__(self, max_size=None,
                 min_freq=None,
                 pad_token="[PAD]",
                 unk_token = "[UNK]",
                 cls_token = "[CLS]",
                 sep_token = "[SEP]",
                 mask_token = "[MASK]",
                 add_unused = False):
        self.max_size = max_size
        self.min_freq = min_freq
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unk_token = unk_token
        self.add_unused = add_unused
        self.word_counter = Counter()

    def create_init_word2id(self):
        """初始化word2idx"""
        ctrl_symbols = [self.pad_token,self.unk_token,self.cls_token,self.sep_token,self.mask_token]
        word2idx = {syb:index for index,syb in enumerate(ctrl_symbols)}
        if self.add_unused:
            for i in range(20):
                word2idx[f'[UNUSED{i}]'] = len(word2idx)
        return word2idx

    def update(self, word_list):
        '''
        依次增加序列中词在词典中的出现频率
        :param word_list:
        :return:
        '''
        self.word_counter.update(word_list)

    def add(self, word):
        '''
        增加一个新词在词典中的出现频率
        :param word:
        :return:
        '''
        self.word_counter[word] += 1

    def build_vocab(self):
        word2idx = self.create_init_word2id()
        max_size = min(self.max_size, len(self.word_counter)) if self.max_size else None
        words = self.word_counter.most_common(max_size)
        if self.min_freq is not None:
            words = filter(lambda kv: kv[1] >= self.min_freq, words)
        for w, _ in words:
            if w not in word2idx:
                word2idx[w] = len(word2idx)
        vocab =  Vocabulary(word2idx=dict(word2idx), unk_token=self.unk_token)
        return vocab

    def clear(self):
        """
        删除Vocabulary中的词表数据。相当于重新初始化一下。
        :return:
        """
        self.word_counter.clear()

        
def get_or_build_vocab(conf):
    vocab_path = conf.vocab_path
    if os.path.exists(vocab_path):
        print("load_vocab....")
        vocab = Vocabulary.load_from_file(vocab_path)
    else:
        builder = VocabularyBuilder()
        fpaths = [conf.train_data_path]
        for fpath in fpaths:
            data = load_json(fpath)
            for item in data:
                text = item['text']
                builder.update(list(text))
        vocab = builder.build_vocab()
        vocab.save(vocab_path)
    return vocab
