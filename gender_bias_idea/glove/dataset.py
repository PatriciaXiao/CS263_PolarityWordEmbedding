import torch
import numpy as np
from collections import Counter, defaultdict
from functools import *
import pandas as pd
from iteration_utilities import deepflatten
import nltk

class Tokenizer:
    def __init__(self):
        self.word_tokenizer = nltk.word_tokenize
        self.pair_phrase_parser = nltk.MWETokenizer

    def tokenize(self, text, n_words = None, parse_phrase=None):
        words = self.word_tokenizer(text)
        if n_words:
            words = words[:n_words]
        if parse_phrase is not None and len(parse_phrase):
            phrases = words
            for phrase_list in parse_phrase:
                tmp_parser = self.pair_phrase_parser(phrase_list[1], separator=phrase_list[0])
                phrases = tmp_parser.tokenize(phrases)
            return phrases
        else:
            return words

    def __call__(self, text, **kwarg):
        return self.tokenize(text, **kwarg)

class GloveDataset:
    
    def __init__(self, text, n_words=200000, window_size=5, tokenizer=Tokenizer(), cuda=False, parse_phrase=None):
        '''
        when n_words = None, accept all words 
        '''
        self._window_size = window_size
        self.cuda = cuda
        self.tokenizer = tokenizer
        self._tokens = self.tokenizer(text, n_words = n_words, parse_phrase=parse_phrase)
        word_counter = Counter()
        word_counter.update(self._tokens)
        #print(word_counter.most_common())
        most_common_words = list(map(lambda word_cnt: word_cnt[0], word_counter.most_common()))
        self._vocab_len = len(most_common_words)
        words_indexes = list(range(self._vocab_len))
        self._word2id = dict(zip(most_common_words, words_indexes))
        self._id2word = dict(zip(words_indexes, most_common_words))
        
        self._token_len = len(self._tokens)
        self._id_tokens = list(map(self._word2id.get, self._tokens))
        
        self._create_coocurrence_matrix()


    def __len__(self):
        '''
        the number of words in total in the text dataset
        '''
        return self._token_len
    def size(self):
        '''
        the vocabulary size
        '''
        return self._vocab_len

    def get(self, word):
        return self._word2id.get(word, -1)
        
    def _create_coocurrence_matrix(self):
        cooc_mat = defaultdict(Counter)
        for i, w in enumerate(self._id_tokens):
            start_i = max(i - self._window_size, 0)
            end_i = min(i + self._window_size + 1, len(self._id_tokens))
            for j in range(start_i, end_i):
                if i != j:
                    c = self._id_tokens[j]
                    cooc_mat[w][c] += 1 / abs(j-i)
                    
        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()
        #Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)
        
        if self.cuda:
            self._i_idx = torch.LongTensor(self._i_idx).cuda()
            self._j_idx = torch.LongTensor(self._j_idx).cuda()
            self._xij = torch.FloatTensor(self._xij).cuda()
        else:
            self._i_idx = torch.LongTensor(self._i_idx)
            self._j_idx = torch.LongTensor(self._j_idx)
            self._xij = torch.FloatTensor(self._xij)
        self._n_ij = len(self._xij)

        # print(max(self._xij)) # in toy set around 614
        # exit(0)
    
    
    def get_batches(self, batch_size):
        # Generate unrepeated random idx
        rand_ids = torch.LongTensor(np.random.choice(self._n_ij, self._n_ij, replace=False))
        
        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]


class PairedPolitical(object):
    def __init__(self, political_pair_fname, hashtag_pair_fname, parse_mode="word", tokenizer=Tokenizer()):
        '''
        parse-mode is either "word" or "phrase"
        '''
        self.parse_mode = parse_mode
        self.tokenizer = tokenizer
        hashtag_df = pd.read_csv(hashtag_pair_fname, sep="\t")
        self.idx2party = list(hashtag_df.keys())
        self.hashtag_pairs = [  list(hashtag_df[self.idx2party[0]]), \
                                list(hashtag_df[self.idx2party[1]])]
        if parse_mode == "word":
            for i in range(len(self.hashtag_pairs)):
                for j in range(len(self.hashtag_pairs[i])):
                    self.hashtag_pairs[i][j] = self.hashtag_pairs[i][j][1:]
        political_df = pd.read_csv(political_pair_fname, sep="\t", header=None)
        self.political_pairs = [list(map(lambda x: x.lower(), political_df[0])), \
                                list(map(lambda x: x.lower(), political_df[1]))]
        self.polarity_pairs = [self.political_pairs[0] + self.hashtag_pairs[0], \
                               self.political_pairs[1] + self.hashtag_pairs[1] ]
        self.parse_phrase()

    def parse_phrase(self):  
        self.phrases = []
        if self.parse_mode == "phrase":
            # dealing with the hashtage
            hashtag_segments = []
            for hashtag_list in self.hashtag_pairs:
                for hashtag in hashtag_list:
                    hashtag_segments.append(tuple(self.tokenizer(hashtag)))
            self.phrases.append(("", hashtag_segments))
            phrase_segments = []
            for phrase_list in self.political_pairs:
                for phrase in phrase_list:
                    tmp_phrase_seg = tuple(self.tokenizer(phrase))
                    if len(tmp_phrase_seg) > 1:
                        phrase_segments.append(tmp_phrase_seg)
            self.phrases.append((" ", phrase_segments))

    def filter(self, dataset=None):
        if self.parse_mode == "word":
            filtered_polarity_list = self.filter_words()
        elif self.parse_mode == "phrase":
            filtered_polarity_list = self.filter_phrases()
        self.filtered_polarity_pairs = self.keep_valid_pairs(self.polarity_pairs, filtered_polarity_list, dataset)
        self.filtered_polarity_set = self.unique_join(self.filtered_polarity_pairs[0], self.filtered_polarity_pairs[1])
        self.n_pairs = len(self.filtered_polarity_pairs[0])
        # get the corresponding word id from the dataset
        self.polarity_pairs_ids = [list(map(lambda x: dataset.get(x), self.filtered_polarity_pairs[i])) for i in range(2)]
        self.polarity_ids_set = self.unique_join(self.polarity_pairs_ids[0], self.polarity_pairs_ids[1])

    def filter_phrases(self):
        polarity_phrases = [self.unique_join(self.hashtag_pairs[i], self.political_pairs[i]) for i in range(2)]
        _shared_phrases = self.intersection(polarity_phrases[0], polarity_phrases[1])
        filtered_polarity_phrases = [list(filter(lambda x: x not in _shared_phrases, polarity_phrases[i])) for i in range(2)]
        return filtered_polarity_phrases

    def filter_words(self):
        polarity_words = [self.unique_join(self.parse(self.hashtag_pairs[i]), self.parse(self.political_pairs[i])) for i in range(2)]
        _shared_words = self.intersection(polarity_words[0], polarity_words[1])
        # filtering the common words and those who are currently not usable
        filtered_polarity_words = [list(filter(lambda x: x not in _shared_words, polarity_words[i])) for i in range(2)]
        return filtered_polarity_words

    def __len__(self):
        return self.n_pairs

    def get_related_pairs(self, set_of_ids):
        pairs = self.polarity_pairs_ids
        kept_pairs = [list(), list()]
        for elem1, elem2 in zip(pairs[0], pairs[1]):
            if elem1 in set_of_ids or elem2 in set_of_ids:
                kept_pairs[0].append(elem1)
                kept_pairs[1].append(elem2)
        return kept_pairs

    def keep_valid_pairs(self, pairs, unrepeated, dataset=None):
        assert len(pairs) == 2 and len(pairs[0]) == len(pairs[1])
        kept_pairs = [list(), list()]
        for elem1, elem2 in zip(pairs[0], pairs[1]):
            if dataset is not None:
                if dataset.get(elem1) == -1 or dataset.get(elem2) == -1:
                    continue
            if elem1 in unrepeated[0] and elem2 in unrepeated[1]:
                kept_pairs[0].append(elem1)
                kept_pairs[1].append(elem2)
        return kept_pairs

    def parse(self, lst):
        return self.unique(self.flatten(list(map(lambda x: self.tokenizer(x), lst))))

    def flatten(self, lst, depth=1):
        return list(deepflatten(lst, depth=depth))

    def unique(self, lst):
        return list(set(lst))

    def unique_join(self, lst_1, lst_2):
        return set(lst_1).union(lst_2)

    def intersection(self, set_1, set_2):
        return set_1.intersection(set_2)

