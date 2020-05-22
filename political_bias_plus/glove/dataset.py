import torch
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from functools import *
import pandas as pd
from iteration_utilities import deepflatten
import nltk
import math
import random
import os
import re

'''
from nltk.tokenize import MWETokenizer
tokenizer = MWETokenizer()
tokenizer.add_mwe(('the', 'west', 'wing'))
tokenizer.tokenize('Something about the west wing'.split())
'''

class Tokenizer:
    def __init__(self):
        # because my dataset is already pre-processed
        self.word_tokenizer = self.naive_split # nltk.word_tokenize
        self.pair_phrase_parser = nltk.MWETokenizer

    def naive_split(self, text):
        return text.split()

    def tokenize(self, text, n_words = None, parse_phrase=None, debug=False):
        words = self.word_tokenizer(text)
        if n_words:
            words = words[:n_words]
        if parse_phrase is not None and len(parse_phrase):
            phrases = words
            for phrase_list in parse_phrase:
                tmp_parser = self.pair_phrase_parser(phrase_list[1], separator=phrase_list[0])
                # using only " " as separator (then hashtags become "# XXX")
                # tmp_parser = self.pair_phrase_parser(phrase_list[1], " ")
                phrases = tmp_parser.tokenize(phrases)
            return phrases
        else:
            return words

    def __call__(self, text, **kwarg):
        return self.tokenize(text, **kwarg)

class GloveDataset:
    
    def __init__(self, text, n_words=200000, window_size=5, tokenizer=Tokenizer(), cuda=False, parse_phrase=None, threshold=2, load_file=None, debug=False):
        '''
        when n_words = None, accept all words 
        '''
        if load_file is not None:
            saved = os.path.exists(load_file[0]) and os.path.exists(load_file[1])
            if saved:
                print("loading from file {} and {}".format(load_file[0], load_file[1]))
            else:
                print("haven't cached file {} or {}, will do it at this run".format(load_file[0], load_file[1]))
        else:
            saved = False
            load_file = ["tmp1.csv", "tmp2.csv"] # to avoid compile error
        self.unk = "<UNK>"
        self.cuda = cuda
        self.tokenizer = tokenizer
        
        if os.path.exists(load_file[0]):
            self._tokens = list(pd.read_csv(load_file[0], sep="\t")["tokens"])
            vocab_file_name = "vocabulary_debug.csv" if debug else "vocabulary.csv"
            most_common_words = list(pd.read_csv(vocab_file_name)["word"])
        else:
            self._tokens = self.tokenizer(text, n_words = n_words, parse_phrase=parse_phrase, debug=True) # this is the slow step
            self.save_file(load_file[0], {"tokens": self._tokens})

        if os.path.exists("vocabulary.csv"):
            self._tokens = list(pd.read_csv(load_file[0], sep="\t")["tokens"])
            most_common_words = list(pd.read_csv("vocabulary.csv")["word"])
        else:
            word_counter = Counter()
            word_counter.update(self._tokens)
            # keep only who appears more than threshold times
            word_min = Counter(list(word_counter.keys()))
            word_remain = word_counter - word_min 
            word_min = Counter(list(word_remain.keys()))
            word_counter = word_remain+word_min
            #print(word_counter.most_common())
            most_common_words = list(map(lambda word_cnt: word_cnt[0], word_counter.most_common()))
            most_common_words += [self.unk]
        self._vocab_len = len(most_common_words)
        words_indexes = list(range(self._vocab_len))
        self._word2id = dict(zip(most_common_words, words_indexes))
        self._id2word = dict(zip(words_indexes, most_common_words))
        
        self._token_len = len(self._tokens)
        # self._id_tokens = list(map(self._word2id.get, self._tokens))
        self._id_tokens = list(map(self.__getitem__, self._tokens))

        self._create_coocurrence_matrix(window_size, load_file=load_file if load_file else None)


    def __len__(self):
        '''
        the number of words in total in the text dataset
        '''
        return self._token_len
    def __getitem__(self, word):
        return self._word2id.get(word, self._word2id[self.unk])
    def size(self):
        '''
        the vocabulary size
        '''
        return self._vocab_len

    def get(self, word):
        return self._word2id.get(word, self._word2id[self.unk])

    def save_file(self, filename, data):
        df = pd.DataFrame(data=data)
        df.to_csv(filename, sep="\t", index=None)
    def load_file(self, filename):
        return pd.read_csv(filename, sep="\t")
        
    def _create_coocurrence_matrix(self, window_size, load_file=None):
        matrix_file = load_file[1]
        if os.path.exists(matrix_file):
            matrix_df = self.load_file(matrix_file)
            self._i_idx = list(matrix_df["i_idx"])
            self._j_idx = list(matrix_df["j_idx"])
            self._xij = list(matrix_df["xij"])
        else:
            cooc_mat = defaultdict(Counter)
            for i, w in enumerate(self._id_tokens):
                start_i = max(i - window_size, 0)
                end_i = min(i + window_size + 1, len(self._id_tokens))
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
            matrix_data = {"i_idx": self._i_idx, "j_idx": self._j_idx, "xij": self._xij}
            self.save_file(matrix_file, matrix_data)
        
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
    def __init__(self, political_pair_fname, hashtag_pair_fname, parse_mode="word", tokenizer=Tokenizer(), phrase_threshold=0.8):
        '''
        parse-mode is either "word" or "phrase"
        '''
        self.parse_mode = parse_mode
        self.phrase_threshold = phrase_threshold
        self.tokenizer = tokenizer
        hashtag_df = pd.read_csv(hashtag_pair_fname, sep="\t")
        self.idx2party = list(hashtag_df.keys())
        self.hashtag_pairs = [  list(hashtag_df[self.idx2party[0]]), \
                                list(hashtag_df[self.idx2party[1]])]
        if parse_mode == "word":
            for i in range(len(self.hashtag_pairs)):
                for j in range(len(self.hashtag_pairs[i])):
                    self.hashtag_pairs[i][j] = self.hashtag_pairs[i][j][1:]
        democratic_fname, republican_fname = political_pair_fname
        democratic_df = pd.read_csv(democratic_fname, sep="\t")
        republican_df = pd.read_csv(republican_fname, sep="\t")
        self.political_pairs = [list(map(lambda x: x.lower(), democratic_df["Democratic"])), \
                                list(map(lambda x: x.lower(), republican_df["Republican"]))]
        self.polarity_pairs = [self.political_pairs[0] + self.hashtag_pairs[0], \
                               self.political_pairs[1] + self.hashtag_pairs[1] ]
        self.parse_phrase()

    def parse_phrase(self):  
        self.phrases = []
        if self.parse_mode == "phrase":
            # dealing with the hashtage
            phrase_segments = []
            for phrase_list in self.political_pairs:
                for phrase in phrase_list:
                    tmp_phrase_seg = tuple(self.tokenizer(phrase))
                    if len(tmp_phrase_seg) > 1:
                        phrase_segments.append(tmp_phrase_seg)
            auto_phrase_file = "../data/AutoPhrase_multi-words.txt"
            if os.path.exists(auto_phrase_file):
                df = pd.read_csv(auto_phrase_file, sep="\t", header=None)
                for confidence, phrase in zip(df[0], df[1]):
                    tmp_phrase_seg = tuple(self.tokenizer(phrase))

                    if confidence > self.phrase_threshold:
                        if len(tmp_phrase_seg) > 1:
                            phrase_segments.append(tmp_phrase_seg)
                    else:
                        break

            self.phrases.append((" ", phrase_segments))

    def filter(self, dataset=None):
        if self.parse_mode == "word":
            filtered_polarity_list = self.filter_words()
        elif self.parse_mode == "phrase":
            filtered_polarity_list = self.filter_phrases()
        self.filtered_polarity_pairs = self.keep_valid_pairs(self.polarity_pairs, filtered_polarity_list, dataset)
        self.filtered_polarity_set = self.unique_join(self.filtered_polarity_pairs[0], self.filtered_polarity_pairs[1])
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

    def get_related_pairs(self, set_of_ids):
        pairs = self.polarity_pairs_ids
        kept_pairs = [list(), list()]
        for i in range(2):
            for elem in pairs[i]:
                if elem in set_of_ids: kept_pairs[i].append(elem)
        return kept_pairs

    def keep_valid_pairs(self, pairs, unrepeated, dataset=None):
        assert len(pairs) == 2
        kept_pairs = [list(), list()]
        for i in range(2):
            for elem in pairs[i]:
                if dataset is not None and dataset.get(elem) == -1: continue
                if elem in unrepeated[i]:
                    kept_pairs[i].append(elem)
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

class PairedDataset(object):
    def __init__(self, text_data, tokenizer=Tokenizer(), parse_phrase=None, split = False):
        assert len(text_data) == 2, "not a paired dataset we want"
        test_set = None
        if split:
            text_data, test_set = self.split_data(text_data)
        else:
            text_data = [text.split("\n") for text in text_data]
        text_data = [line for line in text_data if len(line) > 0]
        if test_set:
            test_set = [line for line in test_set if len(line) > 0]
        self.tokenizer = tokenizer
        self.data = [[self.tokenizer(s, parse_phrase=parse_phrase) for s in text] for text in text_data]
        if test_set is not None:
            self.test_set = [[self.tokenizer(s, parse_phrase=parse_phrase) for s in text] for text in test_set]
        self.train_data = np.array(self.data[0] + self.data[1])
        self.train_label = np.array([0] * len(self.data[0]) + [1] * len(self.data[1]))
        self.test_data = np.array(self.test_set[0] + self.test_set[1])
        self.test_label = np.array([0] * len(self.test_set[0]) + [1] * len(self.test_set[1]) )

    def split_data(self, text_data, train_portion=0.8, min_len_sentence=6):
        train_set = list()
        test_set = list()
        for text in text_data:
            all_sentences = [s for s in re.split('\n|\r|\n\r', text) if len(s.split()) > min_len_sentence]
            random.shuffle(all_sentences)
            split_index = int(train_portion * len(all_sentences))
            train_set.append(all_sentences[:split_index])
            test_set.append(all_sentences[split_index:])
        return train_set, test_set

    def train_loader(self, batch_size = 1024):
        idx = 0
        idxs_list = list(range(len(self.train_data)))
        random.shuffle(idxs_list)
        while idx < len(idxs_list):
            start = idx
            end = idx + batch_size
            idx = end
            yield self.train_data[idxs_list[start:end]], self.train_label[idxs_list[start:end]]

    def test_loader(self, batch_size = 1024):
        idx = 0
        idxs_list = list(range(len(self.test_data)))
        random.shuffle(idxs_list)
        while idx < len(idxs_list):
            start = idx
            end = idx + batch_size
            idx = end
            yield self.test_data[idxs_list[start:end]], self.test_label[idxs_list[start:end]]


