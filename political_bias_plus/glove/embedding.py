import numpy as np
import pandas as pd

class GloveEmbedding(object):
    def __init__(self, load_files=("./vocabulary.csv", "./embeddings.npz")):
        vocab_file, embed_file = load_files
        self.unk = "<UNK>"
        self.embeddings = np.load(embed_file, 'r')["glove"]
        most_common_words = list(pd.read_csv(vocab_file)["word"])
        if self.unk not in most_common_words:
            self.embeddings = np.concatenate([self.embeddings, np.zeros((1,self.embeddings.shape[1]))])
            most_common_words += [self.unk]
        self._vocab_len = len(most_common_words)
        if self._vocab_len != self.embeddings.shape[0]:
            print("embedding and vocabulary shape ({}, {}) don't match, do you still want to continue? [y/n]".format(self._vocab_len, self.embeddings.shape[0]))
            willing = ""
            while len(willing) == 0:
                willing = input()
            if willing[0].lower() == 'y':
                self.embeddings = self.embeddings[:self._vocab_len]
            else:
                exit(0)
        words_indexes = list(range(self._vocab_len))
        self._word2id = dict(zip(most_common_words, words_indexes))
        self._id2word = dict(zip(words_indexes, most_common_words))
    def __len__(self):
        '''
        the number of words in vocabulary
        '''
        return self._vocab_len
    def __getitem__(self, words):
        '''
        any word's embedding
        '''
        if type(words) == str:
            return self.embeddings[self.safe_get(words)]
        if not type(words) == list:
            print("GloveEmbedding: expected input type be a string or a list of string")
            exit(0)
        if len(words) == 0:
            words.append(self.unk)
        word_id_list = list(map(self.safe_get, words))
        return self.embeddings[word_id_list] # if len(words) > 1 else self.embeddings[word_id_list][0]
    def size(self):
        return self.embeddings.shape[1]
    def safe_get(self, word):
        return self._word2id.get(word, self._word2id[self.unk])
    def get(self, word):
        return self._word2id.get(word, -1)
    def embedding_numpy(self):
        return self.embeddings