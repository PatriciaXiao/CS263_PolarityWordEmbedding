import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torchnlp.word_to_vector import GloVe

import numpy as np

glove_embedding = GloVe(cache="../../Twitter_Ideology_Prediction/data/post_processing/.word_vectors_cache")

# print(vars(glove_embedding).keys())
# print(dir(glove_embedding))
# print(type(glove_embedding.stoi))

vocabulary = list(glove_embedding.stoi.keys())
dictionary = dict(zip(list(range(len(vocabulary))), vocabulary)) 
word2idx = glove_embedding.stoi                                     # reversed_dict

'''
print(len(vocabulary))
print(word2idx["trump"])
print(word2idx["Trump"])
print(word2idx["obama"])
print(word2idx["Obama"])
print(word2idx["democratic"])
print(word2idx["Democratic"])
print(word2idx["republican"])
print(word2idx["Republican"])
'''

# n = 1000
n = 40000
top_k = 10
dim = 300

valid_range = list(range(n))

embedding_slice = np.zeros((n, dim))

for i in range(n):
    embedding_slice[i] = glove_embedding[vocabulary[i]]

similarity = np.matmul(embedding_slice, np.transpose(embedding_slice))

words_for_test = ["trump", "Trump", "obama", "Obama", "democratic", "Democratic", "republican", "Republican", "gun", "healthcare"]
idx_for_test = list(map(word2idx.get, words_for_test))
idx_for_test = [i for i in idx_for_test if i in valid_range]
words_for_test = list(map(dictionary.get, idx_for_test))
print("test set: {}".format(",".join(words_for_test)))

nearest_idx = [(-similarity[i, :]).argsort()[1:top_k + 1] for i in idx_for_test]
nearest_words = [list(map(dictionary.get, idx_lst)) for idx_lst in nearest_idx]

for w,ws in zip(words_for_test, nearest_words):
    print("nearest words to {} are: {}".format(w, " ".join(ws)))








