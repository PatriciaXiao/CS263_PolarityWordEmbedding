from glove.embedding import GloveEmbedding
from glove.dataset import PairedDataset, PairedPolitical
from glove.visualize import GloveVisualize

import numpy as np

PARSE_MODE = "phrase"
POLARITY_DIM = 1

glove_embedding = GloveEmbedding()

first_n = 50000 # 20000 is more affordable
top_k = 10

valid_range = list(range(first_n))

most_frequent_words = glove_embedding.embeddings[:first_n]
similarity = np.matmul(most_frequent_words, np.transpose(most_frequent_words))
reversed_dict = glove_embedding._word2id
dictionary = glove_embedding._id2word

# print(most_frequent_words.shape)
# print(similarity.shape)

words_for_test = ["trump", "obama", "hilary", "gun", "healthcare"]

idx_for_test = list(map(reversed_dict.get, words_for_test))
idx_for_test = [i for i in idx_for_test if i in valid_range]
words_for_test = list(map(dictionary.get, idx_for_test))
print("test set: {}".format(",".join(words_for_test)))

nearest_idx = [(-similarity[i, :]).argsort()[1:top_k + 1] for i in idx_for_test]
nearest_words = [list(map(dictionary.get, idx_lst)) for idx_lst in nearest_idx]

for w,ws in zip(words_for_test, nearest_words):
    print("nearest words to {} are: {}".format(w, " ".join(ws)))