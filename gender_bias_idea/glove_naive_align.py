# https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.word_to_vector.html

from torchnlp.word_to_vector import GloVe  # doctest: +SKIP
import numpy as np
from numpy import linalg as LA
import pandas as pd
# this is only my personal local path, feel free to use other options in your implementation
vectors = GloVe(cache="../../Twitter_Ideology_Prediction/data/post_processing/.word_vectors_cache")  # doctest: +SKIP

# todo: add more and get the fair hyperplane, e.g. by PCA etc.
target_pairs = [["republican", "democratics"]]

# print(vectors[target_pairs[0]]) # tensor
# but tensors could be used this way:
# test = np.ones((3, 300))
# test[0] = vectors[target_pairs[0]][0]
# print(test)
# exit(0)


embedding_pairs = np.array([vectors[pair].numpy() for pair in target_pairs])
# print(embedding_pairs.shape)

# find the hyperplane
#     H = { x | u^T x = v }
#     u^T a = v
#     u^T b = v
#     u^T (a - b) = 0
#     u^T is the direction where decides the norm of the hyperplane

''' this is wrong
a = embedding_pairs[0,0]
b = embedding_pairs[0,1]
in_plane = a - b
u = np.ones(in_plane.shape) # therefore it wouldn't be a zero vector which is meaningless
i = 0
for i in range(in_plane.shape[0]):
    if in_plane[i]:
        break
sum_x = np.sum(in_plane) - in_plane[i]
u[i] = -1. * sum_x / in_plane[i]

# print(in_plane)
# print(u)

# in theory the same
# print(np.dot(u, a))
# print(np.dot(u, b))

# v = np.average([np.dot(u, a), np.dot(u, b)])
# print(v)
'''

u = embedding_pairs[0,0] - embedding_pairs[0,1]
v = np.dot(u, (embedding_pairs[0,0] + embedding_pairs[0,1])) / 2

# check all the words
df = pd.read_csv("../word2vec/words_dictionary.csv", sep='\t')
words = list(df['word'])
validation_words = ["gay", "gun", "free", "trump", "california", "hilary", "obama", "immigrant"]
valid_size = len(validation_words)


# next: 
# 1. use word2vec-trained embeddings - because we actually need the special tokens such as HashTag
# 2. use more word pairs instead of just one (and PCA?)

def projection(x, y):
    '''
    projection from x onto y (vector)
    '''
    return np.dot(x, y) / LA.norm(y)

# the translation:
#     projection onto hyperplane 
#         H = { x | u^T x = v }
#     P_H(x) = x - (u^T x - v) / l2_norm(u)
def proj2norm(norm=u, v=v):
    divided_by = LA.norm(norm)
    return lambda x: (np.dot(x, norm) - v) / divided_by

projection_function = proj2norm()

# print(projection_function(vectors["hi"].numpy()))
# exit(0)

def translate(x, proj_func=projection_function):
    proj_vec = projection_function(x)
    translated_vec = x - 2 * proj_vec
    return translated_vec


def judge_side(x, norm=u):
    prod = np.dot(x, norm)
    if prod > 0:
        return 1. 
    elif prod == 0:
        return 0.
    else:
        return -1.

vocabulary_embedding = vectors[words].numpy()
norm_to_vocab = LA.norm(vocabulary_embedding)
normalized_embeddings = vocabulary_embedding / norm_to_vocab

# projected onto the other half of the hyperplane
validation_embeddings = vectors[validation_words].numpy() / norm_to_vocab
translated_embeddings = np.zeros(validation_embeddings.shape)
sides_embeddings = np.zeros(validation_embeddings.shape[0])
for i in range(translated_embeddings.shape[0]):
    translated_embeddings[i] = translate(validation_embeddings[i])
    sides_embeddings[i] = judge_side(validation_embeddings[i])

# find the most similar words to their projection
similarity = np.matmul(translated_embeddings, np.transpose(normalized_embeddings))

# print(similarity.shape) # (8, 340010)

for i in range(valid_size):
    valid_word = validation_words[i]
    top_k = 8    # number of nearest neighbors
    nearest = (-similarity[i, :]).argsort()[1:top_k + 1]
    log_str = 'Other side of %s:' % valid_word
    for k in range(top_k):
        close_word = words[nearest[k]]
        log_str = '%s %s,' % (log_str, close_word)
    log_str += "\n\tit is side {}".format(sides_embeddings[i])
    print(log_str)



