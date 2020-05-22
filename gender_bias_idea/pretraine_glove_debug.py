'''
forcing GloVe (originally dim 300) to capture the polarity in k dimensions
adapted from https://nlpython.com/implementing-glove-model-with-pytorch/
             https://gist.github.com/MatthieuBizien/de26a7a2663f00ca16d8d2558815e9a6
             https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.word_to_vector.html
             https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/word_to_vector/glove.html#GloVe
by Patricia Xiao
'''
# from torchnlp.word_to_vector import GloVe
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch

import numpy as np

import sys
sys.path.append("torchnlp_lib")

from torchnlp_lib.myglove import GloVe

# why not using this vector as Ideology Prediction module
vectors = GloVe(name="twitter.27B", dim=200, cache="../../Twitter_Ideology_Prediction/data/post_processing/.word_vectors_cache")
# vectors = GloVe(cache="../../Twitter_Ideology_Prediction/data/post_processing/.word_vectors_cache")

# conclusion: some hash-tags are not included
# print(vectors['#HelloWorld'])
# print(vectors['#hello'])

# print(vectors._get_token_vector("hello"))
# print(vectors.keys())
# print(vectors.get_map())
# the following two lines are equivalent
# print(vectors["hello"])
# print(vectors.vectors[vectors.get_map()["hello"]])
# vector_tokens = vectors.index_to_token # key: token, value: index
# vector_embeds = vectors.vectors

# print(len(vector_embeds))

pre_trained_embedding = vectors.vectors
pre_trained_keys = vectors.keys()
pre_trained_key2idx = vectors.get_map()


