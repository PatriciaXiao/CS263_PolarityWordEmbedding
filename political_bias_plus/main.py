'''
forcing GloVe (originally dim 300) to capture the polarity in k dimensions
adapted from https://nlpython.com/implementing-glove-model-with-pytorch/
loss is from https://pdfs.semanticscholar.org/babb/f74939612ee2f0203c30a190b4b95881415b.pdf
if using pretrained weights:
			 https://gist.github.com/MatthieuBizien/de26a7a2663f00ca16d8d2558815e9a6
             https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.word_to_vector.html
             https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/word_to_vector/glove.html#GloVe
by Patricia Xiao
'''

import torch
import torch.optim as optim

import numpy as np

from glove.dataset import GloveDataset, PairedPolitical
from glove.model import GloveModel
from glove.loss import Loss
from glove.visualize import GloveVisualize

import pandas as pd
import argparse
import random
import numpy as np
import os
import torch

parser = argparse.ArgumentParser("Political Biased Embedding")
parser.add_argument('--debug', default=False, action='store_true',
                    help='if in debug mode or not')
parser.add_argument('--random', default=False, action='store_true',
                    help='if use real random or not')
parser.add_argument('--visualize', default=False, action='store_true',
                    help='if to visualize or not')
parser.add_argument('-e', '--epochs', default=2, type=int,
                    help='number of epochs')
parser.add_argument('-x', '--x_max', default=100, type=int,
                    help='x_max value for GloVe')
parser.add_argument('--beta', default=1., type=float,
                    help='the beta value for l_D (l_2 mode)')
parser.add_argument('--lambda_d', default=1e-5, type=float,
                    help='the lambda value for l_D')
parser.add_argument('--lambda_e', default=0.8, type=float,
                    help='the lambda value for l_E')
parser.add_argument('--phrase_threshold', default=0.8, type=float,
                    help='the threshold of the phrases to be kept')
parser.add_argument('--save_file', default="tokens.csv,matrix.csv", type=str,
                    help='which mode of L_D to use')
args = parser.parse_args()

if not args.random:
    rd_seed = 36
    np.random.seed(rd_seed)
    random.seed(rd_seed)
    torch.manual_seed(rd_seed)
    # if CUDA:
    #     torch.cuda.manual_seed(rd_seed)

# whether or not cuda is enabled in this system
CUDA = torch.cuda.is_available()
# in debug mode or not
DEBUG = args.debug
# if no limit in dataset size, then this is None
DATASET_MAX_SIZE = None # 10000000
# how many dimensions are used to capture polarity (10 out of 200 by default)
POLARITY_DIM = 10 		# polarity dimension, set according to Kaiwei's paper
EMBED_DIM = 200			# total embedding dimension, set according to a pre-trained twitter GloVe embedding
# set according to the GloVe paper
X_MAX = args.x_max      # default: 100
ALPHA = 0.75
# follows Jieyu paper settings
BETA = args.beta
# the regularization term, if following Jieyu & Kaiwei's paper then they are both 0.8
LAMBDA_D = args.lambda_d # 0.001
LAMBDA_E = args.lambda_e
# settings
FAST_MODE = True
PARSE_MODE = "phrase" #"word"
# load the dataset
debug_data = ["../data/toy.txt"]
mini_data = ["../data/democratic_cleaned_min.txt", "../data/republican_cleaned_min.txt"]
data_fname = debug_data if DEBUG else mini_data
# ("../data/political_pairs.tsv", "../data/political_pairs.tsv")
political_pair_data = ("../data/pairs/democratic_2005.txt", "../data/pairs/republican_2005.txt")
hashtag_pair_data = "../data/hashtag_pairs.csv"
# the pair info
polarity_words = PairedPolitical(political_pair_data, hashtag_pair_data, parse_mode=PARSE_MODE, phrase_threshold=args.phrase_threshold)
# dataset, model, loss
load_file = ["debug_tokens.csv", "debug.csv"] if DEBUG else args.save_file.split(",")
dataset = GloveDataset("\n".join([open(fname, encoding="utf8").read() for fname in data_fname]), n_words=DATASET_MAX_SIZE, parse_phrase=polarity_words.phrases, cuda=CUDA, load_file=load_file, debug=DEBUG)
glove = GloveModel(dataset.size(), EMBED_DIM)
print("Dataset created from the file {}".format(data_fname))
print("\t# of tokens: {}".format(len(dataset)))
print("\tVocabulary size: {}".format(dataset.size()))
polarity_words.filter(dataset)
# the loss
glove_loss = Loss(X_MAX, ALPHA, BETA, polarity_words=polarity_words, polarity_dimensions=POLARITY_DIM, lambda_d = LAMBDA_D, lambda_e = LAMBDA_E, cuda=CUDA)
# the visualizer
visualizer = GloveVisualize(glove, dataset, polarity_words=polarity_words, polarity_dimensions=POLARITY_DIM)
if CUDA:
    glove.cuda()

# store the words in vocabulary
vocab_file_name = "vocabulary_debug.csv" if DEBUG else "vocabulary.csv"
if not os.path.exists(vocab_file_name):
    vocab = list(dataset._word2id.keys())
    data = {"word": vocab}
    df = pd.DataFrame(data=data)
    df.to_csv(vocab_file_name, index=None)

optimizer = optim.Adagrad(glove.parameters(), lr=0.05)
BATCH_SIZE = 4096 #2048
interval = 10 if DEBUG else 100
VISUALIZE = args.visualize
n_batches = int(len(dataset._xij) / BATCH_SIZE)
loss_values = list()
for e in range(1, args.epochs+1):
    batch_i = 0

    # calculate vg at the very begining of each epoch --- Jieyu's paper
    glove_loss.calc_vg(glove.embedding_numpy())
    
    for x_ij, i_idx, j_idx in dataset.get_batches(BATCH_SIZE):
        
        batch_i += 1
        
        optimizer.zero_grad()

        outputs = glove(i_idx, j_idx)
        # the original glove loss
        # loss = glove_loss.glove_loss(x_ij, outputs)
        loss = glove_loss.joint_loss(x_ij, outputs, i_idx, j_idx, glove.embedding())
        
        loss.backward() # retain_graph=True will be needed if we didn't use vg properly
        
        optimizer.step()
        
        loss_values.append(loss.item())

        if batch_i % interval == 0:
            # we propose to update the vg more oftenly
            glove_loss.calc_vg(glove.embedding_numpy())
            print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(e, args.epochs, batch_i, n_batches, np.mean(loss_values[-20:])))
    if not DEBUG:
        glove_embeddings = glove.embedding_numpy()
        np.savez_compressed("./embeddings.npz", glove=glove_embeddings)

if VISUALIZE:
    visualizer.visualize_loss(loss_values, debug=DEBUG)
    visualizer.visualize_embedding(debug=DEBUG)
    visualizer.visualize_embedding(mode="polarity", debug=DEBUG)
    visualizer.visualize_embedding(mode="semantic", debug=DEBUG)








