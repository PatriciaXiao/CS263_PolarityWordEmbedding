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
# from torchnlp.word_to_vector import GloVe
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.optim as optim

import numpy as np

import sys
sys.path.append("torchnlp_lib")

from torchnlp_lib.myglove import GloVe


from glove.dataset import GloveDataset, PairedPolitical
from glove.model import GloveModel
from glove.loss import Loss
from glove.visualize import GloveVisualize

# why not using this vector as Ideology Prediction module
# pre_trained = GloVe(name="twitter.27B", dim=200, cache="../../Twitter_Ideology_Prediction/data/post_processing/.word_vectors_cache")

# whether or not cuda is enabled in this system
CUDA = torch.cuda.is_available()
# in debug mode or not
DEBUG = True #False # True
# if no limit in dataset size, then this is None
DATASET_MAX_SIZE = None # 10000000
# how many dimensions are used to capture polarity (10 out of 200 by default)
POLARITY_DIM = 1 		# polarity dimension, set according to Kaiwei's paper
EMBED_DIM = 200			# total embedding dimension, set according to a pre-trained twitter GloVe embedding
# set according to the GloVe paper
X_MAX = 100
ALPHA = 0.75
# follows Jieyu paper settings
BETA = 1
Ld_MODE = "l1" # "l1" or "l2"
# the regularization term, if following Jieyu & Kaiwei's paper then they are both 0.8
LAMBDA_D = 0.8 # 0.001
LAMBDA_E = 0.8
# settings
FAST_MODE = True
PARSE_MODE = "phrase" #"word"
# load the dataset
debug_data = ["../data/toy.txt"]
data_files = ["../data/democratic_cleaned_min.txt", "../data/republican_cleaned_min.txt"]
data_fname = debug_data if DEBUG else data_files
political_pair_data = "../data/political_pairs.tsv"
hashtag_pair_data = "../data/hashtag_pairs.csv"
# the pair info
polarity_words = PairedPolitical(political_pair_data, hashtag_pair_data, parse_mode=PARSE_MODE)
# dataset, model, loss
dataset = GloveDataset("\n".join([open(fname).read() for fname in data_fname]), n_words=DATASET_MAX_SIZE, parse_phrase=polarity_words.phrases, cuda=CUDA)
glove = GloveModel(dataset.size(), EMBED_DIM)
print("Dataset created from the file {}".format(data_fname))
print("\t# of tokens: {}".format(len(dataset)))
print("\tVocabulary size: {}".format(dataset.size()))
polarity_words.filter(dataset)
print("There are {} politically polarity pairs of words valid for this dataset".format(len(polarity_words)))
# the loss
glove_loss = Loss(X_MAX, ALPHA, BETA, polarity_words=polarity_words, polarity_dimensions=POLARITY_DIM, lambda_d = LAMBDA_D, lambda_e = LAMBDA_E, cuda=CUDA)
# the visualizer
visualizer = GloveVisualize(glove, dataset, polarity_words=polarity_words, polarity_dimensions=POLARITY_DIM)
if CUDA:
    glove.cuda()

optimizer = optim.Adagrad(glove.parameters(), lr=0.05)

N_EPOCHS = 2 if DEBUG else 1 # 1 # 3 #2 # 100
BATCH_SIZE = 2048
interval = 10 if DEBUG else 100
VISUALIZE = False
n_batches = int(len(dataset._xij) / BATCH_SIZE)
loss_values = list()
for e in range(1, N_EPOCHS+1):
    batch_i = 0

    # calculate vg at the very begining of each epoch --- Jieyu's paper
    glove_loss.calc_vg(glove.embedding_numpy())
    
    for x_ij, i_idx, j_idx in dataset.get_batches(BATCH_SIZE):
        
        batch_i += 1
        
        optimizer.zero_grad()

        outputs = glove(i_idx, j_idx)
        # the original glove loss
        # loss = glove_loss.glove_loss(x_ij, outputs)
        loss = glove_loss.joint_loss(x_ij, outputs, i_idx, j_idx, glove.embedding(), ld_mode=Ld_MODE)
        
        loss.backward() # retain_graph=True will be needed if we didn't use vg properly
        
        optimizer.step()
        
        loss_values.append(loss.item())

        if batch_i % interval == 0:
            # we propose to update the vg more oftenly
            glove_loss.calc_vg(glove.embedding_numpy())
            print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(e, N_EPOCHS, batch_i, n_batches, np.mean(loss_values[-20:])))

glove_embeddings = glove.embedding_numpy()
np.savez_compressed("./embeddings.npz", glove=glove_embeddings)

# sentence embedding


if VISUALIZE:
    visualizer.visualize_loss(loss_values, debug=DEBUG)
    visualizer.visualize_embedding(debug=DEBUG)
    visualizer.visualize_embedding(mode="polarity", debug=DEBUG)
    visualizer.visualize_embedding(mode="semantic", debug=DEBUG)

