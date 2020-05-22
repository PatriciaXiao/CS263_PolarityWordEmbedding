from glove.embedding import GloveEmbedding
#from torchnlp.word_to_vector import GloVe
from glove.dataset import PairedDataset, PairedPolitical
from glove.visualize import GloveVisualize

from naive_classify import NaiveClassify

import torch.optim as optim
import torch

import numpy as np

from sklearn.metrics import f1_score as lib_f1_score

PARSE_MODE = "phrase"
POLARITY_DIM = 10
FULL_EMBEDDING = 200
BATCH_SIZE = 10000
MODE = "polarity" # "symantic" #"all" #"polarity"
if MODE == "polarity":
    EMBEDDING_SIZE = POLARITY_DIM
    EMB_START = FULL_EMBEDDING - POLARITY_DIM
    EMB_END = FULL_EMBEDDING
elif MODE == "symantic":
    EMBEDDING_SIZE = FULL_EMBEDDING - POLARITY_DIM
    EMB_START = 0
    EMB_END = FULL_EMBEDDING - POLARITY_DIM
else: 
    EMBEDDING_SIZE = FULL_EMBEDDING
    EMB_START = 0
    EMB_END = FULL_EMBEDDING

N_EPOCH = 1 #0 # 100 # 10 # 600
DEBUG = True #False

glove_embedding = GloveEmbedding()
# glove_embedding = GloVe(cache="../../Twitter_Ideology_Prediction/data/post_processing/.word_vectors_cache")
# GloVe.get = GloVe.__getitem__

political_pair_data = ("../data/pairs/democratic_2005.txt", "../data/pairs/republican_2005.txt")
hashtag_pair_data = "../data/hashtag_pairs.csv"
# the pair info
print("loading political pair info")
polarity_words = PairedPolitical(political_pair_data, hashtag_pair_data, parse_mode=PARSE_MODE)
polarity_words.filter(glove_embedding)


labeled_datafile = ["../data/democratic_cleaned.txt", "../data/republican_cleaned.txt"]
# for debug and fast iteration
if DEBUG:
    labeled_datafile = ["../data/toy.txt", "../data/toy.txt"]
print("loading labeled dataset info")
datasets = PairedDataset([open(fname, encoding="utf8").read() for fname in labeled_datafile], parse_phrase=polarity_words.phrases, split=True)

net = NaiveClassify(EMBEDDING_SIZE, 200, 2)

learning_rate = 0.3 # 0.1 # 0.01
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1_score(output, labels, average='macro'):
    '''
    average could be macro, micro, weighted
    '''
    preds = output.max(1)[1].type_as(labels)
    return lib_f1_score(labels, preds, average=average)

def run_epoch(data_loader, epoch=None, test=False):
    loss_sum = 0
    acc_sum = 0
    f1_sum = 0
    cnt = 0
    for i, data in enumerate(data_loader):
        cnt += 1
        # get the inputs; data is a list of [inputs, labels]
        str_inputs, labels = data
        inputs = np.array([np.average(glove_embedding[string], axis=0)[EMB_START:EMB_END] for string in str_inputs])
        # load into tensor
        inputs = torch.from_numpy(inputs)
        labels = torch.LongTensor(labels)

        # zero the parameter gradients
        if not test: optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = net.loss(outputs, labels)

        if not test:
            loss.backward()
            optimizer.step()

        loss_sum += loss.item()
        print("\t{} {}batch {}: loss {}".format("testing" if test else "training", "epoch {} ".format(epoch + 1) if epoch else "", i+1, loss.item()))
        acc = accuracy(outputs.cpu(), labels.cpu())
        f1 = f1_score(outputs.cpu(), labels.cpu())
        acc_sum += acc.item()
        f1_sum += f1.item()
    return loss_sum / cnt, acc_sum / cnt, f1_sum / cnt

print("loading labeled dataset info")
datasets = PairedDataset([open(fname, encoding="utf8").read() for fname in labeled_datafile], split=True)

print("start training")
for epoch in range(N_EPOCH):
    avg_loss, avg_acc, avg_f1 = run_epoch(datasets.train_loader(batch_size=BATCH_SIZE), epoch=epoch)

    # print statistics
    print("Training epoch {}: avg loss {}, avg acc {}, avg f1 {}".format(epoch + 1, avg_loss, avg_acc, avg_f1))

    avg_loss, avg_acc, avg_f1 = run_epoch(datasets.test_loader(batch_size=BATCH_SIZE), test=True)

    print("Test Results epoch {}: avg loss {}, avg acc {}, avg f1 {}".format(epoch + 1, avg_loss, avg_acc, avg_f1))

