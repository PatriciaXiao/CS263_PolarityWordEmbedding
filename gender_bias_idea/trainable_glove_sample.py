'''
trainable glove
https://nlpython.com/implementing-glove-model-with-pytorch/
'''

from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE

class GloveDataset:
    
    def __init__(self, text, n_words=200000, window_size=5, cuda=False):
        self._window_size = window_size
        self.cuda = CUDA
        self._tokens = text.split()[:n_words]
        word_counter = Counter()
        word_counter.update(self._tokens)
        self._word2id = {w:i for i, (w,_) in enumerate(word_counter.most_common())}
        self._id2word = {i:w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)
        
        self._id_tokens = [self._word2id[w] for w in self._tokens]
        
        self._create_coocurrence_matrix()
        
        print("# of words: {}".format(len(self._tokens)))
        print("Vocabulary length: {}".format(self._vocab_len))
        
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
    
    
    def get_batches(self, batch_size):
        #Generate random idx
        rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))
        
        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]

CUDA = torch.cuda.is_available()

# dataset = GloveDataset(open("../data/text8").read(), 10000000, cuda=CUDA)
dataset = GloveDataset(open("../data/toy.txt").read(), 10000000, cuda=CUDA)

EMBED_DIM = 200
class GloveModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)
        
        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()
        
    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()
        
        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j
        
        return x
glove = GloveModel(dataset._vocab_len, EMBED_DIM)
if CUDA:
    glove.cuda()


def weight_func(x, x_max, alpha, cuda=False):
    wx = (x/x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx.cuda() if cuda else wx

def wmse_loss(weights, inputs, targets, cuda=False):
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    mean_loss = torch.mean(loss)
    return mean_loss.cuda() if cuda else mean_loss


optimizer = optim.Adagrad(glove.parameters(), lr=0.05)


N_EPOCHS = 5 #100
BATCH_SIZE = 2048
X_MAX = 100
ALPHA = 0.75
interval = 10
n_batches = int(len(dataset._xij) / BATCH_SIZE)
loss_values = list()
for e in range(1, N_EPOCHS+1):
    batch_i = 0
    
    for x_ij, i_idx, j_idx in dataset.get_batches(BATCH_SIZE):
        
        batch_i += 1
        
        optimizer.zero_grad()
        
        outputs = glove(i_idx, j_idx)
        weights_x = weight_func(x_ij, X_MAX, ALPHA, cuda=CUDA)
        loss = wmse_loss(weights_x, outputs, torch.log(x_ij), cuda=CUDA)
        
        loss.backward()
        
        optimizer.step()
        
        loss_values.append(loss.item())
        
        # print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(e, N_EPOCHS, batch_i, n_batches, np.mean(loss_values[-20:])))  

        if batch_i % interval == 0:
            print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(e, N_EPOCHS, batch_i, n_batches, np.mean(loss_values[-20:])))  
    
    # print("Saving model...")
    # torch.save(glove.state_dict(), "state_dict.pt")

# visualizing the loss
def visualize_loss():
    plt.plot(loss_values)
    plt.show()
visualize_loss()

# visualizing embedding
def visualize_embedding(top_k = 300):
    emb_i = glove.wi.weight.cpu().data.numpy()
    emb_j = glove.wj.weight.cpu().data.numpy()
    emb = emb_i + emb_j
    tsne = TSNE(metric='cosine', random_state=123)
    embed_tsne = tsne.fit_transform(emb[:top_k, :])
    fig, ax = plt.subplots(figsize=(14, 14))
    for idx in range(top_k):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(dataset._id2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

    plt.show()

# visualize_embedding()




