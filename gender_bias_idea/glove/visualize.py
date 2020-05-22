import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


class GloveVisualize(object):
    def __init__(self, glove, dataset, polarity_words=None, polarity_dimensions=None):
        self.glove = glove
        self.dataset = dataset
        self.polarity_words=polarity_words
        self.polarity_dimensions = polarity_dimensions

    # visualizing the loss
    def visualize_loss(self, loss_values, debug=False):
        plt.plot(loss_values)
        plt.show() if debug else plt.savefig("visualize_loss.pdf")

    # visualizing embedding
    def visualize_embedding(self, top_k = 100, mode="all_dim", debug=False): # top_k = 300
        selected_idx = list(self.polarity_words.polarity_ids_set.union(range(top_k)))
        emb = self.glove.embedding_numpy()
        if mode == "polarity":
            emb = emb[:,-self.polarity_dimensions:]
        elif mode=="semantic":
            emb = emb[:,:-self.polarity_dimensions]
        if emb.shape[1] == 1:
            randomize_padding = np.random.random_sample((emb.shape[0],1))
            emb = np.concatenate([emb, randomize_padding], axis=1)
        if emb.shape[1] > 2:
            tsne = TSNE(metric='cosine', random_state=123)
            embed_tsne = tsne.fit_transform(emb[selected_idx])
        else:
            embed_tsne = emb
        idx2tmp_idx = dict(zip(selected_idx, range(len(selected_idx))))
        fig, ax = plt.subplots(figsize=(12, 12))
        for idx in selected_idx:
            tmp_idx = idx2tmp_idx[idx]
            plt.scatter(*embed_tsne[tmp_idx, :], color='steelblue')
            plt.annotate(self.dataset._id2word[idx], (embed_tsne[tmp_idx, 0], embed_tsne[tmp_idx, 1]), alpha=0.7)
        plt.show() if debug else plt.savefig("visualize_embedding_{}.pdf".format(mode))

