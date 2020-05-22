import torch
import torch.nn as nn

class GloveModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)
        
        # could load pre-trained data here first
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

    def embedding(self):
        emb_i = self.wi.weight
        emb_j = self.wj.weight
        emb = emb_i + emb_j
        return emb

    def embedding_numpy(self):
        emb_i = self.wi.weight.cpu().data.numpy()
        emb_j = self.wj.weight.cpu().data.numpy()
        emb = emb_i + emb_j
        return emb


