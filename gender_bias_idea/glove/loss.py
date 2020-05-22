import torch
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA

class Loss(object):
    def __init__(self, x_max, alpha, beta, polarity_words, polarity_dimensions, lambda_d=0.8, lambda_e=0.8, cuda=False):
        self.x_max = x_max
        self.alpha = alpha
        self.beta = beta
        self.polarity_words = polarity_words
        self.polarity_dimensions = polarity_dimensions
        self.lambda_d = lambda_d
        self.lambda_e = lambda_e
        self.cuda = cuda
        self.vg = None

    def weight_func(self, x):
        '''
        f(x) in GloVe paper
        '''
        wx = (x/self.x_max)**self.alpha
        wx = torch.min(wx, torch.ones_like(wx))
        return wx.cuda() if self.cuda else wx

    def wmse_loss(self, weights, inputs, targets):
        loss = weights * F.mse_loss(inputs, targets, reduction='none') # square, in the original paper
        mean_loss = torch.mean(loss)
        return mean_loss.cuda() if self.cuda else mean_loss

    def glove_loss(self, x_ij, outputs):
        '''
        original Glove loss
        '''
        weights_x = self.weight_func(x_ij)
        loss = self.wmse_loss(weights_x, outputs, torch.log(x_ij))
        return loss

    def pair_difference(self, embedding, idx_set, mode="l1"):
        '''
        check if the current i,j pair involves polarity words
        L_D loss
        '''
        # print(embedding.size()) # [vocab_size, embedding_dim]
        polarity_pairs_ids = self.polarity_words.get_related_pairs(idx_set)
        if len(polarity_pairs_ids[0]) == 0:
            # avoid floating point error
            return 0
        pair_embedding = torch.stack(([embedding[ids] for ids in polarity_pairs_ids]))[:,:,-self.polarity_dimensions:]
        # print(pair_embedding.size()) # 2, n_pairs, polarity_dimensions
        if mode == "l1":
            sums = torch.sum(pair_embedding, dim=1) # 2, polarity_dimensions
            loss = -F.l1_loss(sums[0], sums[1], reduction='none') # 'mean'
        elif mode == "l2":
            e = torch.ones(pair_embedding.shape[-1])
            if self.cuda:
                e = e.cuda()
            beta_1 = -self.beta
            beta_2 = self.beta
            loss = F.mse_loss(pair_embedding[0], beta_1 * e, reduction='none') \
                    + F.mse_loss(pair_embedding[1], beta_2 * e, reduction='none')
        else:
            print("unknown norm for L_D: {}".fromat(mode))
            exit(0)
        return torch.mean(loss)

    def calc_vg_realtime(self, embedding):
        # the direction of the polarity embedding
        polarity_pairs_ids = self.polarity_words.polarity_pairs_ids
        pair_base_embedding = torch.stack(([embedding[ids] for ids in polarity_pairs_ids]))[:,:,:-self.polarity_dimensions]
        # print(pair_base_embedding.size()) # (2, vocab_size, embedding_dim - polarity_dimensions)
        pair_diff_embedding = pair_base_embedding[0] - pair_base_embedding[1]
        n_pairs = len(pair_diff_embedding)
        if n_pairs:
            vg = torch.sum(pair_diff_embedding, dim=0) / n_pairs
        else:
            print("ERROR: missing pairs --- don't have word pairs")
            exit(0)
        vg.unsqueeze_(0) # from (embedding_dim - polarity_dimensions,) to (1, embedding_dim - polarity_dimensions)
        return vg.detach()

    def calc_vg(self, embedding):
        # calculate the direction of the polarity embedding
        # in Kaiwei's paper, there are many pairs, to save time they calculated once an epoch
        # but in our case, the dataset will be extreamly large, at least we shouldn't calculate it once per epoch,
        # maybe calculated by the steps
        # note: this variable should either has grad disabled, or be recalculated every time we call it
        polarity_pairs_ids = self.polarity_words.polarity_pairs_ids
        pair_base_embedding = np.array([embedding[ids] for ids in polarity_pairs_ids])[:,:,:-self.polarity_dimensions]
        # print(pair_base_embedding.size()) # (2, vocab_size, embedding_dim - polarity_dimensions)
        pair_diff_embedding = pair_base_embedding[0] - pair_base_embedding[1]
        n_pairs = len(pair_diff_embedding)
        if n_pairs:
             self.vg = np.average(pair_diff_embedding, axis=0).reshape(1,-1)
        else:
            print("ERROR: missing pairs --- don't have word pairs")
            exit(0)
        # from (embedding_dim - polarity_dimensions,) to (1, embedding_dim - polarity_dimensions)
        # print(self.vg.shape)

    def stay_perpendict(self, embedding, idx_set):
        '''
        L_E: let the w^{(a)}
        '''
        polarity_set = self.polarity_words.polarity_ids_set
        neutral_word_ids = list(idx_set - polarity_set)
        neutral_embeddings = embedding[neutral_word_ids][:,:-self.polarity_dimensions]
        # print(neutral_embeddings.size()) # (n_neutral, embedding_dim-polarity_dimensions)
        zeros = torch.zeros(neutral_embeddings.size()[0])
        if self.cuda:
            zeros = zeros.cuda()
        vg = torch.Tensor(self.vg)
        # vg = self.calc_vg_realtime(embedding)
        loss = F.mse_loss(torch.mm(vg, neutral_embeddings.t())[0], zeros, reduction='none') # 'sum'
        return torch.mean(loss)

    def joint_loss(self, x_ij, outputs, i_idx, j_idx, embedding, ld_mode="l1"):
        original_loss = self.glove_loss(x_ij, outputs)
        idx_set = set(i_idx.cpu().numpy()).union(j_idx.cpu().numpy())
        difference_loss = self.pair_difference(embedding, idx_set, mode=ld_mode)
        null_space_loss = self.stay_perpendict(embedding, idx_set)
        loss = original_loss + self.lambda_d * difference_loss + self.lambda_e * null_space_loss
        return loss


