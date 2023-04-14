import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sampler(nn.Module):
    def __init__(self, d_model, latent_dim, variational):
        super(Sampler, self).__init__()
        self.variational = variational
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)

    def sampling(self, mu, log_var):
        if self.variational:
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.sampling(mu, log_var)
        return z, mu, log_var
        

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        # scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = scores.masked_fill(mask == 0, -1e9)
        

    scores = F.softmax(scores, dim=-1)
    scores_attn = scores

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, scores_attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):        
        bs = q.size(0)

        # 1. perform linear operation and split into N heads
        # 2. transpose to get dimensions bs * N * sl * d_model
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        # print(q.size(), k.size(), v.size())

        # calculate attention using function we will define next
        scores, scores_attn = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat) # 不一樣
        # concat_attn = scores_attn.transpose(1, 2).contiguous().view(bs, -1, scores_attn.size(-1) * self.h)
        # return output, concat_attn
        return output


class FeedForward(nn.Module):
    """ Positional-Wise Feed-Forward Layer """
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # change to gelu. it is said to be good in Transformer
        x = F.gelu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
