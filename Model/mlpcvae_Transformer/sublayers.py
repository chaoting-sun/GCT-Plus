import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy

# layernorm

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

# Multi-head attention

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    scores_attn = scores

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, scores_attn


class MultiHeadAttention(nn.Module):
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

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores, scores_attn = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        concat_attn = scores_attn.transpose(1, 2).contiguous().view(bs, -1, scores_attn.size(-1) * self.h)

        return output, concat_attn


# def attention(query, key, value, mask=None, dropout=None):
#     "Compute 'Scaled Dot Product Attention'"
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#     # print("score, mask:", scores.size(), mask.size())

#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#     p_attn = F.softmax(scores, dim=-1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn


# class MultiHeadAttention(nn.Module):
#     def __init__(self, h, d_model, dropout=0.1):
#         "Take in model size and number of heads."
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % h == 0
#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h
#         # 4 linear layers
#         self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, query, key, value, mask=None):
#         if mask is not None:
#             # Same mask applied to all h heads.
#             mask = mask.unsqueeze(1)
#         bs = query.size(0)

#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         query, key, value = \
#             [l(x).view(bs, -1, self.h, self.d_k).transpose(1, 2)
#              for l, x in zip(self.linears, (query, key, value))]
        
#         # 2) Apply attention on all the projected vectors in batch.
#         x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

#         # 3) "Concat" using a view and apply a final linear.
#         x = x.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)

#         concat_attn = self.attn.transpose(1, 2).contiguous().view(bs, -1, self.attn.size(-1) * self.h)

#         return self.linears[-1](x), concat_attn # (batch_size, max_length_source, d_model)

# Feed-forward layer

class FeedForward(nn.Module):
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
