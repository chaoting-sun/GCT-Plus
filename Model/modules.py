import math
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def nopeak_mask(trg_size, use_cond2dec, pad_idx, cond_dim=0):
    np_mask = np.triu(np.ones((1, trg_size, trg_size)), k=1).astype('uint8')
    if use_cond2dec == True:
        cond_mask = np.zeros((1, cond_dim, cond_dim))
        cond_mask_upperright = np.ones((1, cond_dim, trg_size))
        cond_mask_upperright[:, :, 0] = 0 # ??? cannot understand
        cond_mask_lowerleft = np.zeros((1, trg_size, cond_dim))
        upper_mask = np.concatenate([cond_mask, cond_mask_upperright], axis=2)
        lower_mask = np.concatenate([cond_mask_lowerleft, np_mask], axis=2)
        np_mask = np.concatenate([upper_mask, lower_mask], axis=1)
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    # if device is not None:
        # np_mask = np_mask.to(device)
    return np_mask*pad_idx


def create_condition_mask(conditions):
    condition_mask = torch.unsqueeze(conditions, -2) # (bs,nc)->(bs,1,nc)
    condition_mask = torch.ones_like(condition_mask, dtype=bool)
    return condition_mask


def create_source_mask(source, pad_idx, conditions=None):
    source_mask = (source != pad_idx).unsqueeze(-2) # (bs,strlen)->(bs,1,strlen)
    if conditions is not None:
        condition_mask = create_condition_mask(conditions) # (bs,1,nc)
        return torch.cat([condition_mask, source_mask], dim=2) # (bs,1,nc+strlen), T..TF..F
    return source_mask


def create_target_mask(target, pad_idx, conditions, use_cond2dec):
    """ 
    create a target mask composed of padding/sequence mask 
    target and conditions are tensors.
    """
    # padding mask
    target_mask = (target != pad_idx).unsqueeze(-2) # (bs,strlen)->(bs,1,strlen)
    if use_cond2dec == True:
        condition_mask = create_condition_mask(conditions) # (bs,1,nc)
        target_mask = torch.cat([condition_mask, target_mask], dim=2) # (bs,1,nc+strlen)
    # sequence mask
    np_mask = nopeak_mask(target.size(1), use_cond2dec, pad_idx,
                          conditions.size(-1)) # (bs,1,nc+strlen) or (bs,1,strlen)
    np_mask = np_mask.to(target.get_device())
    
    # torch.set_printoptions(edgeitems=100)
    # print(np_mask.size(), np_mask)
    # print(target_mask.size(), target_mask)

    # exit()
    
    return target_mask & np_mask


def create_masks(source, target, conditions, pad_idx, use_cond2dec=True):
    source_mask = create_source_mask(source, pad_idx, conditions)
    if target is not None:
        target_mask = create_target_mask(target, pad_idx, conditions, use_cond2dec)
    device = source.get_device()
    return source_mask.to(device), target_mask.to(device)


""" 
Clone Layers 
"""

def get_clones(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

""" 
Normalize Layers 
"""

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

"""
Embddings
"""

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # weight matrix, each row present one word
        self.embed = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.embed(x)
        return embed # * math.sqrt(self.d_model)

"""
Positional Encoding
"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        x = self.dropout(x)
        return x

"""
Label Smoothing
"""

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.00):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size # vocabulary size

        # dim. of true_dist: (batch_size*max_trg_len, vocab_len)
        true_dist = x.data.clone()

        # fill true_dist with "self.smoothing / (self.size - 2)" 
        # -2 as we are not distributing the smoothing mass over
        # the pad token idx and over the ground truth index
        true_dist.fill_(self.smoothing / (self.size - 2))
        
        # change the value at the ground truth index to be "self.confidence"
        true_dist.scatter_(dim=1, index=target.data.unsqueeze(1), value=self.confidence)

        # The pad index does not have prob. Set it to be 0
        true_dist[:, self.padding_idx] = 0       

        # the padding positions are not revalent to the sequence, and does not have prob.
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

"""
Optim wrapper that implements rate.
"""

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
               min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def save_state_dict(self):
        return {
            'inner_optimizer_state_dict': self.optimizer.state_dict(),
            'step': self._step,
            'warmup': self.warmup,
            'factor': self.factor,
            'model_size': self.model_size,
            'rate': self._rate
        }

    def load_state_dict(self, state_dict):
        self._rate = state_dict['rate']
        self._step = state_dict['step']
        # print(state_dict['inner_optimizer_state_dict'])
        self.optimizer.load_state_dict(state_dict['inner_optimizer_state_dict'])