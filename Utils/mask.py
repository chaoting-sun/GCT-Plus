import numpy as np

import torch
from torch.autograd import Variable
from torchtext import data


"""
subsequent_mask may use the code in Transformer.
"""


def nopeak_mask(size, nconds, use_cond2dec):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    if use_cond2dec == True:
        cond_mask = np.zeros((1, nconds, nconds))
        cond_mask_upperright = np.ones((1, nconds, size))
        cond_mask_upperright[:, :, 0] = 0
        cond_mask_lowerleft = np.zeros((1, size, nconds))
        upper_mask = np.concatenate([cond_mask, cond_mask_upperright], axis=2)
        lower_mask = np.concatenate([cond_mask_lowerleft, np_mask], axis=2)
        np_mask = np.concatenate([upper_mask, lower_mask], axis=1)
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    return np_mask
    

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

