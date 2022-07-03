import numpy as np
import torch
from torch.autograd import Variable


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

"""
create masks for Transformer
"""
def nopeak_mask(size, cond_dim, use_cond2dec, device=None):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    if use_cond2dec == True:
        cond_mask = np.zeros((1, cond_dim, cond_dim))
        cond_mask_upperright = np.ones((1, cond_dim, size))
        cond_mask_upperright[:, :, 0] = 0 # ??? cannot understand
        cond_mask_lowerleft = np.zeros((1, size, cond_dim))
        upper_mask = np.concatenate([cond_mask, cond_mask_upperright], axis=2)
        lower_mask = np.concatenate([cond_mask_lowerleft, np_mask], axis=2)
        np_mask = np.concatenate([upper_mask, lower_mask], axis=1)
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if device is not None:
        np_mask = np_mask.to(device)
    return np_mask


def create_condition_mask(conditions):
    # (bs, nconds) -> (bs, 1, nconds)    
    cond_mask = torch.unsqueeze(conditions, -2)
    cond_mask = torch.ones_like(cond_mask, dtype=bool)
    return cond_mask


def create_source_mask(source, conditions=None, condition_mask=None):
    # (bs, strlen) -> (bs, 1, strlen)
    source_mask = (source != 0).unsqueeze(-2)
    if conditions is not None:
        if condition_mask is None:
            condition_mask = create_condition_mask(conditions)
        # (bs, 1, strlen+nconds)
        return torch.cat([condition_mask, source_mask], dim=2)
    return source_mask


def create_target_mask(target, condition_mask, use_cond2dec):
    # padding mask
    target_mask = (target != 0).unsqueeze(-2)
    if use_cond2dec == True:
        target_mask = torch.cat([condition_mask, target_mask], dim=2) 
    # sequence mask
    np_mask = nopeak_mask(target.size(1), condition_mask.size(-1), 
                          use_cond2dec, target.get_device())
    return target_mask & np_mask


def create_masks(source, target, condition, use_cond2dec=True):
    condition_mask = create_condition_mask(condition)
    source_mask = create_source_mask(source, condition, condition_mask)
    if target is not None:
        target_mask = create_target_mask(target, condition_mask, use_cond2dec)
    device = source.get_device()
    return source_mask.to(device), target_mask.to(device)