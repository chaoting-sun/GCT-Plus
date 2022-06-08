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
def nopeak_mask(size, cond_dim, device, use_cond2dec):
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
    if device == 0:
      np_mask = np_mask.cuda()
    return np_mask


def create_src_mask(src, cond):
    src_mask = (src != 0).unsqueeze(-2)
    cond_mask = torch.unsqueeze(cond, -2)
    cond_mask = torch.ones_like(cond_mask, dtype=bool)
    src_mask = torch.cat([cond_mask, src_mask], dim=2) # pad mask (cond + smiles)
    return src_mask


def create_trg_mask(trg, cond, use_cond2dec):
    cond_mask = torch.unsqueeze(cond, -2)
    cond_mask = torch.ones_like(cond_mask, dtype=bool)
    # pad mask
    trg_mask = (trg != 0).unsqueeze(-2)
    if use_cond2dec == True:
        trg_mask = torch.cat([cond_mask, trg_mask], dim=2) 
    # seq mask
    np_mask = nopeak_mask(trg.size(1), cond.size(-1), trg.get_device(), use_cond2dec)
    if trg.is_cuda:
        np_mask.cuda()
    # pad + seq mask
    trg_mask = trg_mask & np_mask
    return trg_mask


def create_masks(src, trg, cond, use_cond2dec=True):
    src_mask = (src != 0).unsqueeze(-2)
    cond_mask = torch.unsqueeze(cond, -2)
    cond_mask = torch.ones_like(cond_mask, dtype=bool)
    src_mask = torch.cat([cond_mask, src_mask], dim=2)

    if trg is not None:
        # trg: pad mask (cond + smiles)
        trg_mask = (trg != 0).unsqueeze(-2)
        if use_cond2dec == True:
            trg_mask = torch.cat([cond_mask, trg_mask], dim=2) 
        # trg: seq mask (cond + smiles)
        np_mask = nopeak_mask(trg.size(1), cond.size(-1), src.get_device(), use_cond2dec)
        if trg.is_cuda:
            np_mask.cuda()
        # trg: pad + seq mask (cond + smiles)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask