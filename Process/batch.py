import numpy as np

import torch
from torch.autograd import Variable
# from torchtext import data
from torchtext.legacy import data


"""
subsequent_mask may use the code in Transformer.
"""

def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    if opt.use_cond2dec == True:
        cond_mask = np.zeros((1, opt.cond_dim, opt.cond_dim))
        cond_mask_upperright = np.ones((1, opt.cond_dim, size))
        cond_mask_upperright[:, :, 0] = 0
        cond_mask_lowerleft = np.zeros((1, size, opt.cond_dim))
        upper_mask = np.concatenate([cond_mask, cond_mask_upperright], axis=2)
        lower_mask = np.concatenate([cond_mask_lowerleft, np_mask], axis=2)
        np_mask = np.concatenate([upper_mask, lower_mask], axis=1)
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if opt.device == 0:
      np_mask = np_mask.cuda()
    return np_mask
    

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, conds=None, pad=0):
        
        self.src = src
        # self.src_mask = (src != pad).unsqueeze(-2)

        if trg is not None:
            self.trg = trg[:, :-1]  # the input of the model
            self.trg_y = trg[:, 1:] # the expected output

            # self.trg_mask = self.make_std_mask(self.trg, pad)
            # self.ntokens = (self.trg_y != pad).data.sum()
    
        if conds is not None:
            self.conds = conds

    # @staticmethod
    # def make_std_mask(tgt, pad):
    #     "Create a mask to hide padding and future words."
    #     tgt_mask = (tgt != pad).unsqueeze(-2)
    #     tgt_mask = tgt_mask & Variable(
    #         subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    #     return tgt_mask


# patch on Torchtext's batching process that makes it more efficient
# from http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks



class MyIterator(data.Iterator):
    def __init__(self, dataset, batch_size, sort_key, 
                 device, batch_size_fn, train, repeat, shuffle):
        super().__init__(dataset, batch_size, sort_key, 
                         device, batch_size_fn, train, repeat, shuffle)

    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):

                    p_batch = data.batch(sorted(p, key=self.sort_key),
                                         self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


# the dynamic batching causes some problems
# from https://github.com/pytorch/text/issues/250

global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    # print("batch_size_fn:", max(src_elements, tgt_elements))
    return max(src_elements, tgt_elements)


def rebatch(pad_idx, batch, cond_list):
    "Fix order in torchtext to match ours"
    
    if len(cond_list) > 0:
        all_conds = []
        for c in cond_list:
            cond = getattr(batch, c)
            cond = cond.view(-1, 1)
            all_conds.append(cond)
        cond_tensors = torch.cat(all_conds, dim=1)
    else:
        cond_tensors = None

    return Batch(batch.src, batch.trg, cond_tensors, pad_idx)
