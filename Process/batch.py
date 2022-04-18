import numpy as np

import torch
from torch.autograd import Variable
from torchtext.legacy import data


"""
subsequent_mask may use the code in Transformer.
"""

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        # self.src_mask = (src != pad).unsqueeze(-2)

        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]

            # self.trg_mask = self.make_std_mask(self.trg, pad)
            # self.ntokens = (self.trg_y != pad).data.sum()
    
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


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    print("rebatch function:")
    print(batch.src.size(), batch.trg.size(), batch.logP, batch.QED, batch.tPSA)
    return Batch(batch.src, batch.trg, pad_idx)