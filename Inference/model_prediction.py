from itertools import zip_longest
import zipfile
import torch.nn.functional as F


class Predictor(object):
    def __init__(self, use_cond2dec, decode, encode):
        self.use_cond2dec = use_cond2dec
        self.model_decode = decode
        self.model_encode = encode
    
    def encode(self, src, conds, src_mask):
        z, mu, log_var = self.model_encode(src, conds, src_mask)
        return z, mu, log_var

    def predict(self, **kwargs):
        if self.use_cond2dec:
            output_mol = self.model_decode(**kwargs)[:, 3:, :]
        else:
            output_mol = self.model_decode(**kwargs)
        return F.softmax(output_mol, dim=-1)
