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

    ## older version
    # def predict(self, trg, z, conds, src_mask, trg_mask):
    #     if self.use_cond2dec == True:
    #         outputs = self.model_decode(trg, z, conds, src_mask,
    #                                     trg_mask)[:, 3:, :]
    #     else:
    #         outputs = self.model_decode(trg, z, conds,
    #                                     src_mask, trg_mask)
    #     output_mol = outputs
    #     return 


# class Predictor(object):
#     def __init__(self, decoder, use_cond2dec):
#         self.decoder = decoder
#         self.use_cond2dec = use_cond2dec

#     def predict(self, trg, z, conds, src_mask, trg_mask):
#         if self.use_cond2dec == True:
#             outputs = self.decoder(trg, z, conds, src_mask,
#                                    trg_mask)[:, 3:, :]
#         else:
#             outputs = self.decoder(trg, z, conds,
#                                    src_mask, trg_mask)
#         output_mol = outputs[0]
#         return F.softmax(output_mol, dim=-1)
