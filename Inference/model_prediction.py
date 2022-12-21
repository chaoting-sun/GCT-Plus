from itertools import zip_longest
import zipfile
import torch.nn.functional as F


class Predictor(object):
    def __init__(self, use_cond2dec, decoder, encoder):
        self.use_cond2dec = use_cond2dec
        self.decoder = decoder
        self.encoder = encoder
    
    def encode(self, src, econds, src_mask):
        return self.encoder(src, econds, src_mask)

    def predict(self, trg, z, conds, src_mask, trg_mask):
        if self.use_cond2dec == True:
            outputs = self.decoder(trg, z, conds, src_mask,
                                   trg_mask)[:, 3:, :]
        else:
            outputs = self.decoder(trg, z, conds,
                                   src_mask, trg_mask)
        output_mol = outputs
        return F.softmax(output_mol, dim=-1)


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
