import torch.nn.functional as F


class ModelPrediction(object):
    def __init__(self, predictor, use_cond2dec):
        self.predictor = predictor
        self.use_cond2dec = use_cond2dec
    
    def predict(self, trg, e_outputs, conds, src_mask, trg_mask):
        if self.use_cond2dec == True:
            output_mol = self.predictor(trg, e_outputs, conds,
                                        src_mask, trg_mask)[:, 3:, :]
        else:
            output_mol = self.predictor(trg, e_outputs, conds, src_mask, trg_mask)
        return F.softmax(output_mol, dim=-1)